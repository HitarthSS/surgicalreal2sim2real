"""Controlled stationary Newton contact diagnosis; no teleop, policy or hardware."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

from newton_scene_adapter import physics_verdict, sha256


def recover_failed_trajectory(native, poses, metadata, points, needle, save_every, dt):
    """Recover only recorded poses after failure; never fabricate future frames."""
    indices = list(range(0, len(poses), save_every))
    if indices[-1] != len(poses)-1:
        indices.append(len(poses)-1)
    threads, needles = [], []
    for index in indices:
        snapshot = SimpleNamespace(body_q=SimpleNamespace(numpy=lambda: poses[index]))
        threads.append(native.rod_centerline(snapshot, metadata['rod_body_ids'], points))
        if metadata['needle_body'] is not None:
            needles.append(native.transform_body_points(snapshot, metadata['needle_body'], needle))
    result = dict(times=np.asarray(indices)*dt, states=np.asarray(threads), initial_centerline=points,
                  rod_bodies=np.asarray(metadata['rod_body_ids']))
    if needles:
        result['needle_states'] = np.asarray(needles)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--tool', choices=('on', 'off'), default='on')
    parser.add_argument('--contacts', choices=('on', 'off'), default='on')
    parser.add_argument('--self-contact', choices=('native', 'off', 'local-filter'), default='native')
    parser.add_argument('--nodes', type=int, default=0, help='0 preserves dense reconstruction; otherwise physics-only resampling')
    parser.add_argument('--save-every', type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.steps <= 2400:
        raise ValueError('Bounded static probe only: 1–2400 steps')
    args.out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('isolated_native_newton', args.native)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    with np.load(args.bundle, allow_pickle=False) as a:
        geometry = {k: a[k].copy() for k in a.files}
    if args.nodes != 0 and not 3 <= args.nodes <= 200:
        raise ValueError('Physics nodes must be 0 or 3–200')
    if not 1 <= args.save_every <= args.steps:
        raise ValueError('Invalid save interval')
    dense = geometry['points_newton_m'].copy()
    physics_points = native.resample_arclength(dense, args.nodes) if args.nodes else dense
    source_s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    physics_s = np.linspace(0., source_s[-1], len(physics_points)) if args.nodes else source_s.copy()
    # Diagnostic prototype values from existing teleop defaults. Not calibrated.
    settings = dict(device='cpu', gravity=(0., 0., 0.), radius=.0005,
                    stretch_stiffness=1000., stretch_damping=20., bend_stiffness=.4,
                    bend_damping=.05, closed=False, pin='none', drive_end='none',
                    ground=False, contacts=args.contacts == 'on', contact_interval=1, contact_alpha=0.,
                    contact_stiffness=100., contact_damping=1., friction=2.,
                    iterations=10, steps=args.steps, dt=1/1200, save_every=args.save_every,
                    needle_density=7800., needle_collision_segments=24, contact_gap_m=.0001)
    config = SimpleNamespace(**settings)
    metadata, all_q, all_qd, contacts_seen, last = {}, [], [], [], []

    contact_detail = []

    def extend(builder, newton, wp, rods, needle_body):
        # Same mass scale as teleop; recorded as an artificial stabilizer.
        for b in rods:
            builder.body_mass[b] *= 200.
            builder.body_inv_mass[b] /= 200.
            builder.body_inertia[b] *= 200.
            builder.body_inv_inertia[b] /= 200.
        shapes = [(i, int(b)) for i, b in enumerate(builder.shape_body) if int(b) in set(map(int, rods))]
        body_order = {int(b): i for i, b in enumerate(rods)}
        excluded = 0
        for i, (a, ba) in enumerate(shapes):
            for b, bb in shapes[i + 1:]:
                ia, ib = sorted((body_order[ba], body_order[bb]))
                # Overlapping caps in a continuous local strand must not repel each other.
                local_distance = physics_s[ib] - physics_s[ia + 1]
                if args.self_contact == 'off' or (args.self_contact == 'local-filter' and local_distance <= 2 * settings['radius']):
                    builder.add_shape_collision_filter_pair(a, b)
                    excluded += 1
        metadata['additional_filtered_pairs'] = excluded
        if args.tool == 'on' and len(geometry['tool_faces']):
            mesh = newton.Mesh(vertices=geometry['tool_vertices_newton_m'].astype(np.float32),
                               indices=geometry['tool_faces'].astype(np.int32).ravel())
            body = builder.add_body(xform=wp.transform_identity(), mass=0.,
                                    is_kinematic=True, label='static_newest_tracked_tool')
            builder.add_shape_mesh(body=body, mesh=mesh, label='native_tool_mesh_collider')
            metadata['tool_body'] = int(body)
        metadata['shape_body'] = [int(b) for b in builder.shape_body]
        metadata['rod_body_ids'] = list(map(int, rods))
        metadata.update(rod_bodies=len(rods), needle_body=None if needle_body is None else int(needle_body),
                        body_count=len(builder.body_q), shape_count=len(builder.shape_type),
                        rod_mass_scale=200., newton_module=newton.__file__, warp_version=wp.__version__,
                        builder_default_gap_m=float(builder.rigid_gap),
                        explicit_shape_gap_m=float(builder.default_shape_cfg.gap))

    def inspect_state(state, step, contacts):
        q, qd = state.body_q.numpy().copy(), state.body_qd.numpy().copy()
        all_q.append(q)
        all_qd.append(qd)
        last[:] = [state]
        count = 0 if contacts is None else int(contacts.rigid_contact_count.numpy()[0])
        contacts_seen.append(count)
        if step in (1, args.steps) and contacts is not None:
            a = contacts.rigid_contact_shape0.numpy()[:count]
            b = contacts.rigid_contact_shape1.numpy()[:count]
            bodies = np.asarray(metadata['shape_body'])
            valid = (a >= 0) & (b >= 0)
            pairs = np.column_stack((bodies[a[valid]], bodies[b[valid]]))
            rods = np.asarray(metadata['rod_body_ids'])
            both = np.isin(pairs[:, 0], rods) & np.isin(pairs[:, 1], rods)
            contact_detail.append(dict(step=step, count=count, rod_rod=int(both.sum()),
                                       rod_other=int((~both).sum()), body_pairs=pairs.tolist()))
        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise RuntimeError('nonfinite state/velocity; stop without reset')
        if np.max(np.linalg.norm(q[:, :3] - all_q[0][:, :3], axis=1)) > .02:
            raise RuntimeError('static probe body displacement exceeds 20 mm')

    original_fds = [os.dup(1), os.dup(2)]
    error, result, t0 = None, {}, time.perf_counter()
    log_path = args.out / 'native_stdout_stderr.txt'
    with log_path.open('w') as log:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
        try:
            needle = geometry['needle_points_newton_m']
            radius = float(geometry['needle_tube_radius_m']) if len(needle) else .00035
            result = native.simulate_newton(physics_points, config,
                                            needle_points=needle if len(needle) else None,
                                            needle_radius=radius,
                                            scene_extension=extend, state_observer=inspect_state)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            import traceback
            traceback.print_exc()
        finally:
            sys.stdout.flush(); sys.stderr.flush()
            for fd, original in zip((1, 2), original_fds):
                os.dup2(original, fd); os.close(original)
    elapsed = time.perf_counter() - t0
    if error and all_q and 'rod_body_ids' in metadata:
        result = recover_failed_trajectory(native, all_q, metadata, physics_points,
                                           geometry['needle_points_newton_m'], args.save_every, config.dt)
    positions = np.array(all_q)[:, :, :3] if all_q else np.empty((0, 0, 3))
    verdict = physics_verdict(positions, log_path.read_text())
    if error:
        verdict['passed'] = False
        verdict['failures'].append(error)
    center_states = result.get('states', np.empty((0, len(physics_points), 3)))
    lengths = np.linalg.norm(np.diff(center_states, axis=1), axis=2).sum(axis=1)
    center_drift = float(np.max(np.linalg.norm(center_states - center_states[0], axis=2))) if len(center_states) else None
    initial_length = float(np.linalg.norm(np.diff(physics_points, axis=0), axis=1).sum())
    length_change = float(np.max(np.abs(lengths - initial_length)) / initial_length) if len(lengths) else None
    needle_states = result.get('needle_states', np.empty((0, 0, 3)))
    needle_drift = float(np.max(np.linalg.norm(needle_states - needle_states[0], axis=2))) if len(needle_states) else 0.
    stationary_passed = bool(verdict['passed'] and not error and center_drift is not None and center_drift <= .0005 and length_change <= .01 and needle_drift <= .0005)
    reset_error = None
    if all_q and last:
        # Exercise native state assignment, including needle/tool, without hiding a failure.
        last[0].body_q.assign(all_q[0]); last[0].body_qd.assign(np.zeros_like(all_qd[0]))
        reset_error = float(np.max(np.abs(last[0].body_q.numpy() - all_q[0])))
        if reset_error != 0:
            verdict['passed'] = False
            verdict['failures'].append('reset_pose_mismatch')
    report = dict(verdict, stationary_diagnostic_passed=stationary_passed,
                  valid_scene=False, training_eligible=False,
                  stationary_checks={'max_centerline_drift_m': center_drift,
                                     'max_needle_centerline_drift_m': needle_drift,
                                     'max_relative_length_change': length_change,
                                     'drift_limit_m': .0005, 'length_limit_fraction': .01,
                                     'duration_seconds': max(0, len(all_q)-1) * config.dt,
                                     'requested_duration_seconds': args.steps * config.dt},
                  recorded_partial_trajectory_preserved=bool(error and all_q),
                  diagnostic_options=vars(args) | {'bundle': str(args.bundle), 'native': str(args.native), 'out': str(args.out)},
                  contact_detail=contact_detail,
                  settings=settings, metadata=metadata, runtime_seconds=elapsed,
                  states_checked=len(all_q), max_contacts_seen=max(contacts_seen, default=0),
                  reset_max_pose_error=reset_error, bundle_sha256=sha256(args.bundle),
                  source_sha256=sha256(args.native), runner_sha256=sha256(__file__),
                  api_cost_usd=0, compute_cost_usd=None,
                  purpose='CPU static import/contact diagnostic; zero gravity; no learned policy',
                  limitations=['No real material calibration or support surface',
                               'Static tool has no validated articulation/action mapping',
                               '20 mm bound detects divergence, not physical fidelity',
                               'Short numerical stability is not grasp or trajectory validation'])
    np.savez_compressed(args.out / 'states.npz', body_q=np.array(all_q), body_qd=np.array(all_qd),
                        contact_count=np.array(contacts_seen), dense_source_points=dense,
                        physics_source_arclength_m=physics_s, **result)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    main()
