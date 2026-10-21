"""CPU-only import/static-contact probe; no teleop server, policy or hardware."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.steps <= 100:
        raise ValueError('Bounded static probe only: 1–100 steps')
    args.out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('isolated_native_newton', args.native)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    with np.load(args.bundle, allow_pickle=False) as a:
        geometry = {k: a[k].copy() for k in a.files}
    # Diagnostic prototype values from existing teleop defaults. Not calibrated.
    settings = dict(device='cpu', gravity=(0., 0., 0.), radius=.0005,
                    stretch_stiffness=1000., stretch_damping=20., bend_stiffness=.4,
                    bend_damping=.05, closed=False, pin='none', drive_end='none',
                    ground=False, contacts=True, contact_interval=1, contact_alpha=0.,
                    contact_stiffness=100., contact_damping=1., friction=2.,
                    iterations=10, steps=args.steps, dt=1/1200, save_every=1,
                    needle_density=7800., needle_collision_segments=24, contact_gap_m=.0001)
    config = SimpleNamespace(**settings)
    metadata, all_q, all_qd, contacts_seen, last = {}, [], [], [], []

    def extend(builder, newton, wp, rods, needle_body):
        # Same mass scale as teleop; recorded as an artificial stabilizer.
        for b in rods:
            builder.body_mass[b] *= 200.
            builder.body_inv_mass[b] /= 200.
            builder.body_inertia[b] *= 200.
            builder.body_inv_inertia[b] /= 200.
        if len(geometry['tool_faces']):
            mesh = newton.Mesh(vertices=geometry['tool_vertices_newton_m'].astype(np.float32),
                               indices=geometry['tool_faces'].astype(np.int32).ravel())
            body = builder.add_body(xform=wp.transform_identity(), mass=0.,
                                    is_kinematic=True, label='static_newest_tracked_tool')
            builder.add_shape_mesh(body=body, mesh=mesh, label='native_tool_mesh_collider')
            metadata['tool_body'] = int(body)
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
        contacts_seen.append(0 if contacts is None else int(contacts.rigid_contact_count.numpy()[0]))
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
            result = native.simulate_newton(geometry['points_newton_m'], config,
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
    positions = np.array(all_q)[:, :, :3] if all_q else np.empty((0, 0, 3))
    verdict = physics_verdict(positions, log_path.read_text())
    if error:
        verdict['passed'] = False
        verdict['failures'].append(error)
    reset_error = None
    if all_q and last:
        # Exercise native state assignment, including needle/tool, without hiding a failure.
        last[0].body_q.assign(all_q[0]); last[0].body_qd.assign(np.zeros_like(all_qd[0]))
        reset_error = float(np.max(np.abs(last[0].body_q.numpy() - all_q[0])))
        if reset_error != 0:
            verdict['passed'] = False
            verdict['failures'].append('reset_pose_mismatch')
    report = dict(verdict, settings=settings, metadata=metadata, runtime_seconds=elapsed,
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
                        contact_count=np.array(contacts_seen), **result)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
