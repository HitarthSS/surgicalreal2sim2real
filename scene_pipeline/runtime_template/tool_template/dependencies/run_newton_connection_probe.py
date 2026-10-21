"""Stationary and bidirectional controlled-motion tests of a derived connection."""
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
from needle_thread_attachment import world_endpoint_error
from needle_thread_connection_tool import local_contact_pairs


def recover_failed_trajectory(native, poses, metadata, points, needle, save_every, dt, thread_reversed=False):
    """Recover only recorded poses after failure; never fabricate future frames."""
    indices = list(range(0, len(poses), save_every))
    if indices[-1] != len(poses)-1:
        indices.append(len(poses)-1)
    threads, needles = [], []
    for index in indices:
        snapshot = SimpleNamespace(body_q=SimpleNamespace(numpy=lambda: poses[index]))
        model_points = points[::-1] if thread_reversed else points
        line = native.rod_centerline(snapshot, metadata['rod_body_ids'], model_points)
        threads.append(line[::-1] if thread_reversed else line)
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
    parser.add_argument('--attachment-report', type=Path, required=True)
    parser.add_argument('--motion', choices=('rest', 'needle', 'thread'), default='rest')
    parser.add_argument('--attachment', choices=('on', 'off'), default='on')
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--tool', choices=('on', 'off'), default='on')
    parser.add_argument('--contacts', choices=('on', 'off'), default='on')
    parser.add_argument('--self-contact', choices=('local-filter',), default='local-filter')
    parser.add_argument('--nodes', type=int, default=0, help='0 preserves dense reconstruction; otherwise physics-only resampling')
    parser.add_argument('--thread-radius-m', type=float, default=.0005)
    parser.add_argument('--needle-frame', choices=('world', 'centroid'), default='centroid')
    parser.add_argument('--save-every', type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.steps <= 2400:
        raise ValueError('Bounded static probe only: 1–2400 steps')
    args.out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('isolated_native_newton', args.native)
    if args.needle_frame == 'centroid' and 'body_origin = points.mean(axis=0)' not in args.native.read_text():
        raise ValueError('Centroid frame requires matching isolated implementation')
    if args.needle_frame == 'world' and 'body_origin = points.mean(axis=0)' in args.native.read_text():
        raise ValueError('World frame requires original isolated implementation')
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    with np.load(args.bundle, allow_pickle=False) as a:
        geometry = {k: a[k].copy() for k in a.files}
    if args.nodes != 0 and not 3 <= args.nodes <= 200:
        raise ValueError('Physics nodes must be 0 or 3–200')
    if not 1 <= args.save_every <= args.steps:
        raise ValueError('Invalid save interval')
    if args.nodes != 0:
        raise ValueError('Preserve dense source samples for this attachment pilot')
    connection = json.loads(args.attachment_report.read_text())
    if connection.get('scene_bundle_sha256') != sha256(args.bundle):
        raise ValueError('Connection does not describe these exact derived scene bytes')
    proposal = connection['attachment']
    if not proposal['applied']:
        raise ValueError('Attachment proposal rejected: ' + ','.join(proposal['failed_checks']))
    attachment = proposal if args.attachment == 'on' else None
    # Keep body ordering identical in connected and disconnected controls.
    reversed_for_attachment = proposal['thread_endpoint_index'] == 0
    dense = geometry['points_newton_m'].copy()
    physics_points = native.resample_arclength(dense, args.nodes) if args.nodes else dense
    source_s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    physics_s = np.linspace(0., source_s[-1], len(physics_points)) if args.nodes else source_s.copy()
    if not .00005 <= args.thread_radius_m <= .0005:
        raise ValueError('Thread radius sensitivity bounded to .05–.5 mm; uncalibrated')
    # Diagnostic prototype values from existing teleop defaults. Not calibrated.
    settings = dict(device='cpu', gravity=(0., 0., 0.), radius=args.thread_radius_m,
                    stretch_stiffness=1000., stretch_damping=20., bend_stiffness=.4,
                    bend_damping=.05, closed=False, pin='none', drive_end='none',
                    ground=False, contacts=args.contacts == 'on', diagnostic_disable_contacts=args.contacts == 'off', contact_interval=1, contact_alpha=0.,
                    contact_stiffness=100., contact_damping=1., friction=2.,
                    iterations=20, steps=args.steps, dt=1/1200, save_every=args.save_every,
                    needle_density=7800., needle_collision_segments=24, contact_gap_m=.0001,
                    joint_alpha=0., joint_linear_ke=1e5, body_contact_buffer_size=256,
                    force_thread_reversal=reversed_for_attachment)
    config = SimpleNamespace(**settings)
    metadata, all_q, all_qd, contacts_seen, last = {}, [], [], [], []

    contact_detail = []
    endpoint_errors = []
    driven = {}

    def extend(builder, newton, wp, rods, needle_body):
        # Same mass scale as teleop; recorded as an artificial stabilizer.
        for b in rods:
            builder.body_mass[b] *= 200.
            builder.body_inv_mass[b] /= 200.
            builder.body_inertia[b] *= 200.
            builder.body_inv_inertia[b] /= 200.
        model_points = physics_points[::-1] if reversed_for_attachment else physics_points
        rod_pairs, junction, filtering = local_contact_pairs(model_points,settings['radius'],
            float(geometry['needle_tube_radius_m']),settings['contact_gap_m'],proposal['connector_length_m'])
        shapes_by_body = {}
        for i,b in enumerate(builder.shape_body):shapes_by_body.setdefault(int(b),[]).append(i)
        excluded = 0
        for i,j in rod_pairs:
            for a in shapes_by_body[int(rods[i])]:
                for b in shapes_by_body[int(rods[j])]:
                    builder.add_shape_collision_filter_pair(a,b);excluded += 1
        # Same local exclusion geometry in detached controls isolates the joint.
        junction_pairs=[]
        for i in junction:
            for a in shapes_by_body[int(rods[i])]:
                for b in shapes_by_body[int(needle_body)]:
                    builder.add_shape_collision_filter_pair(a,b);junction_pairs.append([a,b])
        metadata.update(additional_filtered_pairs=excluded,local_contact_filter=filtering,
                        junction_filtered_shape_pairs=junction_pairs)
        if args.motion != 'rest':
            body = int(needle_body) if args.motion == 'needle' else int(rods[-1])
            builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
            native.set_body_kinematic(builder,wp,body)
            driven.update(body=body,initial_pose=np.asarray(builder.body_q[body],dtype=float))
            metadata['driven_body']=body
        if args.tool == 'on' and len(geometry['tool_faces']):
            mesh = newton.Mesh(vertices=geometry['tool_vertices_newton_m'].astype(np.float32),
                               indices=geometry['tool_faces'].astype(np.int32).ravel())
            body = builder.add_body(xform=wp.transform_identity(), mass=0.,
                                    is_kinematic=True, label='static_newest_tracked_tool')
            builder.add_shape_mesh(body=body, mesh=mesh, label='native_tool_mesh_collider')
            metadata['tool_body'] = int(body)
        metadata['shape_body'] = [int(b) for b in builder.shape_body]
        metadata['rod_body_ids'] = list(map(int, rods))
        metadata['thread_reversed_for_attachment'] = reversed_for_attachment
        metadata['attachment_joint_ids'] = [i for i,label in enumerate(builder.joint_label) if label == 'needle_thread_attachment']
        metadata['needle_local_origin_m'] = geometry['needle_points_newton_m'].mean(axis=0).tolist() if needle_body is not None and args.needle_frame == 'centroid' else [0., 0., 0.]
        metadata['needle_mass_kg'] = float(builder.body_mass[needle_body]) if needle_body is not None else None
        metadata['needle_com_local_m'] = list(builder.body_com[needle_body]) if needle_body is not None else None
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
        if metadata.get('needle_body') is not None:
            endpoint_errors.append(world_endpoint_error(native, q, metadata['rod_body_ids'],
                metadata['needle_body'], physics_points, geometry['needle_points_newton_m'], proposal,
                thread_reversed=reversed_for_attachment, needle_local_origin=metadata['needle_local_origin_m']))
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
    # Local coupling test: translate one joined body 2mm along +X, with no
    # orientation constraint on its dynamic counterpart. This is not a robot demo.
    def prescribe_motion(state0,state1,step,dt):
        if not driven:return
        t=step*dt
        u=np.clip((t-.1)/.7,0.,1.)
        offset=.002*u*u*(3.-2.*u)
        xyz=driven['initial_pose'][:3]+np.array([offset,0.,0.])
        native.set_body_translation(state0,driven['body'],xyz)
        native.set_body_translation(state1,driven['body'],xyz)
        # State twist is COM linear velocity followed by angular velocity.
        velocity=.002*6*u*(1-u)/.7 if .1<t<.8 else 0.
        for state in (state0,state1):
            qd=state.body_qd.numpy();qd[driven['body'],:]=0.;qd[driven['body'],0]=velocity
            state.body_qd.assign(qd)
    with log_path.open('w') as log:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
        try:
            needle = geometry['needle_points_newton_m']
            radius = float(geometry['needle_tube_radius_m']) if len(needle) else .00035
            result = native.simulate_newton(physics_points, config,
                                            needle_points=needle if len(needle) else None,
                                            needle_radius=radius, needle_thread_attachment=attachment,
                                            scene_extension=extend, state_observer=inspect_state,
                                            step_controller=prescribe_motion)
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
                                           geometry['needle_points_newton_m'] - np.asarray(metadata['needle_local_origin_m']), args.save_every, config.dt, reversed_for_attachment)
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
    after_settle = np.asarray(endpoint_errors)[int(.05/config.dt):]
    attachment_check = dict(applied=bool(attachment), joint_ids=metadata.get('attachment_joint_ids', []),
        initial_endpoint_gap_m=endpoint_errors[0] if endpoint_errors else None,
        final_endpoint_gap_m=endpoint_errors[-1] if endpoint_errors else None,
        max_endpoint_gap_after_50ms_m=float(after_settle.max()) if len(after_settle) else None,
        numerical_constraint_passed=bool(attachment and len(after_settle) and after_settle.max() <= .0001),
        tolerance_m=.0001, force_transfer_tested=args.motion!='rest', physical_swage_validated=False)
    motion_check = None
    if args.motion != 'rest' and len(center_states) and len(needle_states):
        ti,ni=proposal['thread_endpoint_index'],proposal['needle_endpoint_index']
        td=center_states[-1,ti]-center_states[0,ti]
        nd=needle_states[-1,ni]-needle_states[0,ni]
        follower=td if args.motion=='needle' else nd
        completed=(len(all_q)-1)==args.steps
        motion_check=dict(mode=args.motion,commanded_translation_m=.002,
            thread_endpoint_displacement_m=td.tolist(),needle_endpoint_displacement_m=nd.tolist(),
            follower_x_displacement_m=float(follower[0]),
            passed=bool(completed and attachment and endpoint_errors and max(endpoint_errors)<=.0001
                        and follower[0]>=.0018 and np.linalg.norm(follower[1:])<=.0002
                        and verdict['passed'] and length_change is not None and length_change<=.01),
            detached_control=args.attachment=='off',local_junction_test_not_robot_demo=True)
    numerical_passed=bool(attachment_check['numerical_constraint_passed'] and
                           (stationary_passed if args.motion=='rest' else motion_check and motion_check['passed']))
    report = dict(verdict, stationary_diagnostic_passed=stationary_passed,
                  attachment_proposal=proposal, attachment_check=attachment_check,motion_check=motion_check,
                  connection_numerically_validated=numerical_passed,
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
                  attachment_report_sha256=sha256(args.attachment_report),
                  source_sha256=sha256(args.native), runner_sha256=sha256(__file__),
                  api_cost_usd=0, compute_cost_usd=None,
                  purpose='CPU static import/contact diagnostic; zero gravity; no learned policy',
                  limitations=['No real material calibration or support surface',
                               'Static tool has no validated articulation/action mapping',
                               '20 mm bound detects divergence, not physical fidelity',
                               'Short numerical stability is not grasp or trajectory validation'])
    np.savez_compressed(args.out / 'states.npz', body_q=np.array(all_q), body_qd=np.array(all_qd),
                        contact_count=np.array(contacts_seen), attachment_endpoint_error_m=np.array(endpoint_errors), dense_source_points=dense,
                        physics_source_arclength_m=physics_s, **result)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    main()
