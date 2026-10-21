"""Inspect saved reconstructed geometry and Newton trajectories with ViewerGL.

Offline playback only: no UDP, teleoperation or robot connection. Green/orange
indicate visible/inferred source support. This camera is for inspection, not a
calibrated policy observation. Failed runs remain viewable and explicitly labeled.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def existing_tube_mesh(path):
    """Reuse only existing pure geometry helpers, without unrelated URDF imports."""
    tree = ast.parse(Path(path).read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('normalize', 'tube_mesh')]
    if {node.name for node in selected} != {'normalize', 'tube_mesh'}:
        raise ValueError('Existing mesh helpers missing')
    namespace = {'np': np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['tube_mesh']


def source_inference(bundle, physical_points):
    dense = bundle['points_newton_m']
    flags = bundle.get('inferred', np.zeros(len(dense), dtype=bool)).astype(bool)
    if len(physical_points) == len(dense) and np.allclose(physical_points, dense):
        return flags[:-1] | flags[1:]
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    target = np.linspace(0., s[-1], len(physical_points))
    # A physics interval is inferred if ANY source interval it covers is inferred.
    result = []
    for a, b in zip(target[:-1], target[1:]):
        overlap = (s[:-1] < b) & (s[1:] > a)
        result.append(bool(np.any((flags[:-1] | flags[1:])[overlap])))
    return np.asarray(result, dtype=bool)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--states', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--mesh-helper', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--frames', type=int, default=61)
    p.add_argument('--fps', type=float, default=30.)
    p.add_argument('--view', choices=('task', 'whole-scene'), default='task')
    args = p.parse_args()
    import cv2
    import newton
    import newton.viewer
    import warp as wp
    if not 2 <= args.frames <= 300 or not .1 <= args.fps <= 120:
        raise ValueError('Invalid bounded playback duration')
    args.out.mkdir(parents=True, exist_ok=False)
    with np.load(args.bundle, allow_pickle=False) as data:
        bundle = {k: data[k].copy() for k in data.files}
    with np.load(args.states, allow_pickle=False) as data:
        states, times = data['states'].copy(), data['times'].copy()
        needle_states = data['needle_states'].copy() if 'needle_states' in data else None
        contact_counts = data['contact_count'].copy() if 'contact_count' in data else np.zeros(len(states))
    if states.ndim != 3 or states.shape[2] != 3 or not np.isfinite(states).all():
        raise ValueError('No finite saved centerlines; inspect failure logs')
    report = json.loads(args.report.read_text())
    tube = existing_tube_mesh(args.mesh_helper)
    inferred_segments = source_inference(bundle, states[0])
    builder = newton.ModelBuilder(gravity=0.)
    model = builder.finalize(device='cuda:0')
    state = model.state()
    viewer = newton.viewer.ViewerGL(width=960, height=720, headless=args.headless)
    viewer.set_model(model)
    points = bundle['points_newton_m']
    if len(bundle['needle_points_newton_m']):
        points = np.vstack([points, bundle['needle_points_newton_m']])
    if args.view == 'whole-scene':
        points = np.vstack([points, bundle['tool_vertices_newton_m']])
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = (lo + hi) / 2
    distance = max(float(np.linalg.norm(hi - lo)), .04) * 1.6
    position = center + np.array([.45, -1., .45]) * distance
    direction = center - position
    yaw = float(np.degrees(np.arctan2(direction[1], direction[0])))
    pitch = float(np.degrees(np.arctan2(direction[2], np.linalg.norm(direction[:2]))))
    viewer.set_camera(pos=wp.vec3(*position), pitch=pitch, yaw=yaw)
    viewer.camera.fov = 38.
    status = 'STATIONARY CHECK PASSED; MATERIAL/TASK UNVALIDATED' if report.get('stationary_diagnostic_passed') else 'FAILED STATIONARY CHECK; DO NOT TRAIN'
    movie = cv2.VideoWriter(str(args.out / 'newton_playback.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (960, 720))
    if not movie.isOpened():
        raise RuntimeError('Video encoder unavailable')
    def mesh(name, vertices, faces, color):
        if not len(faces):
            return
        viewer.log_mesh(name, wp.array(vertices.astype(np.float32), dtype=wp.vec3, device='cuda:0'),
                        wp.array(faces.astype(np.int32).ravel(), dtype=wp.int32, device='cuda:0'),
                        color=color, backface_culling=False)
    def draw(index, frame_time):
        viewer.begin_frame(frame_time)
        viewer.log_state(state)
        mesh('/newest_native_tracker_tool', bundle['tool_vertices_newton_m'], bundle['tool_faces'], (.55, .57, .6))
        # Use the source labels even after simulation; dynamics cannot turn inference into observation.
        v, f = tube(states[index], float(report['settings']['radius']), 10)
        per_segment = 20
        side_faces = f[:(len(states[index])-1)*per_segment]
        for flag, label, color in [(False, 'visible_support', (.15, .8, .25)), (True, 'inferred', (1., .48, .05))]:
            selected = side_faces[np.repeat(inferred_segments == flag, per_segment)]
            mesh('/thread_' + label, v, selected, color)
        if needle_states is not None:
            v, f = tube(needle_states[index], float(bundle['needle_tube_radius_m']), 12)
            hidden = bundle.get('needle_inferred', np.ones(len(needle_states[index]), dtype=bool))
            sides = f[:(len(hidden)-1)*24]
            hidden = hidden[:-1] | hidden[1:]
            for flag, label, color in [(False, 'supported_estimate', (.2, .75, .9)), (True, 'inferred', (1., .48, .05))]:
                mesh('/needle_' + label, v, sides[np.repeat(hidden == flag, 24)], color)
        viewer.log_scalar('/diagnostic/time_s', float(times[index]))
        viewer.log_scalar('/diagnostic/stationary_passed', float(report.get('stationary_diagnostic_passed', False)))
        contact_index = min(int(round(times[index] / report['settings']['dt'])), len(contact_counts)-1)
        viewer.log_scalar('/diagnostic/contact_count', float(contact_counts[contact_index]))
        viewer.end_frame()
    try:
        if args.headless:
            indices = np.rint(np.linspace(0, len(states)-1, args.frames)).astype(int)
            for n, index in enumerate(indices):
                draw(int(index), n / args.fps)
                rgb = viewer.get_frame(render_ui=False).numpy()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                contact_index = min(int(round(times[index] / report['settings']['dt'])), len(contact_counts)-1)
                cv2.rectangle(bgr, (0, 0), (960, 62), (20, 20, 20), -1)
                cv2.putText(bgr, status, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .49, (255,255,255), 1)
                cv2.putText(bgr, f't={times[index]:.4f}s contacts={int(contact_counts[contact_index])} green=visible support orange=inferred | Newton', (12, 49), cv2.FONT_HERSHEY_SIMPLEX, .49, (255,255,255), 1)
                movie.write(bgr)
                if n in (0, len(indices)-1):
                    cv2.imwrite(str(args.out / ('before.png' if n == 0 else 'after.png')), bgr)
        else:
            start, index = time.perf_counter(), 0
            while viewer.is_running():
                if viewer.should_step():
                    index = int((time.perf_counter() - start) * args.fps) % len(states)
                draw(index, float(times[index]))
                time.sleep(1 / args.fps)
    finally:
        movie.release()
        viewer.close()
    provenance = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (args.bundle, args.states, args.report, args.mesh_helper, Path(__file__))}
    (args.out / 'viewer_manifest.json').write_text(json.dumps(dict(renderer='newton.viewer.ViewerGL',
         source_sha256=provenance, calibrated_policy_camera=False, physics_report=report,
         label='diagnostic offline playback', hardware_connection=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
