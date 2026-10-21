"""Build a known development handoff and an explicitly synthetic needle fixture."""
import json
from pathlib import Path
import sys

import numpy as np

from newton_scene_adapter import assemble, sha256, transform, CAMERA_TO_NEWTON


def main():
    project = Path(__file__).resolve().parents[3]
    root = project / 'research/ismr_2027/newton_adapter_2026-10-08'
    frame = project / 'research/ismr_2027/agent_feedback_2026-10-08/pilot/frame_000787/agent/call_001'
    latest = project / 'research/ismr_2027/latest5_2026-10-06/hitarth_frame_000787'
    paths = dict(thread_samples=frame / 'backend/shared_thread_samples.npz',
                 thread_result=frame / 'tool_result.json', K=root / 'inputs/K.csv',
                 tool_obj=latest / 'scene/combined_scene_online_tracker.obj',
                 tracker_result=root / 'inputs/tracker_result.npz')
    contract = dict(scene_id='development_frame_787_thread_newest_static_tool', split='development',
                    session_id='dvrk_20260902_130146', units='m', frame='rectified_left_camera',
                    calibration_id='existing_rectified_K_baseline_unverified_physical_setup',
                    baseline_m=.006820275085, inputs={k: str(p) for k, p in paths.items()},
                    input_sha256={k: sha256(p) for k, p in paths.items()}, needle=None,
                    manual_interventions='Existing manually reviewed tool mask/jaw tips and fitted native mesh reused',
                    tool_pose_evidence='Fitted estimate, not independent pose reference')
    arrays, report = assemble(contract, root)
    out = root / 'real_frame_787'
    out.mkdir(exist_ok=False)
    (out / 'contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    (out / 'handoff.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(out / 'scene_bundle.npz', **arrays)
    # Reuse existing tube/OBJ exporter without importing previous physics geometry.
    sys.path.insert(0, str(project / 'integrated_surgical_scene_pipeline/tools'))
    from export_frame_scene_obj import tube_mesh, write_obj
    thread_v, thread_f = tube_mesh(arrays['points_newton_m'], .0003, 12)
    seg_flags = arrays['inferred'][:-1] | arrays['inferred'][1:]
    meshes = []
    for flag, name in ((False, 'visible_curve_support'), (True, 'inferred_bridge')):
        faces = thread_f[:len(seg_flags)*24].reshape(len(seg_flags), 24, 3)[seg_flags == flag].reshape(-1, 3)
        meshes.append((name, 'thread' if not flag else 'needle', thread_v, faces))
    meshes.append(('static_newest_tracked_tool', 'tool', arrays['tool_vertices_newton_m'], arrays['tool_faces']))
    write_obj(out / 'scene_newton_m.obj', meshes)
    # Fixed-size synthetic fixture exercises the existing rigid-needle builder.
    fixture = root / 'synthetic_needle_fixture'
    fixture.mkdir(exist_ok=False)
    theta = np.linspace(0, np.pi, 25)
    local = np.column_stack((.01 * np.cos(theta), .01 * np.sin(theta), np.zeros(25)))
    pose = np.eye(4); pose[:3, 3] = [.05, 0, .15]
    needle_camera = transform(local, pose)
    thread_camera = np.column_stack((np.linspace(-.02, 0, 21), np.full(21, .03), np.full(21, .15)))
    synthetic = dict(points_newton_m=transform(thread_camera, CAMERA_TO_NEWTON),
                     needle_points_newton_m=transform(needle_camera, CAMERA_TO_NEWTON),
                     needle_tube_radius_m=np.array(.00035),
                     tool_vertices_newton_m=np.empty((0, 3)), tool_faces=np.empty((0, 3), dtype=np.int32))
    np.savez_compressed(fixture / 'scene_bundle.npz', **synthetic)
    (fixture / 'contract.json').write_text(json.dumps(dict(scene_id='synthetic_import_fixture',
        evidence='synthetic_fixture', units='m', frame='newton_z_up', needle=dict(local_points_m=local.tolist(),
        T_camera_from_needle=pose.tolist(), tube_radius_m=.00035, curve_radius_m=.01),
        thread='21 collinear synthetic points; no image or real-scene reference',
        physical_validation=False, training_eligible=False), indent=2) + '\n')
    v, f = tube_mesh(synthetic['needle_points_newton_m'], .00035, 12)
    tv, tf = tube_mesh(synthetic['points_newton_m'], .0003, 12)
    write_obj(fixture / 'scene_newton_m.obj', [('synthetic_needle', 'needle', v, f), ('synthetic_thread', 'thread', tv, tf)])
    (root / 'build_provenance.json').write_text(json.dumps(dict(adapter_sha256=sha256(project / 'research/ismr_2027/code/newton_scene_adapter.py'),
        builder_sha256=sha256(__file__), exporter_sha256=sha256(project / 'integrated_surgical_scene_pipeline/tools/export_frame_scene_obj.py'),
        real_report=report, fixture='Synthetic; not a reconstructed or measured real needle'), indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
