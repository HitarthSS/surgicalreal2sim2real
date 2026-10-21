"""Fresh checked geometry for one known development frame, no model or training."""
import argparse
import json
from pathlib import Path

import numpy as np

from newton_scene_adapter import assemble, sha256
from reconstruction_tools import DEFAULT_TOOLS, ReconstructionSession
from view_newton_reconstruction import existing_tube_mesh


def export_obj(path, arrays, mesh_helper, thread_radius_m):
    tube = existing_tube_mesh(mesh_helper)
    vertices, groups, offset = [], [], 0
    def add(name, v, f):
        nonlocal offset
        vertices.extend("v " + " ".join(f"{x:.12g}" for x in point) for point in v)
        groups.append("o " + name)
        groups.extend("f " + " ".join(str(int(i)+offset+1) for i in face) for face in f)
        offset += len(v)
    def labeled_curve(points, inferred, radius, prefix):
        v, f = tube(points, radius, 12)
        flags = inferred[:-1] | inferred[1:]
        side = f[:len(flags)*24].reshape(len(flags), 24, 3)
        for flag, name in ((False, "supported_estimate"), (True, "inferred")):
            add(prefix + "_" + name, v, side[flags == flag].reshape(-1, 3))
    labeled_curve(arrays['points_newton_m'], arrays['inferred'], thread_radius_m, 'thread')
    labeled_curve(arrays['needle_points_newton_m'], arrays['needle_inferred'],
                  float(arrays['needle_tube_radius_m']), 'needle')
    add('newest_static_tracked_tool', arrays['tool_vertices_newton_m'], arrays['tool_faces'])
    path.write_text('# Newton z-up; meters; estimated geometry; prototype tube radii unmeasured.\n'
                    + '\n'.join(vertices + groups) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[3]
    research = project / 'research/ismr_2027'
    args.out.mkdir(parents=True, exist_ok=False)
    case = research / 'development_2026-10-09/inputs/frame_000787/case.json'
    results = {}
    for target, operation in (('thread', 'thread_occlusion'), ('needle', 'needle_arc')):
        session = ReconstructionSession(case, args.out / target, 'heuristic', 'recorded_defaults_no_model',
                                        DEFAULT_TOOLS, max_calls=1)
        result = session.call({'tool': operation, 'parameters': {},
                               'reason': 'Re-run existing shared operation on the known development case; no model used.'})
        results[target] = result
        if not result['accepted']:
            raise ValueError('Shared target rejected: ' + target)
    inputs = dict(thread_samples=args.out / 'thread/call_001/backend/shared_thread_samples.npz',
                  thread_result=args.out / 'thread/call_001/tool_result.json',
                  needle_samples=args.out / 'needle/call_001/backend/shared_needle_arc.npz',
                  needle_result=args.out / 'needle/call_001/tool_result.json',
                  K=case.parent / 'K.csv',
                  tool_obj=research / 'latest5_2026-10-06/hitarth_frame_000787/scene/combined_scene_online_tracker.obj',
                  tracker_result=research / 'newton_adapter_2026-10-08/inputs/tracker_result.npz')
    contract = dict(scene_id='development_787_current_checked_artifacts', session_id='dvrk_20260902_130146',
                    split='development', units='m', frame='rectified_left_camera',
                    calibration_id='saved_K_baseline_physical_scale_unverified', baseline_m=.006820275085,
                    inputs={key: str(path.resolve()) for key, path in inputs.items()},
                    input_sha256={key: sha256(path) for key, path in inputs.items()},
                    needle={'evidence': 'shared_verified_free_radius_arc'},
                    manual_interventions='Reuse original reviewed tool mask/jaw tips and native tracker fit',
                    tool_pose_evidence='Fitted estimate; not independent ground truth')
    arrays, report = assemble(contract, project)
    scene = args.out / 'scene'; scene.mkdir()
    np.savez_compressed(scene / 'scene_bundle.npz', **arrays)
    (scene / 'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    (scene / 'handoff.json').write_text(json.dumps(report, indent=2)+'\n')
    export_obj(scene / 'scene_newton_m.obj', arrays,
               project / 'integrated_surgical_scene_pipeline/tools/export_frame_scene_obj.py', .0005)
    summary = dict(handoff=report, results=results, actual_model_used=False,
                   production_baseline_run_in_this_builder=False,
                   prototype_thread_radius_m=.0005, physical_dimensions_verified=False,
                   attachment_state='unknown; no attachment constraint invented',
                   source_sha256={str(p): sha256(p) for p in (Path(__file__),
                       Path(__file__).with_name('newton_scene_adapter.py'),
                       project / 'integrated_surgical_scene_pipeline/tools/export_frame_scene_obj.py')})
    (args.out / 'reconstruction_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({'scene': str(scene), 'thread_passed': True, 'needle_passed': True,
                      'training_eligible': False}))


if __name__ == '__main__':
    main()
