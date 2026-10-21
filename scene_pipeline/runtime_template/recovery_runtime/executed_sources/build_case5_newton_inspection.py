"""Package frozen failed case-5 candidates for diagnostics, never strict assembly."""
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from newton_scene_adapter import CAMERA_TO_NEWTON, transform, sha256, centerline
from view_newton_reconstruction import existing_tube_mesh


def main():
    project = Path(__file__).resolve().parents[3]
    research = project / 'research/ismr_2027'
    out = research / 'case5_newton_2026-10-12'
    out.mkdir(exist_ok=False)
    code = out / 'code'; code.mkdir()
    for name in ('build_case5_newton_inspection.py', 'view_case5_newton_inspection.py',
                 'newton_scene_adapter.py', 'view_newton_reconstruction.py',
                 'run_newton_contact_diagnostic.py', 'simulate_newton_connected_thread.py'):
        shutil.copy2(research / 'code' / name, code / name)
    helper = project / 'integrated_surgical_scene_pipeline/tools/export_frame_scene_obj.py'
    shutil.copy2(helper, code / helper.name)
    tube = existing_tube_mesh(helper)
    revision = research / 'complex_refiner_2026-10-12/executed_run/revision4'
    case = 'snapshot_20261009_115511'
    inputs = revision / 'inputs' / case
    shutil.copytree(inputs, out / 'inputs')
    original = research / 'action_cards_2026-10-12/causal_revision_v3_2/executed_run/results' / case
    needle_path = original / 'needle_context/call_001/backend/shared_needle_arc.npz'
    needle = np.load(needle_path, allow_pickle=False)
    needle_points = transform(centerline(needle['points_m']), CAMERA_TO_NEWTON)
    depth = np.load(inputs / 'depth_npy.npy', allow_pickle=False)
    K = np.loadtxt(inputs / 'K.csv', delimiter=',')
    def backproject(y, x):
        z = depth[y, x]
        return np.column_stack(((x-K[0,2])*z/K[0,0], (y-K[1,2])*z/K[1,1], z))
    valid = np.isfinite(depth) & (depth > .03) & (depth < .5)
    # Partial tool surface is visualization context only, never a fitted collider.
    mask = (np.asarray(Image.open(inputs / 'left_occluder.png').convert('L')) > 0) & valid
    y, x = np.mgrid[0:depth.shape[0]:2, 0:depth.shape[1]:2]
    take = mask[y, x]
    ids = np.full(y.shape, -1, dtype=int); ids[take] = np.arange(take.sum())
    vertices = transform(backproject(y[take], x[take]), CAMERA_TO_NEWTON)
    faces = []
    for dy in range(ids.shape[0]-1):
        for dx in range(ids.shape[1]-1):
            a,b,c,d = ids[dy,dx], ids[dy,dx+1], ids[dy+1,dx+1], ids[dy+1,dx]
            for f in ((a,b,c),(a,c,d)):
                if min(f) >= 0:
                    v = vertices[list(f)]
                    if max(np.linalg.norm(v[0]-v[1]), np.linalg.norm(v[1]-v[2]), np.linalg.norm(v[0]-v[2])) <= .002:
                        faces.append(f)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1,3)
    thread_mask = (np.asarray(Image.open(inputs / 'thread_left_mask.png').convert('L')) > 0) & valid
    yy,xx = np.where(thread_mask)
    context = transform(backproject(yy[::3], xx[::3]), CAMERA_TO_NEWTON)
    sources = {
        'fixed': original / 'thread/fixed/call_001',
        'heuristic': revision / 'controls' / case / 'tool_004/call_001',
        'agent': revision / 'agent' / case / 'tool_004/call_001',
    }
    ledger = dict(case_id=case, source_revision='executed v4; before SAM3 follow-up',
                  inspection_only=True, valid_scene=False, training_eligible=False,
                  independent_accuracy_measured=False, metric_calibration='provisional',
                  camera_to_newton=CAMERA_TO_NEWTON.tolist(), methods={})
    common = []
    for name, source in sources.items():
        dest = out / name; dest.mkdir()
        result = json.loads((source / 'tool_result.json').read_text())
        if result.get('accepted') is not False:
            raise ValueError('This explicit rejected-candidate diagnostic is not an accepted-scene exporter')
        shared = np.load(source / 'backend/shared_thread_samples.npz', allow_pickle=False)
        points = transform(centerline(shared['points_m']), CAMERA_TO_NEWTON)
        common.append(points)
        bundle = dict(points_newton_m=points, inferred=shared['inferred'],
                      needle_points_newton_m=needle_points, needle_inferred=needle['inferred'],
                      needle_tube_radius_m=needle['tube_radius_m'],
                      tool_vertices_newton_m=np.empty((0,3)), tool_faces=np.empty((0,3), dtype=np.int32),
                      context_tool_vertices=vertices, context_tool_faces=faces,
                      context_thread_points=context, inspection_only=np.array(True),
                      training_eligible=np.array(False), attachment_applied=np.array(False))
        np.savez_compressed(dest / 'inspection_bundle.npz', **bundle)
        shutil.copy2(source / 'backend/shared_thread_samples.npz', dest / 'source_thread_samples.npz')
        shutil.copy2(source / 'tool_result.json', dest / 'source_tool_result.json')
        shutil.copy2(needle_path, dest / 'source_needle_arc.npz')
        steps = []
        if name != 'fixed':
            controller = revision / ('controls' if name == 'heuristic' else 'agent') / case
            shutil.copytree(controller, dest / 'controller_trace')
            for event_path in sorted(controller.glob('event_*.json')):
                e = json.loads(event_path.read_text())
                steps.append(dict(action=e.get('decision',{}).get('action_id', e.get('action_id')),
                                  resolved_action=e.get('resolved_action'), status=e.get('status')))
        metadata = dict(method=name, reconstruction_accepted=False,
                        verification=result['verification'], controller_steps=steps,
                        tool_fit='missing; partial depth surface is visual context only',
                        attachment='not established', physics='pending diagnostic only',
                        pipeline='stereo -> masks/depth -> estimated needle + rejected thread -> diagnostic Newton',
                        source_sha256={str(p.relative_to(project)): sha256(p) for p in
                                       (source/'backend/shared_thread_samples.npz',source/'tool_result.json',needle_path)},
                        inspection_only=True, training_eligible=False)
        (dest / 'inspection.json').write_text(json.dumps(metadata,indent=2)+'\n')
        ledger['methods'][name] = metadata
        v,f = tube(points,.0005,10)
        inferred = shared['inferred'][:-1] | shared['inferred'][1:]
        meshes = []
        for flag,label in ((False,'input_supported_estimate'),(True,'unsupported_inferred_candidate')):
            sides = f[:len(inferred)*20].reshape(-1,20,3)[inferred == flag].reshape(-1,3)
            meshes.append((label,v,sides))
        nv,nf = tube(needle_points,float(needle['tube_radius_m']),12)
        meshes.extend([('estimated_needle',nv,nf),('partial_tool_depth_surface_VISUAL_ONLY',vertices,faces)])
        colors = [( .15,.8,.25),(1.,.48,.05),(.2,.75,.9),(.55,.57,.60)]
        obj = ['# DIAGNOSTIC ONLY; rejected thread; no verified attachment; units meters','mtllib scene.mtl']
        mtl = []; offset=1
        for (label,vs,fs),color in zip(meshes,colors):
            obj.extend(['o '+label,'usemtl '+label])
            obj.extend('v '+' '.join(f'{c:.10g}' for c in row) for row in vs)
            obj.extend('f '+' '.join(str(int(c)+offset) for c in row) for row in fs)
            offset += len(vs)
            mtl.extend(['newmtl '+label,'Kd '+' '.join(map(str,color)),''])
        (dest/'scene_newton_m.obj').write_text('\n'.join(obj)+'\n')
        (dest/'scene.mtl').write_text('\n'.join(mtl)+'\n')
    bounds = np.vstack(common+[needle_points,vertices,context])
    ledger['common_camera_bounds_m'] = [bounds.min(axis=0).tolist(),bounds.max(axis=0).tolist()]
    ledger['build_source_sha256'] = {p.name:sha256(p) for p in code.iterdir()}
    (out/'pipeline_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    print(out)


if __name__ == '__main__':
    main()
