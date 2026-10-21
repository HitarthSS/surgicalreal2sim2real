"""Inspectable matched reconstruction exports, with no simulation approval."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from complex_case_diagnostics import write_json, project
from newton_scene_adapter import CAMERA_TO_NEWTON, transform, sha256, rigid_transform
from thread_visibility_repairs_v5 import load_parts
from view_newton_reconstruction import existing_tube_mesh


def projection_mask(vertices, faces, K, baseline=0.):
    points=vertices.copy();points[:,0]-=baseline
    xy=np.zeros((len(points),2))
    front=points[:,2]>1e-8
    xy[front]=points[front,:2]/points[front,2,None]*[K[0,0],K[1,1]]+K[:2,2]
    xy=np.clip(xy,-10000,10000)
    image=Image.new('L',(640,480));draw=ImageDraw.Draw(image)
    for face in faces:
        if np.all(front[face]):draw.polygon([tuple(p) for p in xy[face]],fill=255)
    return np.asarray(image)>0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','mesh-helper'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    case=json.loads((root/'case.json').read_text());K=np.loadtxt(case['inputs']['K'],delimiter=',')
    needle_path=root/'common_needle/call_001/backend/shared_needle_arc.npz'
    if not json.loads((root/'needle_result.json').read_text())['accepted']:
        raise ValueError('Needle failed checks; do not silently substitute an arc')
    with np.load(needle_path,allow_pickle=False) as n:
        needle=n['points_m'].copy();needle_radius=float(n['tube_radius_m'])
        needle_inferred=n['inferred'].copy()
    tools=[];tool_reports=[];depth=np.load(case['inputs']['depth_npy'])
    for name in ('tool_A','tool_B'):
        folder=root/'tool_initialization'/name
        mesh_path=folder/'mesh/tool_camera_m.npz'
        with np.load(mesh_path,allow_pickle=False) as m:
            tool={k:m[k].copy() for k in m.files}
        rigid_transform(tool['frame4_to_camera'])
        mask_path=folder/'robot/sam3_tool/tool_mask.png'
        mask=np.asarray(Image.open(mask_path))>0
        fitted=projection_mask(tool['vertices'],tool['faces'],K)
        iou=float((mask&fitted).sum()/(mask|fitted).sum())
        vertices=tool['vertices'];z=vertices[:,2]
        front=z>1e-8;xy=np.full((len(z),2),-1.)
        xy[front]=vertices[front,:2]/z[front,None]*[K[0,0],K[1,1]]+K[:2,2]
        visible=front&(xy[:,0]>=0)&(xy[:,0]<639)&(xy[:,1]>=0)&(xy[:,1]<479)
        pixels=np.rint(xy[visible]).astype(int);estimated_z=z[visible]
        stereo_z=depth[pixels[:,1],pixels[:,0]]
        good=np.isfinite(stereo_z)&(stereo_z>.03)&(stereo_z<.5)
        depth_consistency=dict(samples=int(good.sum()),
            median_mesh_vertex_depth_m=float(np.median(estimated_z[good])),
            median_stereo_depth_m=float(np.median(stereo_z[good])),
            median_absolute_difference_m=float(np.median(abs(estimated_z[good]-stereo_z[good]))),
            metric_kind='projected mesh vertices vs estimated FFS depth; includes front/back surfaces; not independent accuracy')
        tool_reports.append(dict(visual_instance=name,psm_identity=None,
            pose_frame='native Frame 4 to rectified_left_camera',
            frame4_to_camera=tool['frame4_to_camera'].tolist(),
            distal_joints_rad=tool['distal_joints_rad'].tolist(),
            input_mask_iou=iou,depth_consistency=depth_consistency,
            metric_kind='agreement with manually supplied fit input; not independent accuracy',
            source_sha256={str(path):sha256(path) for path in (mesh_path,mask_path,
                folder/'robot/sam3_tool/manual_review.json',folder/'robot/online_tracker/tracker_result.npz')}))
        tools.append(tool)
    tube=existing_tube_mesh(a.mesh_helper)
    panel=Image.new('RGB',(1280,3*520),'white');draw=ImageDraw.Draw(panel)
    for column,side in enumerate(('left','right')):
        with Image.open(case['inputs'][side]) as image:panel.paste(image.convert('RGB'),(column*640,40))
    draw.text((10,12),'Original FULL stereo frames | no assistant crop',fill='black')
    rows=[]
    for row,method in enumerate(('heuristic','agent'),start=1):
        summary_path=root/'controllers'/method/'summary.json';summary=json.loads(summary_path.read_text())
        candidate=summary['candidate'];fragment_path=Path(candidate['fragment_path'])
        if sha256(fragment_path)!=candidate['fingerprints']['fragments']['sha256']:
            raise ValueError('Changed thread proposal')
        parts=load_parts(fragment_path)
        if len(parts)!=1 or not candidate['report']['numerical_checks_passed']:
            raise ValueError('Do not export failed or disconnected fragments as a complete curve')
        part=parts[0];points=part['points_m']
        folder=root/'exports'/method;folder.mkdir(parents=True,exist_ok=False)
        v,f=tube(transform(points,CAMERA_TO_NEWTON),.0005,10)
        nv,nf=tube(transform(needle,CAMERA_TO_NEWTON),needle_radius,12)
        meshes=[('estimated_thread',v,f,(.15,.8,.25)),('estimated_needle',nv,nf,(.2,.75,.9))]
        for name,tool,color in zip(('tool_A','tool_B'),tools,((1.,.55,.15),(.70,.35,.90))):
            meshes.append((name,transform(tool['vertices'],CAMERA_TO_NEWTON),tool['faces'],color))
        obj=['# INSPECTION ONLY; provisional metric scale; no attachment/physics validity','mtllib scene.mtl']
        mtl=[];offset=1
        for name,vertices,faces,color in meshes:
            obj.extend(['o '+name,'usemtl '+name]);obj.extend('v '+' '.join(f'{x:.12g}'for x in v)for v in vertices)
            obj.extend('f '+' '.join(str(int(x)+offset)for x in face)for face in faces);offset+=len(vertices)
            mtl.extend(['newmtl '+name,'Kd '+' '.join(map(str,color)),''])
        (folder/'scene_newton_m.obj').write_text('\n'.join(obj)+'\n')
        (folder/'scene.mtl').write_text('\n'.join(mtl)+'\n')
        bundle=dict(points_newton_m=transform(points,CAMERA_TO_NEWTON),
            K=K,baseline_m=np.array(case['baseline_m']),
            geometry_inferred=part['geometry_inferred'],depth_inferred=part['depth_inferred'],
            needle_points_newton_m=transform(needle,CAMERA_TO_NEWTON),needle_inferred=needle_inferred,
            needle_tube_radius_m=needle_radius,inspection_only=np.array(True),training_eligible=np.array(False))
        for name,tool in zip(('tool_A','tool_B'),tools):
            bundle[name+'_vertices']=transform(tool['vertices'],CAMERA_TO_NEWTON)
            bundle[name+'_faces']=tool['faces']
        np.savez_compressed(folder/'inspection_bundle.npz',**bundle)
        np.savez_compressed(folder/'thread_camera_m.npz',**part)
        metadata=dict(case_id=case['case_id'],method=method,actual_model_used=summary['actual_model_used'],
            action_ids=[r['decision']['action_id'] for r in summary['history'] if 'decision'in r],
            thread_checks=candidate['report'],tools=tool_reports,needle_source_sha256=sha256(needle_path),
            inspection_only=True,simulation_ready=False,training_eligible=False,
            attachment='not established',physical_units_confirmed=False,
            dimensions_unmeasured=['thread radius 0.5mm visualization assumption','needle tube radius assumption'],
            independent_accuracy_measured=False,physics_run=False,
            source_sha256={str(p):sha256(p)for p in (summary_path,fragment_path,needle_path,a.mesh_helper,Path(__file__))})
        write_json(folder/'scene.json',metadata)
        rows.append(dict(method=method,thread_sha256=sha256(fragment_path),
            runtime_seconds=summary['runtime_seconds'],estimated_api_cost_usd=summary.get('estimated_api_cost_usd'),
            actions=metadata['action_ids'],coverage=candidate['report']['visible_mask_coverage']))
        for col,side in enumerate(('left','right')):
            with Image.open(case['inputs'][side]) as image:rgb=np.array(image.convert('RGB'))
            baseline=case['baseline_m'] if side=='right' else 0.
            for tool,color in zip(tools,((255,140,38),(178,89,230))):
                mask=projection_mask(tool['vertices'],tool['faces'],K,baseline)
                rgb[mask]=(rgb[mask]*.65+np.array(color)*.35).astype('uint8')
            im=Image.fromarray(rgb);d=ImageDraw.Draw(im)
            for curve,color in ((needle,(20,205,235)),(points,(30,230,50))):
                d.line([tuple(v)for v in project(curve,K,baseline)],fill=color,width=2)
            im.save(folder/(side+'_overlay.png'));panel.paste(im,(col*640,row*520+40))
        draw.text((10,row*520+10),method.upper()+' | green=thread, cyan=needle, orange/purple=two fitted tool meshes | no verified attachment',fill='black')
    panel.save(root/'comparison.png')
    with np.load(root/'exports/heuristic/thread_camera_m.npz') as h, np.load(root/'exports/agent/thread_camera_m.npz') as v:
        identical=all(np.array_equal(h[key],v[key]) for key in h.files)
    write_json(root/'reconstruction_comparison.json',dict(rows=rows,
        geometry_identical=identical,
        tool_reports=tool_reports,complete_inspection_export=True,simulation_ready=False,
        independent_accuracy_measured=False,training_eligible=False))


if __name__=='__main__':main()
