"""Portable complete-stage handoff; missing/deferred geometry remains explicit."""
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import zipfile
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'executed_sources'))
sys.path.insert(0,str(ROOT/'tool_template'))
from newton_scene_adapter import CAMERA_TO_NEWTON,transform
from shared_tool_estimation import project,rasterize
import package_advisor_base as legacy
from output_policy import classify


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def package(folder):
    tools=json.loads((folder/'tools/complete_tools.json').read_text())
    stereo=json.loads((folder/'case.json').read_text())['inputs'];review_rows=[]
    def review_row(title,paths):
        pair=np.hstack([cv2.imread(str(paths[s])) for s in ['left','right']]);label=np.zeros((40,pair.shape[1],3),'uint8')
        cv2.putText(label,title,(10,26),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
        review_rows.append(np.vstack([label,pair]))
    review_row(folder.name+' ORIGINAL FULL STEREO | left / right',stereo)
    for method in ['heuristic','agent']:
        source=ROOT/'export_context'/method/folder.name
        legacy.one(source,method,'next10_multipart_occlusion_v1')
        out=ROOT/'advisor_handoff'/folder.name/method
        previous=json.loads((out/'scene.json').read_text());scene=dict(previous)
        (out/'provenance').mkdir(exist_ok=True)
        shutil.copy2(out/'scene.json',out/'provenance/pre_tool_scene.json')
        shutil.copy2(out/'README.md',out/'provenance/pre_tool_README.md')
        geometry=out/'geometry';geometry.mkdir(exist_ok=True)
        path=geometry/'scene_newton_m.obj'
        obj=path.read_text() if path.exists() else '# Available diagnostic geometry only\n'
        offset=sum(line.startswith('v ') for line in obj.splitlines())+1
        items=[];meshes=[];deferred=[]
        for name in ['tool_A','tool_B']:
            selected=tools.get('tools',{}).get(method,{}).get(name)
            if not selected:
                items.append(dict(visual_instance=name,status='fit_unavailable',accepted_tool_hypothesis=False));deferred.append(name);continue
            with np.load(selected['mesh'],allow_pickle=False) as d:mesh={k:d[k].copy() for k in d.files}
            availability=classify(selected,mesh)
            if not availability['reconstruction_available']:
                items.append(dict(visual_instance=name,status=availability['reconstruction_status'],accepted_tool_hypothesis=False,output_policy=availability));deferred.append(name);continue
            camera=mesh['vertices'];pose=mesh['frame4_to_camera'];world=transform(camera,CAMERA_TO_NEWTON)
            local=np.linalg.solve(pose[:3,:3],(camera-pose[:3,3]).T).T
            target=out/'tools'/name;target.mkdir(parents=True)
            np.savez_compressed(target/'tool_geometry.npz',vertices_camera_m=camera,vertices_newton_m=world,
                vertices_native_frame4_m=local,faces=mesh['faces'],T_camera_from_native_frame4=pose,
                T_newton_from_native_frame4=CAMERA_TO_NEWTON@pose,distal_joints_rad=mesh['distal_joints_rad'])
            save(target/'selection.json',selected)
            for side in ['left','right']:
                shutil.copy2(selected['verification']['feedback_images'][side],target/(side+'_fit_overlay.png'))
                mask=folder/'tools'/method/name/(side+'_final_proposed_mask.png')
                if mask.exists():shutil.copy2(mask,target/(side+'_proposed_mask.png'))
            accepted=selected['accepted_tool_hypothesis']
            items.append(dict(visual_instance=name,status='accepted_hypothesis' if accepted else 'approximate_reconstruction',output_policy=availability,
                accepted_tool_hypothesis=accepted,agent_review_status=selected['agent_review_status'],
                robot_arm_id=None,geometry='tools/'+name+'/tool_geometry.npz',native_frame='Frame4',
                native_joint_order=['Joint5','Joint6','Joint7','Joint8'],independent_accuracy_measured=False,physics_validated=False))
            if not accepted:deferred.append(name)
            obj+='\no '+name+('\n' if accepted else '_APPROXIMATE\n')
            obj+='\n'.join('v '+' '.join(f'{v:.12g}' for v in point) for point in world)+'\n'
            obj+='\n'.join('f '+' '.join(str(int(i)+offset) for i in face) for face in mesh['faces'])+'\n';offset+=len(world)
            meshes.append((name,world,mesh['faces']))
            review_row(method+' '+name+' | '+('accepted hypothesis' if accepted else 'APPROXIMATE FIT'),selected['verification']['feedback_images'])
        path.write_text(obj)
        scene.update(output_policy='tool_best_effort_v2',available_visual_tool_count=len(meshes),approximate_visual_tools=[i['visual_instance'] for i in items if i.get('status')=='approximate_reconstruction'],entire_pipeline_attempted=True)
        scene.update(visual_tools=items,fitted_visible_tools_available=bool(meshes),fitted_robot_tools_available=False,
            accepted_visual_tool_count=sum(i['accepted_tool_hypothesis'] for i in items),
            deferred_visual_tools=deferred,full_pipeline_complete_hypothesis=bool(scene['thread'] and scene['needle'] and not deferred and scene.get('reconstruction_status')=='connected_proposal' and scene.get('all_current_endpoints_assessed')),
            full_arm_mapping_available=False,tool_revision='heuristic baseline + coded orchestrator/perception/native SAM3/refit/verifier specialists',
            simulation_ready=False,physics_validated=False,training_eligible=False)
        bundle_path=out/'newton/scene_bundle.npz'
        if bundle_path.exists():
            with np.load(bundle_path,allow_pickle=False) as d:bundle={k:d[k].copy() for k in d.files}
            for name,vertices,faces in meshes:bundle[name+'_vertices_newton_m']=vertices;bundle[name+'_faces']=faces
            np.savez_compressed(out/'newton/scene_bundle_with_tools.npz',**bundle)
            contract=json.loads((out/'newton/connection_report.json').read_text());anchor=contract['attachment']
            assert np.array_equal(bundle['points_newton_m'][anchor['thread_endpoint_index']],bundle['needle_points_newton_m'][anchor['needle_endpoint_index']])
            contract.update(original_scene_bundle_sha256=contract['scene_bundle_sha256'],scene_bundle_sha256=sha(out/'newton/scene_bundle_with_tools.npz'),deferred_visual_tools=deferred,tool_colliders_validated=False)
            save(out/'newton/connection_with_tools_report.json',contract)
            scene.update(newton_scene_bundle_with_tools='newton/scene_bundle_with_tools.npz',newton_scene_bundle_with_tools_contract='newton/connection_with_tools_report.json')
        save(out/'scene.json',scene)
        save(out/'deferred_visual_tools.json',dict(exclude_as_accepted_colliders=deferred,diagnostic_meshes_retained=True))
        camera_K=np.loadtxt(stereo['K'],delimiter=',');baseline=json.loads((folder/'case.json').read_text())['baseline_m']
        full_paths={};review=out/'review';review.mkdir(exist_ok=True)
        for side in ['left','right']:
            image=cv2.imread(stereo[side]);b=baseline if side=='right' else 0
            for instance,world,faces in meshes:
                camera=transform(world,np.linalg.inv(CAMERA_TO_NEWTON))
                rendered,_=rasterize(camera,faces,camera_K,b,image.shape[:2])
                contours,_=cv2.findContours(rendered.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image,contours,-1,(0,0,255) if instance in deferred else (255,0,255),1)
            for filename,color in [('needle_geometry.npz',(255,255,0)),('thread_curve.npz',(0,255,0))]:
                gp=out/filename
                if not gp.exists():continue
                with np.load(gp,allow_pickle=False) as data:
                    points=data['points_camera_m'];inferred=data['geometry_inferred'] if 'geometry_inferred' in data else np.zeros(len(points),bool)
                pixels=np.rint(project(points,camera_K,b)).astype('int32')
                for i in range(len(pixels)-1):
                    if points[i,2]>.005 and points[i+1,2]>.005:
                        cv2.line(image,tuple(pixels[i]),tuple(pixels[i+1]),(0,165,255) if inferred[i] or inferred[i+1] else color,2)
            p=review/('full_scene_'+side+'.jpg');cv2.imwrite(str(p),image);full_paths[side]=str(p)
        review_row(method+' FULL SCENE | inferred thread orange; approximate tools red',full_paths)
        (out/'README.md').write_text('Load scene.json. Geometry, thread centerline/spline, estimated needle, calibrated camera arrays, fitted native distal tools and poses/joints are included WHEN available. '
            'Usable imperfect tool fits are included as approximate reconstructions. Missing geometry is an explicit component failure. Numerical/visual checks and collider readiness remain separate. '
            'Geometry units are meters at provisional scale. Visual toolA/B are not assigned to PSM1/PSM2; robot base/camera mapping and full-arm VR control remain missing. '
            'No new Newton motion test or physical accuracy validation ran. Original stereo fields are complete. '
            'Use the exact paired scene_bundle_with_tools/connection_with_tools_report when present; inferred needle/thread anchors remain labeled.\n')
        files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='package_verified.json'}
        save(out/'package_verified.json',dict(files=files,controller=method,case_id=folder.name,independent_accuracy_measured=False,physics_validated=False))
    cv2.imwrite(str(folder/'full_pipeline_comparison.jpg'),np.vstack(review_rows),[cv2.IMWRITE_JPEG_QUALITY,88])
    save(folder/'export_complete.json',dict(controller_packages=2,full_stereo_review=str(folder/'full_pipeline_comparison.jpg')))


def finalize():
    manifest=json.loads((ROOT/'job_manifest.json').read_text());rows=[]
    for cid in manifest['cases']:
        name='snapshot_20261015_'+cid
        for method in ['heuristic','agent']:
            folder=ROOT/'advisor_handoff'/name/method
            scene=json.loads((folder/'scene.json').read_text());verified=json.loads((folder/'package_verified.json').read_text())
            assert all(sha(folder/p)==h for p,h in verified['files'].items())
            rows.append(dict(case_id=name,controller=method,complete=scene['full_pipeline_complete_hypothesis'],deferred_tools=scene['deferred_visual_tools'],path=str((folder/'scene.json').relative_to(ROOT/'advisor_handoff'))))
    assets=ROOT/'advisor_handoff/native_model';assets.mkdir(exist_ok=True)
    repo=Path('/home/hitarth/online_dvrk_tracking_clean_extracted/online_dvrk_tracking_clean')
    for name in ['low_res_shaft_multi_cylinder.ply','low_res_logo_low_res_1.ply','low_res_jawright_lowres.ply','low_res_jawleft_lowres.ply']:
        shutil.copy2(repo/'urdfs/dVRK/meshes'/name,assets/name)
    shutil.copy2(repo/'diffcali/eval_dvrk/LND_fk.py',assets/'native_kinematics_reference.py')
    save(ROOT/'advisor_handoff/index.json',dict(scenes=rows,attempted_scenes=len(manifest['cases']),controller_packages=2*len(manifest['cases']),independent_accuracy_measured=False,physics_validated=False))
    (ROOT/'advisor_handoff/README.md').write_text('Ten earlier development same-session development snapshots; heuristic and actual agent. See each scene.json for component status. Partial/deferred results are retained. No full-arm mapping, metric accuracy or Newton dynamics validation.\n')
    with zipfile.ZipFile(ROOT/'advisor_overnight_remaining33_original_schema.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in (ROOT/'advisor_handoff').rglob('*'):
            if path.is_file():archive.write(path,path.relative_to(ROOT))
    with zipfile.ZipFile(ROOT/'advisor_overnight_remaining33_original_schema.zip') as archive:assert archive.testzip() is None
    save(ROOT/'results_summary.json',dict(scenes=rows,packages_verified=2*len(manifest['cases']),archive_sha256=sha(ROOT/'advisor_overnight_remaining33_original_schema.zip'),archive_bytes=(ROOT/'advisor_overnight_remaining33_original_schema.zip').stat().st_size))


if __name__=='__main__':finalize() if sys.argv[1]=='--finalize' else package(Path(sys.argv[1]))
