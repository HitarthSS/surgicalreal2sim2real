def main(run_root):
 """Read-only, portable reconstruction collections with fitted distal joint angles."""
 from pathlib import Path
 import json,hashlib,shutil,re,sys,zipfile,csv
 import numpy as np
 D=Path(run_root).resolve();R=D;OUT=D/'reconstructions';OUT.mkdir(exist_ok=False)
 config=json.loads((D/'config.json').read_text())
 first_day=json.loads((R/'scope.json').read_text())['scenes'][0]['case_id'].split('_')[1]
 sys.path[:0]=[str(R/'gpt'/('day'+first_day)/'executed_sources')]
 from newton_scene_adapter import CAMERA_TO_NEWTON,transform

 def read(p):return json.loads(Path(p).read_text()) if Path(p).exists() else {}
 def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
 def save(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
 def cp(a,b):b=Path(b);b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b)
 def load(p):
  with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
 def tube(p,r=.00015):
  p=np.asarray(p,float);p=p[np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-12]]
  if len(p)<2:return np.empty((0,3)),np.empty((0,3),int)
  t=np.gradient(p,axis=0);t/=np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-12);rings=[];prior=None
  for point,v in zip(p,t):
   n=np.eye(3)[np.argmin(abs(v))] if prior is None else prior;n=n-n.dot(v)*v
   if np.linalg.norm(n)<1e-8:n=np.eye(3)[np.argmin(abs(v))];n=n-n.dot(v)*v
   n/=np.linalg.norm(n);prior=n;b=np.cross(v,n);theta=np.arange(8)*np.pi/4;rings.append(point+r*(np.cos(theta)[:,None]*n+np.sin(theta)[:,None]*b))
  faces=[]
  for i in range(len(p)-1):
   for j in range(8):
    a=i*8+j;b=i*8+(j+1)%8;c=(i+1)*8+j;d=(i+1)*8+(j+1)%8;faces.extend([[a,c,b],[b,c,d]])
  return np.vstack(rings),np.asarray(faces,int)

 def package(provider,row,method,root):
  cid=row['case_id'];F=D/provider/('day'+cid.split('_')[1])/cid;s=read(F/method/'summary.json')
  C=read(F/'case.json');candidate=s.get('candidate') or {};report=candidate.get('report',{});out=root/method/cid;out.mkdir(parents=True);protected={};arrays=dict(T_newton_from_camera=CAMERA_TO_NEWTON,K=np.loadtxt(C['inputs']['K'],delimiter=','),baseline_m=np.asarray(C['baseline_m']));objects=[];obj='mtllib scene.mtl\n';offset=1
  def add(name,v,f,material):
   nonlocal obj,offset
   assert np.isfinite(v).all() and f.min(initial=0)>=0 and f.max(initial=0)<len(v)
   obj+='\no '+name+'\nusemtl '+material+'\n'+'\n'.join('v '+' '.join(f'{float(x):.12g}' for x in p) for p in v)+'\n'+'\n'.join('f '+' '.join(str(int(x)+offset) for x in p) for p in f)+'\n';offset+=len(v);objects.append(name)
  for side in ('left','right'):cp(C['inputs'][side],out/'observations'/(side+'_rect.png'))
  cp(C['inputs']['K'],out/'observations/K.csv')
  raw=Path(C['inputs']['left']).parent
  metadata=C.get('observation_metadata',{})
  for name in ['calibration.json','psm_state.json']:
   info=metadata.get(name);path=Path(info['path']) if info else raw/name
   if path.exists():
    if info:assert sha(path)==info['sha256'],'Observation metadata changed'
    cp(path,out/'observations'/name)
  if not (out/'observations/calibration.json').exists():raise FileNotFoundError('Calibration metadata missing: '+cid)
  save(out/'observations/camera.json',dict(K=arrays['K'].tolist(),baseline_m=C['baseline_m'],T_newton_from_camera=CAMERA_TO_NEWTON.tolist(),units='meters_provisional',image_fields='complete_original_stereo'))
  fragment_count=0
  if candidate:
   path=Path(candidate['fragment_path']);assert sha(path)==candidate['fingerprints']['fragments']['sha256'];protected[str(path)]=sha(path);cp(path,out/'thread_fragments.npz');parts=load(path)
   for k,p in parts.items():
    if not re.fullmatch(r'f\d+_points_m',k):continue
    name=k[:-9];fragment_count+=1;arrays[name+'_points_camera_m']=p;arrays[name+'_points_newton_m']=transform(p,CAMERA_TO_NEWTON)
    for flag in ('geometry_inferred','depth_inferred','observed_left','observed_right'):
     if name+'_'+flag in parts:arrays[name+'_'+flag]=parts[name+'_'+flag]
    v,f=tube(arrays[name+'_points_newton_m']);add('thread_fragment_'+name,v,f,'thread')
    if report.get('numerical_checks_passed') and report.get('fragment_count')==1:
     curve=dict(points_camera_m=p,points_newton_m=arrays[name+'_points_newton_m'],T_newton_from_camera=CAMERA_TO_NEWTON)
     curve.update({flag:parts[name+'_'+flag] for flag in ('geometry_inferred','depth_inferred','observed_left','observed_right') if name+'_'+flag in parts});np.savez_compressed(out/'thread_curve.npz',**curve)
  needle=(s.get('needle_provenance') or {}).get('path');needle_available=False
  if needle and Path(needle).exists():
   protected[needle]=sha(needle);n=load(needle);p=n.get('points_m',n.get('points_camera_m'));assert p is not None;n['points_camera_m']=p;n['points_newton_m']=transform(p,CAMERA_TO_NEWTON);n['T_newton_from_camera']=CAMERA_TO_NEWTON;np.savez_compressed(out/'needle_geometry.npz',**n);arrays['needle_points_camera_m']=p;arrays['needle_points_newton_m']=n['points_newton_m'];v,f=tube(n['points_newton_m'],float(n.get('tube_radius_m',.0003)));add('needle',v,f,'needle');needle_available=True
  att=s.get('attachment') or {};anchor=None
  if att.get('status')=='geometric_attachment_proposal' and att.get('artifact'):
   path=att['artifact'];protected[path]=sha(path);bundle=load(path);m=att['metadata'];ti=m['thread_endpoint_index'];ni=m['needle_endpoint_index'];assert np.array_equal(bundle['points_newton_m'][ti],bundle['needle_points_newton_m'][ni]);assert np.array_equal(bundle['points_camera_m'][ti],bundle['needle_points_camera_m'][ni]);cp(path,out/'newton/attachment_fragment_bundle.npz');anchor=dict(bundle='newton/attachment_fragment_bundle.npz',bundle_sha256=sha(path),metadata=m,other_fragments_remain_disconnected=fragment_count>1,physics_validated=False);save(out/'newton/attachment_contract.json',anchor)
   if m.get('connector_added'):
    i=m['inferred_connector_point_index'];j=1 if i==0 else i-1;v,f=tube(bundle['points_newton_m'][[j,i]]);add('inferred_attachment_connector',v,f,'inferred')
  tools=read(F/'tools/complete_tools.json').get('tools',{}).get(method,{});toolrows=[];jointrows=[]
  for name,t in sorted(tools.items()):
   path=t.get('mesh')
   if not path or not Path(path).exists():continue
   z=load(path);protected[path]=sha(path);v=z['vertices'];faces=z['faces'];pose=z['frame4_to_camera'];q=np.asarray(z['distal_joints_rad'],float).reshape(-1);assert q.shape==(4,) and np.isfinite(q).all();assert pose.shape==(4,4)
   local=np.linalg.solve(pose[:3,:3],(v-pose[:3,3]).T).T;world=transform(v,CAMERA_TO_NEWTON);target=out/'tools'/name;target.mkdir(parents=True)
   np.savez_compressed(target/'tool_geometry.npz',vertices_camera_m=v,vertices_newton_m=world,vertices_native_frame4_m=local,faces=faces,T_camera_from_native_frame4=pose,T_newton_from_native_frame4=CAMERA_TO_NEWTON@pose,distal_joints_rad=q)
   np.savez_compressed(target/'pose_and_joint_angles.npz',distal_joints_rad=q,distal_joints_deg=np.rad2deg(q),joint_names=np.asarray(['Joint5','Joint6','Joint7','Joint8']),T_camera_from_native_frame4=pose,T_newton_from_native_frame4=CAMERA_TO_NEWTON@pose)
   j=dict(joint_names=['Joint5','Joint6','Joint7','Joint8'],distal_joints_rad=q.tolist(),distal_joints_deg=np.rad2deg(q).tolist(),source='estimated_native_tool_fit',visual_instance=name,robot_arm_identity=None,full_arm_joint_angles_available=False);save(target/'joint_angles.json',j);jointrows.append(dict(tool=name,**j));accepted=bool(t.get('accepted_tool_hypothesis',t.get('verification',{}).get('accepted_numerical_hypothesis',False)));add(name,world,faces,'tool' if accepted else 'inferred')
   toolrows.append(dict(visual_instance=name,geometry=f'tools/{name}/tool_geometry.npz',pose_and_joint_angles=f'tools/{name}/pose_and_joint_angles.npz',joint_angles_json=f'tools/{name}/joint_angles.json',accepted_numerical_hypothesis=accepted,robot_arm_identity=None))
   arrays[name+'_vertices_newton_m']=world;arrays[name+'_faces']=faces;arrays[name+'_distal_joints_rad']=q;arrays[name+'_T_camera_from_native_frame4']=pose
  save(out/'joint_angles.json',dict(fitted_tools=jointrows,recorded_robot_states='observations/psm_state.json' if (out/'observations/psm_state.json').exists() else None,recorded_base_joint_states_are_separate_from_visual_fits=True,full_arm_mapping_available=False))
  np.savez_compressed(out/'newton_all_geometry.npz',**arrays);(out/'geometry').mkdir(exist_ok=True);(out/'geometry/scene_newton_m.obj').write_text(obj);(out/'geometry/scene.mtl').write_text('newmtl thread\nKd 0.2 0.9 0.2\nnewmtl needle\nKd 0.2 0.8 0.9\nnewmtl tool\nKd 0.7 0.3 0.8\nnewmtl inferred\nKd 1.0 0.5 0.1\n')
  cp(D/'review'/method/(cid+'.jpg'),out/'review/overlay.jpg')
  for name in ('history.json','endpoint_ledger.json'):
   if (F/method/name).exists():cp(F/method/name,out/'diagnostics'/name)
  save(out/'diagnostics/thread_report.json',report);save(out/'provenance.json',dict(source_summary_sha256=sha(F/method/'summary.json') if (F/method/'summary.json').exists() else None,source_geometry_hashes=protected,scientific_outputs_modified=False,source_run=D.name))
  scene=dict(schema='reconstruction_collection_scene_v1',scene_id=cid,provider=provider,controller=method,difficulty=row['difficulty'],partition=row.get('partition','development'),thread=dict(fragment_count=fragment_count,checks_passed=bool(report.get('numerical_checks_passed')),assembly_complete=fragment_count==1 and bool(report.get('numerical_checks_passed')),fragments='thread_fragments.npz' if candidate else None,connected_curve='thread_curve.npz' if (out/'thread_curve.npz').exists() else None),needle='needle_geometry.npz' if needle_available else None,visual_tools=toolrows,attachment_contract='newton/attachment_contract.json' if anchor else None,all_geometry='newton_all_geometry.npz',inspection_obj='geometry/scene_newton_m.obj',joint_angles='joint_angles.json',units='meters_provisional',T_newton_from_camera=CAMERA_TO_NEWTON.tolist(),independent_accuracy_measured=False,physics_validated=False,full_arm_mapping_available=False)
  save(out/'scene.json',scene);(out/'README.md').write_text('Start with scene.json. OBJ includes all available estimated thread fragments, rigid needle and distal tools. Fragments remain separate. Orange/inferred geometry is a hypothesis. Distal tool joint angles are in joint_angles.json and tools/tool_A or tool_B/pose_and_joint_angles.npz (Joint5,Joint6,Joint7,Joint8; radians and degrees). Fitted visual tool IDs are not assigned to robot arms. Recorded robot states, when available, are separate observations. No full-arm camera-to-robot mapping or Newton motion validation is supplied. Thread visualization radius0.15mm is an unmeasured assumption. Partial and failed reconstructions remain indexed.\n')
  assert all(sha(p)==h for p,h in protected.items());files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()};save(out/'package_verified.json',dict(files=files,source_geometry_unchanged=True));return dict(scene_id=cid,controller=method,difficulty=row['difficulty'],thread_fragments=fragment_count,thread_checked=bool(report.get('numerical_checks_passed')),needle_available=needle_available,tools=len(toolrows),tools_with_joint_angles=len(jointrows),path=str((out/'scene.json').relative_to(root)))

 rows=read(R/'scope.json')['scenes'];result=[]
 for provider,label in [('gpt','GPT61')]:
  name=f'{label}_Reconstructions';root=OUT/name;root.mkdir();index=[]
  for row in rows:
   for method in ('heuristic','agent'):
    try:index.append(package(provider,row,method,root))
    except Exception as error:
     failures=read(OUT/'packaging_failures.json') or []
     failures.append(dict(case_id=row['case_id'],method=method,error=type(error).__name__+': '+str(error)));save(OUT/'packaging_failures.json',failures)
  native=root/'native_model';native.mkdir();assets=Path(config['native_tracker'])
  for n in ['low_res_shaft_multi_cylinder.ply','low_res_logo_low_res_1.ply','low_res_jawright_lowres.ply','low_res_jawleft_lowres.ply']:cp(assets/'urdfs/dVRK/meshes'/n,native/n)
  cp(assets/'diffcali/eval_dvrk/LND_fk.py',native/'native_kinematics_reference.py')
  save(root/'index.json',dict(scenes=index,acquisition_scenes=len(rows),controller_packages=len(index),joint_units=['radians','degrees'],joint_order=['Joint5','Joint6','Joint7','Joint8'],independent_accuracy_measured=False,physics_validated=False))
  with (root/'index.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(index[0]));w.writeheader();w.writerows(index)
  (root/'README.md').write_text(f'{label}:{len(rows)} development scenes. Agent and heuristic folders are separate. Start with index.csv, then scene.json. Every fitted tool has distal joint angles in JSON and NPZ (Joint5–Joint8,radians/degrees), plus fitted native Frame4 camera/Newton pose. These are estimated distal angles, not a calibrated full robot arm. Original observation robot states are included when available. Complete NPZ arrays and per-scene OBJ inspection meshes include disconnected fragments. No new inference or geometry changes were made while packaging. Source run failures and baseline selections remain preserved; valid geometry checks do not establish independent accuracy or physics validity.\n')
  cp(D/'review/agent_grid.jpg',root/'agent_grid.jpg')
  verified=0;joints=0
  for row in index:
   folder=root/Path(row['path']).parent;v=read(folder/'package_verified.json');assert all(sha(folder/n)==h for n,h in v['files'].items());verified+=1
   for p in folder.glob('tools/*/pose_and_joint_angles.npz'):
    z=load(p);assert z['distal_joints_rad'].shape==(4,);assert np.allclose(z['distal_joints_deg'],np.rad2deg(z['distal_joints_rad']));joints+=1
  archive=OUT/(name+'.zip')
  with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,strict_timestamps=False) as z:
   for p in root.rglob('*'):
    if p.is_file():z.write(p,p.relative_to(OUT))
  with zipfile.ZipFile(archive) as z:assert z.testzip() is None;assert not any('advisor' in n.lower() for n in z.namelist())
  receipt=dict(provider=provider,archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),scenes=len(rows),controller_packages_verified=verified,fitted_tool_joint_angle_files_verified=joints,all_tool_meshes_have_angles=sum(r['tools'] for r in index)==joints,expected_controller_packages=2*len(rows),complete=len(index)==2*len(rows),zip_integrity=True,scientific_outputs_unchanged=True);result.append(receipt);save(OUT/(name+'_verification.json'),receipt)
 save(OUT/'verification.json',result);print(json.dumps(result))
