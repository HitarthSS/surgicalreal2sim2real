"""Export every available estimated object, preserving disconnected fragments.

Read-only scientific inputs; standalone packaging revision. No reconstruction,
model call, calibration change, extra join or physics validation occurs here.
"""
from pathlib import Path
import json,shutil,sys,hashlib,zipfile,re
import numpy as np,cv2
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent
sys.path[:0]=[str(SOURCE/'executed_sources'),str(SOURCE/'tool_template')]
from newton_scene_adapter import CAMERA_TO_NEWTON,transform
from shared_tool_estimation import project,rasterize

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def load(p):
 with np.load(p,allow_pickle=False) as d:return {k:d[k].copy() for k in d.files}
def tube(points,radius):
 p=np.asarray(points,float);p=p[np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-12]]
 if len(p)<2:return np.empty((0,3)),np.empty((0,3),int)
 tangent=np.gradient(p,axis=0);tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-12)
 previous=None;rings=[]
 for point,t in zip(p,tangent):
  n=np.eye(3)[np.argmin(np.abs(t))] if previous is None else previous
  n=n-np.dot(n,t)*t
  if np.linalg.norm(n)<1e-8:n=np.eye(3)[np.argmin(np.abs(t))];n=n-np.dot(n,t)*t
  n/=np.linalg.norm(n);b=np.cross(t,n);previous=n
  theta=np.arange(8)*2*np.pi/8;rings.append(point+radius*(np.cos(theta)[:,None]*n+np.sin(theta)[:,None]*b))
 vertices=np.vstack(rings);faces=[]
 for i in range(len(p)-1):
  for j in range(8):
   a=i*8+j;b=i*8+(j+1)%8;c=(i+1)*8+j;d=(i+1)*8+(j+1)%8;faces.extend([[a,c,b],[b,c,d]])
 return vertices,np.asarray(faces,int)
def object_text(name,vertices,faces,offset,material):
 assert np.isfinite(vertices).all() and faces.min(initial=0)>=0 and faces.max(initial=0)<len(vertices)
 text='\no '+name+'\nusemtl '+material+'\n'+'\n'.join('v '+' '.join(f'{float(v):.12g}' for v in p) for p in vertices)+'\n'+'\n'.join('f '+' '.join(str(int(i)+offset) for i in f) for f in faces)+'\n'
 return text,offset+len(vertices)
def export(short):
 cid='snapshot_20261015_'+short;folder=SOURCE/cid
 assert (folder/'export_complete.json').exists(),'Original export not finished'
 comparison=[];case=json.loads((folder/'case.json').read_text());images={s:cv2.imread(case['inputs'][s]) for s in ['left','right']};K=np.loadtxt(case['inputs']['K'],delimiter=',')
 def row(title,arrays):
  pair=np.hstack([arrays[s] for s in ['left','right']]);label=np.zeros((38,pair.shape[1],3),'uint8');cv2.putText(label,title,(8,26),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),1,cv2.LINE_AA);comparison.append(np.vstack([label,pair]))
 row(cid+' ORIGINAL FULL STEREO | left / right',images);rows=[]
 for method in ['heuristic','agent']:
  original=SOURCE/'advisor_handoff'/cid/method;verified=json.loads((original/'package_verified.json').read_text());assert all(sha(original/n)==h for n,h in verified['files'].items())
  out=ROOT/'advisor_handoff'/cid/method;assert not out.exists(),'Never overwrite a completed or partial export revision'
  shutil.copytree(original,out);(out/'provenance/original_package_verified.json').write_text(json.dumps(verified,indent=2));(out/'package_verified.json').unlink()
  summary=json.loads((folder/method/'summary.json').read_text()) if (folder/method/'summary.json').exists() else {};candidate=summary.get('candidate') or {};scene=json.loads((out/'scene.json').read_text());parts={};metadata=[];unified=dict(T_newton_from_camera=CAMERA_TO_NEWTON);geometry=out/'geometry';geometry.mkdir(exist_ok=True)
  if candidate:
   path=Path(candidate['fragment_path']);assert sha(path)==candidate['fingerprints']['fragments']['sha256']
   raw=load(path);shutil.copy2(path,out/'thread_fragments.npz')
   for key in raw:
    if re.fullmatch(r'f\d+_points_m',key):
     name=key[:-9];points=raw[key];assert points.ndim==2 and points.shape[1]==3 and np.isfinite(points).all();parts[name]=points
     unified[name+'_points_camera_m']=points;unified[name+'_points_newton_m']=transform(points,CAMERA_TO_NEWTON)
     for flag in ['geometry_inferred','depth_inferred','observed_left','observed_right']:
      if name+'_'+flag in raw:unified[name+'_'+flag]=raw[name+'_'+flag]
   scene['thread']=dict(file='thread_fragments.npz',status='connected_estimate' if len(parts)==1 else 'partial_fragments',fragment_count=len(parts),keys='fN_points_m in camera meters; each fragment is ordered independently',original_source_sha256=sha(path),fragment_connections_invented_by_exporter=False,independent_accuracy_measured=False)
  needle=load(out/'needle_geometry.npz') if (out/'needle_geometry.npz').exists() else None
  attachment=summary.get('attachment') or {};connector=None
  if attachment.get('status')=='geometric_attachment_proposal' and attachment.get('artifact'):
   apath=Path(attachment['artifact']);bundle=load(apath);meta=attachment['metadata'];ti=meta['thread_endpoint_index'];ni=meta['needle_endpoint_index']
   assert np.array_equal(bundle['points_newton_m'][ti],bundle['needle_points_newton_m'][ni]),'Shared anchor is not exact'
   target=out/'newton';target.mkdir(exist_ok=True);shutil.copy2(apath,target/'attachment_fragment_bundle.npz')
   contract=dict(attachment=meta,source_fragment_id=attachment['selection']['fragment'],scene_bundle_sha256=sha(apath),source_geometry_sha256=attachment['source_geometry_sha256'],other_fragments_remain_disconnected=len(parts)>1,point_joint_required=True,physics_validated=False,independent_accuracy_measured=False)
   save(target/'attachment_fragment_contract.json',contract);save(target/'attachment_decision.json',attachment)
   scene.update(newton_attachment_fragment_bundle='newton/attachment_fragment_bundle.npz',newton_attachment_fragment_contract='newton/attachment_fragment_contract.json')
   if scene.get('needle'):scene['needle'].update(needle_thread_attachment_established=False,needle_thread_geometric_attachment_proposed=True)
   if meta.get('connector_added'):
    i=meta['inferred_connector_point_index'];neighbor=1 if i==0 else i-1;connector=bundle['points_camera_m'][[neighbor,i]]
    unified['inferred_attachment_connector_camera_m']=connector;unified['inferred_attachment_connector_newton_m']=transform(connector,CAMERA_TO_NEWTON)
  obj='# Provisional meters. Estimated/inferred geometry. No Newton dynamics validation.\nmtllib scene.mtl\n';offset=1;mesh_objects=[];overlays={s:im.copy() for s,im in images.items()}
  def add_curve(name,points,radius,material,color,inferred=None):
   nonlocal obj,offset
   world=transform(points,CAMERA_TO_NEWTON);vertices,faces=tube(world,radius)
   if len(vertices):text,offset=object_text(name,vertices,faces,offset,material);obj+=text;mesh_objects.append(name)
   for side in overlays:
    pixels=np.rint(project(points,K,case['baseline_m'] if side=='right' else 0)).astype(int)
    for i in range(len(points)-1):
     if points[i,2]>.005 and points[i+1,2]>.005:cv2.line(overlays[side],tuple(pixels[i]),tuple(pixels[i+1]),(0,165,255) if inferred is not None and (inferred[i] or inferred[i+1]) else color,2)
  for name,points in sorted(parts.items()):add_curve('thread_fragment_'+name,points,.00015,'thread',(0,255,0),raw.get(name+'_geometry_inferred'))
  if needle:
   add_curve('needle_estimated',needle['points_camera_m'],float(needle['tube_radius_m']),'needle',(255,255,0));unified['needle_points_camera_m']=needle['points_camera_m'];unified['needle_points_newton_m']=transform(needle['points_camera_m'],CAMERA_TO_NEWTON)
  if connector is not None:add_curve('needle_thread_inferred_connector',connector,.00015,'inferred',(0,165,255))
  for item in scene.get('visual_tools',[]):
   if not item.get('geometry'):continue
   tool=load(out/item['geometry']);name=item['visual_instance'];text,offset=object_text(name,tool['vertices_newton_m'],tool['faces'],offset,'tool' if item['accepted_tool_hypothesis'] else 'inferred');obj+=text;mesh_objects.append(name)
   unified[name+'_vertices_newton_m']=tool['vertices_newton_m'];unified[name+'_faces']=tool['faces']
   for side in overlays:
    mask,_=rasterize(tool['vertices_camera_m'],tool['faces'],K,case['baseline_m'] if side=='right' else 0,overlays[side].shape[:2]);contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(overlays[side],contours,-1,(255,0,255) if item['accepted_tool_hypothesis'] else (0,165,255),1)
  (geometry/'scene_newton_m.obj').write_text(obj);(geometry/'scene.mtl').write_text('newmtl thread\nKd 0.2 0.9 0.2\nnewmtl needle\nKd 0.2 0.8 0.9\nnewmtl tool\nKd 0.6 0.6 0.7\nnewmtl inferred\nKd 1.0 0.5 0.1\n')
  np.savez_compressed(out/'newton_all_geometry.npz',**unified)
  scene.update(all_available_geometry_obj='geometry/scene_newton_m.obj',all_available_geometry_npz='newton_all_geometry.npz',partial_thread_preserved=bool(parts),thread_visualization_radius_m=.00015,thread_radius_is_unmeasured_visualization_assumption=True,export_revision='all_available_geometry_v1',physics_validated=False,training_eligible=False)
  save(out/'scene.json',scene);save(out/'geometry/objects.json',dict(objects=mesh_objects,independent_accuracy_measured=False,physics_validated=False,partial_fragments_not_joined=True))
  (out/'README.md').write_text('Open geometry/scene_newton_m.obj to inspect every available estimated object. scene.json and newton_all_geometry.npz carry camera/Newton frames and statuses. Thread fragments are separate, never secretly joined. thread_fragments.npz preserves exact source arrays and uncertainty labels. When present, newton/attachment_fragment_bundle.npz and its exact SHA-bound contract provide the inferred needle attachment to ONE fragment; other fragments remain disconnected. Thread tube radius0.15mm is a visualization assumption, not measured material geometry. Tools are distal articulated estimates, visual IDs are not mapped to robot arms. Units/scale are provisional. No independent accuracy, Newton dynamics or policy eligibility validation.\n')
  for side in overlays:cv2.imwrite(str(out/'review'/('all_geometry_'+side+'.jpg')),overlays[side],[cv2.IMWRITE_JPEG_QUALITY,88])
  row(method+' | fragments '+str(len(parts))+' | green thread, cyan needle, magenta tools, orange inferred',overlays)
  files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='package_verified.json'};save(out/'package_verified.json',dict(files=files,source_package_files_verified=len(verified['files']),source_package_sha256=sha(original/'package_verified.json'),independent_accuracy_measured=False,physics_validated=False))
  rows.append(dict(case=short,method=method,fragments=len(parts),needle=needle is not None,tools=sum(bool(i.get('geometry')) for i in scene.get('visual_tools',[])),inferred_attachment=connector is not None,objects=mesh_objects))
 review=ROOT/'review';review.mkdir(exist_ok=True);cv2.imwrite(str(review/(short+'_all_geometry.jpg')),np.vstack(comparison),[cv2.IMWRITE_JPEG_QUALITY,88]);save(ROOT/(short+'_export.json'),rows);return rows

def finalize():
 cases=json.loads((SOURCE/'job_manifest.json').read_text())['cases'];rows=[]
 for c in cases:
  rows.extend(json.loads((ROOT/(c+'_export.json')).read_text()))
  for method in ['heuristic','agent']:
   p=ROOT/'advisor_handoff'/('snapshot_20261015_'+c)/method;v=json.loads((p/'package_verified.json').read_text());assert all(sha(p/n)==h for n,h in v['files'].items())
 shutil.copytree(SOURCE/'advisor_handoff/native_model',ROOT/'advisor_handoff/native_model')
 shutil.copy2(ROOT/'load_scene.py',ROOT/'advisor_handoff/load_scene.py')
 save(ROOT/'advisor_handoff/index.json',dict(scenes=rows,attempted_cases=len(cases),controller_packages=1*len(cases),independent_accuracy_measured=False,physics_validated=False))
 (ROOT/'advisor_handoff/README.md').write_text('Ten same-session development scenes, heuristic and agent. Includes ALL available thread fragments, needle and distal-tool geometry, with inferred attachment contracts where available. Inspect per-scene failures/status before teleoperation. No independent accuracy or Newton dynamics validation.\n')
 archive=ROOT/'advisor_overnight_remaining33_all_geometry.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for p in (ROOT/'advisor_handoff').rglob('*'):
   if p.is_file():z.write(p,p.relative_to(ROOT))
 with zipfile.ZipFile(archive) as z:assert z.testzip() is None
 save(ROOT/'verification.json',dict(controller_packages_verified=2*len(cases),rows=rows,archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,original_scientific_outputs_unchanged=True,independent_accuracy_measured=False,physics_validated=False))
if __name__=='__main__':finalize() if sys.argv[1]=='--finalize' else export(sys.argv[1])
