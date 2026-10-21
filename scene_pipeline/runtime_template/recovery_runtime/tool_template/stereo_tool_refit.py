"""CPU stereo reinitialization and articulated fitting with distal coverage loss."""
from pathlib import Path
import json,sys,time
import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from geometry_repairs import distal_region,tool_coverage,project

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tool_template'))
from tool_repair import NativeTool,silhouette,LOW,HIGH
from reinitialize import fit_shaft_line,stereo_axis,align_axis

def load(p):
 with np.load(p,allow_pickle=False) as data:return {k:data[k].copy() for k in data.files}

def source_masks(source,name,selected):
 path=Path(selected['mesh'])
 for parent in path.parents:
  if not parent.is_relative_to(source):break
  if all((parent/(s+'_mask.png')).is_file() for s in ['left','right']):
   return {s:cv2.imread(str(parent/(s+'_mask.png')),0)>0 for s in ['left','right']}
 fallback=source/'tools/heuristic'/name
 if not all((fallback/(s+'_mask.png')).is_file() for s in ['left','right']):
  raise ValueError('Missing actual mask evidence; request segmentation')
 return {s:cv2.imread(str(fallback/(s+'_mask.png')),0)>0 for s in ['left','right']}

def refit(native,original,masks,K,baseline,depth,border,max_evaluations=500):
 """Use the same bounded search for both controllers; never alter CAD dimensions."""
 native.validate_source(original)
 q0=original['distal_joints_rad'].copy();pose0=original['frame4_to_camera'].copy()
 shape=(masks['left'].shape[0]//2,masks['left'].shape[1]//2);small=K.copy();small[:2]*=.5
 targets={s:cv2.resize(m.astype('uint8'),(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST).astype(bool) for s,m in masks.items()}
 target_edges={};distal_edges={};distal_targets={};tip_targets={};edge_distances={}
 for s,m in targets.items():
  boundary=m&~cv2.erode(m.astype('uint8'),np.ones((3,3),'uint8')).astype(bool)
  boundary[:2]=False;boundary[-2:]=False;boundary[:,:2]=False;boundary[:,-2:]=False
  y,x=np.nonzero(boundary);target_edges[s]=np.c_[x,y][::max(1,len(x)//100)][:100]
  d=distal_region(m,border);distal_targets[s]=d
  tip_targets[s]=distal_region(m,border,.15)
  edge_distances[s]=cv2.distanceTransform((~boundary).astype('uint8'),cv2.DIST_L2,5)
  y,x=np.nonzero(boundary&d);distal_edges[s]=np.c_[x,y][::max(1,len(x)//80)][:80]
  if len(target_edges[s])<5 or len(distal_edges[s])<5:raise ValueError('No supported distal silhouette')
 shaft=native.body(q0)[native.slices[0]];axis=np.linalg.svd(shaft-shaft.mean(0),full_matrices=False)[2][0]
 body=native.body(q0);anchor=body[native.slices[1]].mean(0)
 if (anchor-shaft.mean(0))@axis<0:axis=-axis
 seeds=[('preserved_pose',pose0,q0)]
 try:
  l,_=fit_shaft_line(masks['left'],border);r,_=fit_shaft_line(masks['right'],border)
  direction,_,condition=stereo_axis(l,r,K,baseline)
  if direction[0]*(1 if border=='left' else -1)<0:direction=-direction
  rotation=align_axis(axis,direction)
  d=distal_region(masks['left'],border,.25);y,x=np.nonzero(d);z=depth[y,x];good=np.isfinite(z)&(z>.03)&(z<.5)
  if good.sum()<20:raise ValueError('Insufficient depth anchors')
  target=np.median(np.c_[(x[good]-K[0,2])*z[good]/K[0,0],(y[good]-K[1,2])*z[good]/K[1,1],z[good]],axis=0)
  for roll in [0.,np.pi/2,np.pi, -np.pi/2]:
   rr=Rotation.from_rotvec(direction*roll).as_matrix()@rotation
   pose=np.eye(4);pose[:3,:3]=rr;pose[:3,3]=target-rr@anchor
   if .03<pose[2,3]<.5:seeds.append(('stereo_shaft_roll_'+str(roll),pose,q0))
  # A failed initialization must not trap all new hypotheses around its old
  # articulation. Add the same neutral physical-joint seed for every instance.
  neutral=np.array([0.,0.,.35,.35]);body_neutral=native.body(neutral)
  shaft_neutral=body_neutral[native.slices[0]];axis_neutral=np.linalg.svd(shaft_neutral-shaft_neutral.mean(0),full_matrices=False)[2][0]
  anchor_neutral=body_neutral[native.slices[1]].mean(0)
  if (anchor_neutral-shaft_neutral.mean(0))@axis_neutral<0:axis_neutral=-axis_neutral
  neutral_rotation=align_axis(axis_neutral,direction)
  for roll in [0.,np.pi/2,np.pi,-np.pi/2]:
   rr=Rotation.from_rotvec(direction*roll).as_matrix()@neutral_rotation
   pose=np.eye(4);pose[:3,:3]=rr;pose[:3,3]=target-rr@anchor_neutral
   if .03<pose[2,3]<.5:seeds.append(('independent_neutral_articulation_'+str(roll),pose,neutral))
 except ValueError:condition=None
 # Seed ranking uses visible surfaces only; final optimization uses rendered
 # silhouettes. Pulling all interior mesh vertices toward a contour can collapse
 # a plausible fit and is not an appropriate surface-fitting objective.
 ids=np.arange(len(body))[::max(1,len(body)//700)][:700]
 distal_ids=np.arange(native.slices[1].start,len(body))[::max(1,(len(body)-native.slices[1].start)//500)][:500]
 def moved(delta,pose,q):
  rr=Rotation.from_rotvec(delta[:3]).as_matrix()@pose[:3,:3];t=pose[:3,3]+delta[3:6]
  qq=np.clip(q+delta[6:],LOW,HIGH);vertices=native.body(qq)@rr.T+t
  result=np.eye(4);result[:3,:3]=rr;result[:3,3]=t
  return vertices,result,qq
 def residual(delta,pose,q):
  vertices,_,_=moved(delta,pose,q);values=[]
  for s in ['left','right']:
   b=baseline if s=='right' else 0.;xy=project(vertices[ids],small,b);tip=project(vertices[distal_ids],small,b)
   values.extend(cKDTree(xy).query(target_edges[s])[0]/4)
   values.extend(cKDTree(tip).query(distal_edges[s])[0]/2)
   # Full projected silhouette, not only proximal overlap. Keep vector length fixed.
   yx=project(vertices[distal_ids],small,b)
   values.extend(cKDTree(target_edges[s]).query(yx)[0]/8)
  values.extend(delta[:3]/np.pi*.04);values.extend(delta[3:6]/.05*.04);values.extend(delta[6:]/np.pi*.04)
  return np.asarray(values)
 def silhouette_cost(delta,pose,q):
  vertices,final,_=moved(delta,pose,q)
  if not .03<=final[2,3]<=.5 or not np.isfinite(vertices).all() or (vertices[native.slices[1].start:,2]<=.005).any():return 1e5
  value=0.
  for s,target in targets.items():
   rendered=silhouette(vertices,native.faces,small,baseline if s=='right' else 0.,shape)
   union=(rendered|target).sum();intersection=(rendered&target).sum()
   distal=distal_targets[s];tip=tip_targets[s]
   value+=1-intersection/max(1,union)
   value+=.8*(1-(rendered&distal).sum()/max(1,distal.sum()))
   value+=.6*(1-(rendered&tip).sum()/max(1,tip.sum()))
   boundary=rendered&~cv2.erode(rendered.astype('uint8'),np.ones((3,3),'uint8')).astype(bool)
   boundary[:2]=False;boundary[-2:]=False;boundary[:,:2]=False;boundary[:,-2:]=False
   if boundary.any():value+=.01*float(np.minimum(edge_distances[s][boundary],40).mean())
  return float(value+.002*np.mean(delta*delta))
 # Include the original among evaluated candidates even if a numerical search
 # or an independent seed becomes worse. All candidates share the same masks.
 seeds=sorted(seeds,key=lambda x:silhouette_cost(np.zeros(10),x[1],x[2]))
 attempts=[]
 for label,pose,q in seeds[:2]:
  bounds=np.r_[np.full(3,np.pi/2),np.full(3,.05),np.full(4,.9)]
  lower=np.maximum(-bounds,np.r_[np.full(6,-np.inf),LOW-q]);upper=np.minimum(bounds,np.r_[np.full(6,np.inf),HIGH-q])
  scales=np.r_[np.ones(3),np.full(3,.02),np.ones(4)];best=[silhouette_cost(np.zeros(10),pose,q),np.zeros(10)]
  def objective(x):
   delta=x*scales;value=silhouette_cost(delta,pose,q)
   if value<best[0]:best[:]=[value,delta.copy()]
   return value
  fit=minimize(objective,np.zeros(10),method='Powell',bounds=list(zip(lower/scales,upper/scales)),
               options=dict(maxfev=max_evaluations,xtol=.002,ftol=.0001))
  vertices,final,qq=moved(best[1],pose,q)
  candidate=dict(vertices=vertices,faces=native.faces,frame4_to_camera=final,distal_joints_rad=qq)
  views={s:tool_coverage(silhouette(vertices,native.faces,K,baseline if s=='right' else 0.,masks[s].shape),masks[s],border) for s in ['left','right']}
  attempts.append(dict(label=label,mesh=candidate,views=views,evaluations=fit.nfev,objective=best[0]))
 old={s:tool_coverage(silhouette(original['vertices'],native.faces,K,baseline if s=='right' else 0.,masks[s].shape),masks[s],border) for s in ['left','right']}
 def quality(views):return min(v['distal_coverage'] for v in views.values())+.5*min(v['iou'] for v in views.values())
 best=max(attempts,key=lambda x:quality(x['views']))
 promoted=quality(best['views'])>quality(old)+.03 and all(best['views'][s]['observed_coverage']>=old[s]['observed_coverage']-.03 and best['views'][s]['iou']>=old[s]['iou']-.02 for s in old)
 promoted=promoted and all(v['iou']>=.35 and v['reconstruction_visible'] for v in best['views'].values())
 selected=best['mesh'] if promoted else original
 report=dict(baseline=old,candidate=best['views'],promoted=bool(promoted),selection='repaired' if promoted else 'preserved_baseline',
             selected_seed=best['label'],attempts=[dict(seed=v['label'],views=v['views'],evaluations=v['evaluations']) for v in attempts],
             native_dimensions_unchanged=True,physical_joint_limits_unchanged=True,stereo_conditioning=condition,
             independent_accuracy_measured=False,physics_validated=False)
 return selected,report
