"""Propose whole-needle rigid poses at unique, stereo-near thread endpoints.

This generates attachment hypotheses, never deforms an arc or certifies a swage.
The unchanged full-arc checks and a subsequent actual VLM decision must select it.
"""
import copy,hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from fixed_needle_fitter import (inputs,thread_endpoints,load_npz,mask_points,project,
    distance_fields,evaluate,render,save,sha,CATALOG,SIDES)
from rigid_needle_contract import from_pose,audit

LIMITS=dict(max_projected_endpoint_gap_px=8.,max_initial_endpoint_gap_m=.012,
            max_rotation_change_rad=.35,max_pose_evaluations=100)

def pose_at_endpoint(artifact,point,index,rotation_delta):
 T=artifact['T_camera_from_template'];R=Rotation.from_rotvec(rotation_delta).as_matrix()@T[:3,:3]
 local=artifact['template_points_m'][index];center=np.asarray(point)-R@local
 return from_pose(float(artifact['template_radius_m']),float(artifact['template_span_degrees']),R,center)

def augment(case,parts,catalog,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 images,masks,thread,occ,K,depth,b=inputs(case);ends=thread_endpoints(parts);fields=distance_fields(masks,thread)
 observed={s:mask_points(masks[s],192) for s in SIDES};result=copy.deepcopy(catalog);generated=[];diagnostics=[]
 # Each already eligible fixed size/arc remains a preserved candidate.
 for row in catalog['shortlist']:
  if not row['report']['geometry_eligible']:continue
  old=load_npz(row['needle_path']);points=old['points_m'];near=[]
  for e in ends:
   p=np.asarray(e['point'])
   for idx in (0,len(points)-1):
    px={s:float(np.linalg.norm(project(points[idx:idx+1],K,b if s=='right' else 0)[0]-project(p[None],K,b if s=='right' else 0)[0])) for s in SIDES}
    gap=float(np.linalg.norm(p-points[idx]))
    if max(px.values())<=LIMITS['max_projected_endpoint_gap_px'] and gap<=LIMITS['max_initial_endpoint_gap_m']:
     near.append((e,idx,px,gap))
  # Do not guess between multiple plausible thread endpoints or needle ends.
  if len(near)!=1:
   diagnostics.append(dict(candidate=row['candidate_id'],status='no_unique_stereo_near_endpoint',count=len(near)));continue
  e,idx,px,gap=near[0];p=np.asarray(e['point'])
  def residual(q):
   a=pose_at_endpoint(old,p,idx,q);values=[]
   for s in SIDES:
    xy=project(a['points_m'],K,b if s=='right' else 0)
    values.extend(cKDTree(xy).query(observed[s])[0])
   return np.r_[values,np.asarray(q)*.2]
  fit=least_squares(residual,np.zeros(3),bounds=(-LIMITS['max_rotation_change_rad']/np.sqrt(3),LIMITS['max_rotation_change_rad']/np.sqrt(3)),
     diff_step=.001,max_nfev=LIMITS['max_pose_evaluations'],loss='soft_l1',f_scale=2.)
  a=pose_at_endpoint(old,p,idx,fit.x)
  report,inferred=evaluate(a['points_m'],masks,occ,thread,K,b,ends,row['report']['views'],fields=fields)
  contract=audit(a,CATALOG,require_pose=True);assert contract['usable'];report['rigid_contract']=contract
  a.update(inferred=inferred,visible_stereo_support=~inferred,reliability=(~inferred).astype(float),
    tube_radius_m=np.asarray(CATALOG['tube_radius_m']),size_prior_used=np.asarray(True))
  cid=row['candidate_id']+'_endpoint_rigid';path=out/(cid+'.npz');np.savez_compressed(path,**a)
  candidate=dict(candidate_id=cid,needle_path=str(path.resolve()),needle_sha256=sha(path),radius_mm=row['radius_mm'],span_degrees=row['span_degrees'],
     extension_fraction=row['extension_fraction'],report=report,initial_rigid_candidate=row['candidate_id'],
     endpoint_hypothesis=dict(thread_endpoint_id=e['id'],needle_endpoint_index=idx,initial_gap_m=gap,initial_projection_gaps_px=px,
        pose_rotation_change_rad=float(np.linalg.norm(fit.x)),endpoint_position_is_estimated=True,physical_swage_requires_VLM=True))
  generated.append(candidate);diagnostics.append(dict(candidate=cid,report=report,endpoint_hypothesis=candidate['endpoint_hypothesis']))
 result['candidates'].extend(generated);short=[];rows=[]
 # Keep every old geometry in candidates. Show an improved admissible attachment
 # only when both old and new satisfy the original visible-shape gates.
 for old in catalog['shortlist']:
  options=[r for r in generated if r['initial_rigid_candidate']==old['candidate_id'] and r['report']['attachment_eligible']]
  selected=max(options,key=lambda r:r['report']['score']) if options else old
  short.append(selected)
  a=load_npz(selected['needle_path']);path=out/(selected['candidate_id']+'.jpg')
  rows.append(render(images,a['points_m'],parts,K,b,a['inferred'],selected['report']['connection'],
      selected['candidate_id']+' | '+('attachment hypothesis eligible' if selected['report']['attachment_eligible'] else 'attachment unresolved'),path))
  selected['overlay']=str(path.resolve())
 import cv2
 cv2.imwrite(str(out/'catalog.jpg'),np.vstack(rows));result['shortlist']=short;result['catalog_image']=str((out/'catalog.jpg').resolve())
 result['endpoint_pose_refinement']=dict(limits=LIMITS,diagnostics=diagnostics,source_catalog_preserved=True,needle_shape_changed=False,verification_limits_unchanged=True)
 result['state_version']=hashlib.sha256(json.dumps(dict(parent=catalog['state_version'],candidates=[(r['candidate_id'],r['needle_sha256']) for r in result['candidates']]),sort_keys=True).encode()).hexdigest()
 save(out/'catalog.json',result);return result
