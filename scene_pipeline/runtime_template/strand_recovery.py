"""Preserve recoverable 3D strands when a different strand lacks depth anchors."""
from pathlib import Path
import json,hashlib
import cv2,numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from geometry_repairs import project

def lift_strands(case,pipeline,reference,out,paths,audit):
 from thread_visibility_repairs_v5 import lift_ordered_reference_paths,RepairLimits,load_parts,save_parts,overlays,endpoint_table
 limits=RepairLimits();out=Path(out);out.mkdir(parents=True,exist_ok=True)
 parts=[];failures=set();pending=[];pending_arrays={};metrics=[]
 for index,path in enumerate(paths):
  folder=out/('strand_'+str(index))
  try:
   report=lift_ordered_reference_paths(case,pipeline,reference,folder,[path],1,0,limits)
   parts.extend(load_parts(folder/'fragments.npz'));metrics.extend(report['fragments'])
   failures.update(set(report['failures'])-{'visible_fragments_not_fully_represented'})
  except ValueError as error:
   key='strand_'+str(index)+'_path_rc';pending_arrays[key]=path
   pending.append(dict(strand=index,reason=str(error),path_key=key,samples=len(path),endpoints_xy=path[[0,-1],::-1].tolist()))
 if pending_arrays:np.savez_compressed(out/'pending_depth_paths.npz',**pending_arrays)
 if not parts:
  (out/'pending_depth_report.json').write_text(json.dumps(dict(pending_strands=pending,metric3D_available=False,topology_audit=audit),indent=2)+'\n')
  raise ValueError('No strand has sufficient metric depth; 2D paths retained in pending_depth_paths.npz')
 masks={s:cv2.imread(case['inputs']['thread_'+s+'_mask'],0)>0 for s in ['left','right']}
 images={s:cv2.cvtColor(cv2.imread(case['inputs'][s]),cv2.COLOR_BGR2RGB) for s in ['left','right']}
 K=np.loadtxt(case['inputs']['K'],delimiter=',');baseline=case['baseline_m'];coverage={}
 for side,mask in masks.items():
  canvas=np.zeros(mask.shape,'uint8')
  for part in parts:
   xy=project(part['points_m'],K,baseline if side=='right' else 0.)
   cv2.polylines(canvas,[np.rint(xy).astype('int32')],False,1,1)
  sk=skeletonize(mask);d=distance_transform_edt(1-canvas)
  coverage[side]=float((d[sk]<=limits.mask_check_px).mean()) if sk.any() else 0.
 if min(coverage.values())<limits.min_visible_coverage:failures.add('visible_fragments_not_fully_represented')
 if pending:
  failures.add('visible_strands_pending_depth_repair')
 save_parts(out/'fragments.npz',parts);overlays(parts,images,K,baseline,out)
 report=dict(schema='strand_preserving_recovery_v1',fragment_count=len(parts),
  numerical_checks_passed=not failures,failures=sorted(failures),visible_mask_coverage=coverage,
  fragments=metrics,pending_depth_strands=pending,topology_audit=audit,
  physical_crossing_welds_created=False,original_masks_modified=False,
  endpoints=endpoint_table(parts,K,baseline,masks['left'].shape,limits),
  independent_accuracy_measured=False,physics_validated=False)
 if pending:report['pending_depth_artifact']=dict(path=str(out/'pending_depth_paths.npz'),sha256=hashlib.sha256((out/'pending_depth_paths.npz').read_bytes()).hexdigest(),coordinates='original image row,column pixels; not metric3D')
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report
