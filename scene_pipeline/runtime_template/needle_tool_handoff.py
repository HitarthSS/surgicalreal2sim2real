"""Carry selected tool-mask occlusion into the late rigid needle stage.

An executed but unaccepted needle-mask repair is a separate candidate requiring
new visual selection; it must never silently replace the preserved baseline.
"""
from pathlib import Path
from copy import deepcopy
import json,hashlib
import cv2,numpy as np
from connection_finalize import verified_tool_masks
from run_shared_refiner_v6 import conservative_occluder

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):Path(p).write_text(json.dumps(value,indent=2)+'\n')
def prepare(case,needle,folder,method,out):
 folder=Path(folder);out=Path(out);out.mkdir(parents=True,exist_ok=False)
 candidate=deepcopy(case);requires_visual=False;mask_receipt=None
 # Keep heuristic evidence separate. Only consider an actually executed SAM3
 # proposal after a failed needle, and keep its unaccepted status explicit.
 if method=='agent' and not (needle and Path(needle).is_file()):
  assessments=sorted((folder/'perception').glob('assessment_*/assessment.json'))
  for path in reversed(assessments):
   a=read(path)
   if 'needle' not in a.get('changed_masks',{}) or a.get('sanity_failures') or a.get('needle',{}).get('native_failures'):continue
   proposal=read(a['case_path'])
   if not all(sha(proposal['inputs'][k])==sha(case['inputs'][k]) for k in ('left','right','K','depth_npy')):raise ValueError('Needle proposal changed image/calibration/depth')
   if not all(a['mask_changes'][s]['needle']['after_pixels']>0 for s in ('left','right')):continue
   # Assessment already checks the executed SAM3 receipt and its fingerprints.
   for item in a['original_fingerprints'].values():
    if sha(item['path'])!=item['sha256']:raise ValueError('Perception assessment source changed')
   candidate=deepcopy(case)
   for side in ('left','right'):candidate['inputs']['needle_'+side+'_mask']=proposal['inputs']['needle_'+side+'_mask']
   requires_visual=True;mask_receipt=dict(assessment=str(path),sha256=sha(path),status='executed_mask_candidate_not_preaccepted')
   break
 selected=folder/'tools/complete_tools.json'
 masks,receipts,rejected=verified_tool_masks(candidate,folder,method,read(selected) if selected.exists() else {})
 audits={}
 for side in ('left','right'):
  image=cv2.imread(candidate['inputs'][side],0)
  raw=cv2.imread(candidate['inputs'][side+'_occluder'],0)>0 if candidate['inputs'].get(side+'_occluder') else np.zeros(image.shape,bool)
  for mask in masks[side]:raw|=mask
  needle_mask=cv2.imread(candidate['inputs']['needle_'+side+'_mask'],0);thread=cv2.imread(candidate['inputs']['thread_'+side+'_mask'],0)
  occ,audit=conservative_occluder(raw,needle_mask,thread)
  if occ.sum()>occ.size*.6:raise ValueError('Occlusion exceeds unchanged area bound')
  path=out/(side+'_occluder.png');cv2.imwrite(str(path),occ.astype('uint8')*255)
  candidate['inputs'][side+'_occluder']=str(path.resolve());audits[side]=audit
 save(out/'case.json',candidate)
 save(out/'receipt.json',dict(controller=method,tool_receipts=receipts,rejected_sources=rejected,views=audits,
   perception_candidate=mask_receipt,requires_actual_VLM_selection=requires_visual,occlusion_not_ground_truth=True))
 return candidate,requires_visual,str((out/'case.json').resolve())
