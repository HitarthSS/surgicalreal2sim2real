"""Preserve baseline geometry without overwriting independently revised evidence."""
from copy import deepcopy
from pathlib import Path
import hashlib,json

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seed_working_evidence(target,source,out):
 new=target.case['inputs'];old=source.case['inputs']
 for name in ['left','right','K','depth_npy']:
  if sha(new[name])!=sha(old[name]):raise ValueError('Baseline and refinement must share images, calibration and depth')
 names={k for k in new if k.endswith('_mask')}|{k for k in old if k.endswith('_mask')}
 same=all(k in new and k in old and sha(new[k])==sha(old[k]) for k in names)
 if same:
  for name in ['case','version','working_path','routed','last_audit','automatic_keys']:
   setattr(target,name,deepcopy(getattr(source,name)))
 else:
  # Carry proven source-bound occluder observations, but never the old thread or
  # needle masks. A later geometry failure may still return the full old bundle.
  carried={}
  for key in ['left_occluder','right_occluder']:
   if key in old and key not in new:new[key]=old[key];carried[key]=old[key]
  if carried:
   path=Path(out)/'perception_with_baseline_occluders.json';path.write_text(json.dumps(target.case,indent=2)+'\n')
   target.working_path=path;target.version+=1
   target.register_derived(dict(carried,working_case=str(path)))
  target.refresh()
 return dict(same_perception=same,refined_masks_preserved=not same,baseline_geometry_preserved=True)

def needs_perception_feedback(baseline):
 needle=baseline.get('needle',{});thread=baseline.get('thread',{})
 return not (needle.get('accepted') and needle.get('complete_visible_needle') and thread.get('checks_passed') and not baseline.get('sanity_failures'))
