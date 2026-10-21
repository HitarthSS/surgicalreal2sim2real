"""Bounded runtime contracts for tool routing, perception and stereo verification."""
import json,hashlib,math
from native_keypoint_contract import validate_view
TOOLS=('tool_A','tool_B');SIDES=('left','right')
PROMPTS=('surgical grasper','robotic surgical grasper','surgical instrument')
MAX_CALLS=6

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def envelope(value,state,fields):
 if not isinstance(value,dict) or set(value)!=set(fields)|{'scene_version','evidence_ids','reason'}:raise ValueError('Exact role response fields required')
 if value['scene_version']!=state['scene_version']:raise ValueError('Stale evidence version')
 ids=value['evidence_ids']
 if not isinstance(ids,list) or not {'image.left','image.right'}<=set(ids)<=set(state['evidence_ids']):raise ValueError('Both full original stereo views must be cited')
 if not isinstance(value['reason'],str) or not 1<=len(value['reason'])<=600:raise ValueError('Concise reason required')
 return value

def route(value,state):
 envelope(value,state,['tasks']);rows=value['tasks']
 if not isinstance(rows,list) or len(rows)!=2 or {r.get('instance_id') for r in rows}!=set(TOOLS):raise ValueError('Assess both visible IDs, including missing baseline tools')
 for r in rows:
  if set(r)!={'instance_id','action','gross_mismatch','reason'} or r['action'] not in ('resegment_and_refit','keep_baseline'):raise ValueError('Invalid routing action')
  if type(r['gross_mismatch']) is not bool:raise ValueError('gross_mismatch must be boolean')
  item=state['tools'][r['instance_id']]
  if (r['gross_mismatch'] or not item['baseline_available'] or item.get('repair_required',False)) and r['action']!='resegment_and_refit':raise ValueError('Missing, offscreen or visibly mismatched tool requires an actual repair attempt')
 return value

def perception(value,state):
 envelope(value,state,['instance_id','views'])
 if value['instance_id']!=state['instance_id'] or set(value['views'])!=set(SIDES):raise ValueError('Wrong instance or missing stereo view')
 for side,p in value['views'].items():
  if set(p)!={'prompt','confidence_threshold','boxes','jaw_tips_xy','jaw_tip_visibility','jaw_tip_reason'} or p['prompt'] not in PROMPTS or p['confidence_threshold'] not in (.1,.2,.35):raise ValueError('Unknown segmentation parameters')
  if not isinstance(p['boxes'],list) or not 1<=len(p['boxes'])<=4 or not any(b.get('positive') is True for b in p['boxes']):raise ValueError('At least one positive target box required')
  w,h=state['image_sizes'][side]
  for b in p['boxes']:
   if set(b)!={'xyxy','positive'} or type(b['positive']) is not bool:raise ValueError('Invalid box')
   x=b['xyxy']
   if not isinstance(x,list) or len(x)!=4 or any(type(v) not in (float,int) or not math.isfinite(v) for v in x):raise ValueError('Finite box coordinates required')
   if not 0<=x[0]<x[2]<=w or not 0<=x[1]<x[3]<=h or min(x[2]-x[0],x[3]-x[1])<2:raise ValueError('Box outside full original image')
  validate_view(p,state['image_sizes'][side])
  tips=p['jaw_tips_xy']
  if not isinstance(tips,list) or len(tips) not in (0,2):raise ValueError('Two observed tips or []; never require invented tips')
  for tip in tips:
   if len(tip)!=2 or any(type(v) not in (float,int) or not math.isfinite(v) for v in tip) or not 0<=tip[0]<w or not 0<=tip[1]<h:raise ValueError('Invalid tip coordinate')
 return value

def review(value,state):
 envelope(value,state,['instance_id','selection','mask_supported','candidate_better','catastrophic_failure'])
 if value['instance_id']!=state['instance_id'] or value['selection'] not in ('candidate','baseline'):raise ValueError('Invalid review selection')
 for key in ['mask_supported','candidate_better','catastrophic_failure']:
  if set(value[key])!=set(SIDES) or any(type(v) is not bool for v in value[key].values()):raise ValueError('Both-view boolean review required')
 if value['selection']=='candidate' and (not all(value['mask_supported'].values()) or not all(value['candidate_better'].values()) or any(value['catastrophic_failure'].values())):raise ValueError('Candidate needs supported masks and improvement in both views')
 return value

def promotion(baseline,candidate,visual):
 """Compare BOTH geometries on the SAME revised evidence, never against two different masks."""
 if not candidate.get('accepted_numerical_hypothesis'):return False,'candidate_failed_physical_or_stereo_checks'
 if 'distal_checks' in candidate and not all(v.get('reconstruction_visible') and v.get('distal_coverage',0.)>=.8 for v in candidate['distal_checks'].values()):return False,'candidate_missing_distal_tool_coverage'
 if visual['selection']!='candidate':return False,'visual_review_retained_baseline'
 if not all(visual['mask_supported'].values()) or not all(visual['candidate_better'].values()) or any(visual['catastrophic_failure'].values()):return False,'visual_evidence_failed'
 if baseline is None:return True,'missing_tool_recovered_on_verified_visual_evidence'
 a=baseline['views'];b=candidate['views']
 gain=sum(b[s]['mask_iou']-a[s]['mask_iou'] for s in SIDES)/2
 if gain<.02:return False,'insufficient_common_evidence_overlap_improvement'
 if min(b[s]['mask_iou']-a[s]['mask_iou'] for s in SIDES)<-.01:return False,'stereo_overlap_regression'
 # Poor baseline depth is not an authority; candidate must still pass original depth/workspace limits.
 if baseline.get('accepted_numerical_hypothesis'):
  for s in SIDES:
   if b[s]['median_absolute_depth_difference_m']>a[s]['median_absolute_depth_difference_m']+.001:return False,'verified_baseline_depth_regression'
 return True,'joint_visual_and_common_evidence_consistency_improvement'

def keep_missing_approximate(baseline,candidate,visual):
 """Availability, not promotion. Existing usable baselines still cannot be replaced."""
 if not candidate:return False
 if 'invalid_pose_or_joint_bounds' in candidate.get('failures',[]):return False
 if baseline is not None:
  views=baseline.get('views',{})
  # A stored mesh with no visible projection is not a usable reconstruction.
  # Preserve it on disk, but do not let its mere existence veto a supported fit.
  unavailable=set(views)==set(SIDES) and all(v.get('mask_iou',0.)<.05 and v.get('depth_samples',0)<20 for v in views.values())
  if baseline.get('accepted_numerical_hypothesis') or not unavailable:return False
 views=candidate.get('views',{})
 if set(views)!=set(SIDES) or not all(v.get('mask_iou',0.)>=.35 and v.get('depth_samples',0)>=100 for v in views.values()):return False
 return all(visual['mask_supported'].values()) and all(visual['candidate_better'].values()) and not any(visual['catastrophic_failure'].values())

def prefer_supported_approximate(baseline,candidate,visual):
 """Prefer a supported visual improvement without certifying estimated depth.

 A previously verified baseline is immutable. Estimated-depth and incomplete-distal checks remain explicitly unverified;
 structural failures veto selection and distal coverage cannot regress.
 Both geometries must be evaluated on the same revised SAM3 masks by the caller.
 """
 if not baseline or baseline.get('accepted_numerical_hypothesis'):return False,'verified_or_missing_baseline_not_this_policy'
 if not candidate or candidate.get('accepted_numerical_hypothesis'):return False,'use_regular_verified_promotion'
 allowed={'left_stereo_depth_mismatch','right_stereo_depth_mismatch','left_distal_shape_mismatch','right_distal_shape_mismatch'}
 failures=set(candidate.get('failures',[]))
 if not failures or not failures<=allowed:return False,'candidate_has_structural_or_unsupported_failures'
 if visual.get('selection')!='candidate':return False,'visual_review_did_not_select_candidate'
 if not all(visual.get('mask_supported',{}).get(s) and visual.get('candidate_better',{}).get(s)
            and visual.get('catastrophic_failure',{}).get(s) is False for s in SIDES):return False,'both_view_support_required'
 a=baseline.get('views',{});b=candidate.get('views',{});distal=candidate.get('distal_checks',{});old_distal=baseline.get('distal_checks',{})
 if not all(s in a and s in b and s in distal and s in old_distal for s in SIDES):return False,'missing_common_evidence_checks'
 if not all(b[s].get('mask_iou',0)>=.5 and b[s].get('depth_samples',0)>=100 and
            b[s]['mask_iou']>=a[s].get('mask_iou',0)+.03 for s in SIDES):return False,'both_view_overlap_improvement_required'
 if not all(distal[s].get(k,0)>=old_distal[s].get(k,0)-.01
            for s in SIDES for k in ('observed_coverage','distal_coverage','tip_region_coverage')):return False,'distal_or_tip_regression'
 return True,'better_stereo_visual_estimate_remaining_checks_explicitly_unverified'
