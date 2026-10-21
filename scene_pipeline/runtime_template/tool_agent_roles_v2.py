"""Runtime model specialists, not Codex chats. Share one centrally counted backend."""
from pathlib import Path
import json
from tool_agent_protocol import digest,route,perception,review
COMMON='''Use both complete original stereo views. IDs tool_A=left entry and tool_B=right entry are visual IDs, not robot arm identities. Never confuse curved needle or thin thread with rigid tool metal. Calibration/native dimensions are protected. Unknown hidden geometry is uncertainty, not a reason to suppress an otherwise usable reconstruction. Return only the exact requested JSON envelope with scene_version,evidence_ids,reason. Do not claim ground truth or physics validity.'''
ROLES={
'orchestrator':COMMON+'''\nYou schedule independent tool specialists. Inspect raw images and baseline overlays. For EACH tool return tasks:[{instance_id,action:"resegment_and_refit"|"keep_baseline",gross_mismatch:boolean,reason}]. A missing tool, projection on background, truncated shaft/head, or visibly wrong jaw position requires resegment_and_refit. A useful baseline is preserved until a verified improvement exists. Include both tools even if the baseline missed one. Choose keep_baseline only when no supported repair is needed. Budget permits at most one new segmentation/refit per tool in this pilot.''',
'perception':COMMON+'''\nYou propose native SAM3 prompts and bounds for ONE instrument independently of the baseline's faulty masks. Return instance_id,views:{left:{prompt,confidence_threshold,boxes:[{xyxy:[x0,y0,x1,y1],positive:true|false}],jaw_tips_xy:[[x,y],[x,y]],jaw_tip_visibility:"both_visible"|"not_both_visible",jaw_tip_reason:"..."},right:{...}}. Prompt menu: surgical grasper, robotic surgical grasper, surgical instrument. Threshold .1,.2,.35. Original full-frame pixel edge boxes 0..width/height; images are not cropped. Use a positive box around the visible target instrument; negative boxes may exclude the other instrument or needle. Boxes guide SAM3, they are not masks. Up to4 boxes/view. MANDATORY jaw-tip assessment in EACH stereo view for EACH tool: return jaw_tip_visibility="both_visible" with jaw_tips_xy containing EXACTLY2[x,y] points at the physical DISTAL endpoints of the two opposing jaw fingers, plus jaw_tip_reason. Do not mark the wrist pivot, shaft, needle grasp point, jaw base or tool silhouette corners. Closed jaws may have overlapping endpoint projections; do not spread them apart. Use ORIGINAL full-frame pixels. If the pair cannot both be observed because of occlusion or leaving frame, return jaw_tip_visibility="not_both_visible",jaw_tips_xy=[],jaw_tip_reason explaining which tip is hidden. Do not use missing confidence as a shortcut when both endpoints are visible. Native keypoint loss handles jaw ordering by testing both permutations. The same two physical endpoints must be identified in both views when visible. Never invent an occluded tip. Distinguish shaft, wrist and jaws. Do not reuse a wrong baseline region just to satisfy overlap.''',
'verifier':COMMON+'''\nYou independently review original images, proposed native SAM3 masks, and baseline/candidate fit overlays. Inspect each stereo view for correct instrument identity, shaft/head/jaw alignment, visible-object preservation and catastrophic geometry. Baseline and candidate are numerically rechecked on the SAME proposed masks, which are uncertain observations, not ground truth. Return instance_id,selection:"candidate"|"baseline",mask_supported:{left:bool,right:bool},candidate_better:{left:bool,right:bool},catastrophic_failure:{left:bool,right:bool}. Select candidate only when the mask traces the actual visible target and its geometry is visibly better in both views. If the baseline is absent, judge candidate against the visible tool. Poor baseline masks must not veto a supported new segmentation. Numerical acceptance is additionally enforced by the host.'''}
class Specialists:
 def __init__(self,backend,out):self.backend=backend;self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
 def call(self,role,state,evidence,*,reserve_calls=0):
  """One explicit validation-feedback opportunity, charged to the SAME budget.

  A bad box must not silently erase an instrument. The model corrects its own
  request; the host never clips coordinates or changes a negative box to positive.
  Reserve review/other-tool decisions before consuming the optional correction.
  """
  original=dict(state)
  for attempt in range(2):
   index_before=self.backend.index
   try:return self._call(role,state,evidence)
   except (ValueError,KeyError,TypeError) as error:
    if attempt or self.backend.index==index_before or self.backend.index+1+reserve_calls>self.backend.max_calls:raise
    feedback=dict(error=type(error).__name__+': '+str(error),instruction='Correct the invalid structured request using the same full stereo evidence. Invalid attempt remains charged. Do not invent landmarks or return an empty positive box list.')
    state=dict(original,validation_feedback=feedback)
  raise RuntimeError('Unreachable validation state')
 def _call(self,role,state,evidence):
  state=dict(state,evidence_ids=[eid for eid,_ in evidence]);state['scene_version']='v'+digest(state)[:12]
  folder=self.out/f'{self.backend.index+1:02d}_{role}';folder.mkdir()
  keys={'orchestrator':['tasks'],'perception':['instance_id','views'],'verifier':['instance_id','selection','mask_supported','candidate_better','catastrophic_failure']}[role]
  schema='EXACT TOP-LEVEL JSON KEYS: '+','.join(['scene_version','evidence_ids','reason']+keys)+'. reason MUST be a plain string, never an object. Put role payload fields at top level, not inside reason. No extra fields.'
  independence='\nYour selection describes VISUAL evidence only. If candidate is visibly better in both views and masks are supported, select candidate even when the numeric check fails; the host separately enforces promotion and approximate-availability policy. Do not change any numerical field.' if role=='verifier' else ''
  text=ROLES[role]+independence+'\n'+schema+'\nSTATE:'+json.dumps(state,separators=(',',':'))
  if len(text)>14000:raise ValueError('Bounded role context exceeded; do not truncate evidence')
  (folder/'state.json').write_text(json.dumps(state,indent=2))
  content=[dict(type='text',text=text)]
  for eid,path in evidence:content += [dict(type='text',text='EVIDENCE_ID='+eid),dict(type='image',url=str(path))]
  raw=self.backend([dict(role='user',content=content)])
  try:
   value=json.loads(raw);value={'orchestrator':route,'perception':perception,'verifier':review}[role](value,state)
  except (ValueError,KeyError,TypeError) as error:
   (folder/'validation_error.json').write_text(json.dumps(dict(error=type(error).__name__+': '+str(error),attempt_remains_charged=True),indent=2))
   raise
  (folder/'decision.json').write_text(json.dumps(value,indent=2));return value
