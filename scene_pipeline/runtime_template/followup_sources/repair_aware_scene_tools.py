"""Drop-in shared scene tools: preserve strands and verify proposed gap topology."""
from pathlib import Path
import sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scene_sources'))
from shared_scene_tools_v7 import SharedSceneToolsV7
from reconstruction_tools import fingerprints
from compact_agent_protocol import digest
from geometry_repairs import trace_strands,inferred_connection_guard
from strand_recovery import lift_strands

class RepairAwareSceneTools(SharedSceneToolsV7):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.occlusion_requests=0;self.rejected_menu_cards=[]
  self.paths.update({n:str(ROOT/n) for n in ['geometry_repairs.py','strand_recovery.py','repair_aware_scene_tools.py','coincident_stitch.py']})
  self.expected=fingerprints(self.paths)
 def action_key(self,card):
  if card['operation']=='strand_recovery':
   inputs={s:self.case['inputs']['thread_'+s+'_mask'] for s in ['left','right']}
   return digest(dict(operation=card['operation'],parameters=card['parameters'],mask_fingerprints=fingerprints(inputs)))
  return super().action_key(card)
 def recover_strands(self,reference,out):
  mask=cv2.imread(self.case['inputs']['thread_'+reference+'_mask'],0)>0
  paths,audit=trace_strands(mask)
  report=lift_strands(self.case,self.pipeline,reference,out,paths,audit)
  proposal=self.package(Path(out)/'fragments.npz',report,out)
  if report.get('pending_depth_artifact'):
   proposal['protected_paths']['pending_depth_paths']=report['pending_depth_artifact']['path']
   proposal['fingerprints']=fingerprints(proposal['protected_paths'])
  proposal=self.audit_geometry(proposal)
  return proposal,dict(operation='strand_recovery',reference=reference,topology_audit=audit,pending_depth_strands=report['pending_depth_strands'])
 def reconstruct_first(self,out):
  candidate,report,card=super().reconstruct_first(out)
  if candidate is not None:return candidate,report,card
  attempts=[]
  for side in ['left','right']:
   try:
    proposal,operation=self.recover_strands(side,Path(out)/('preserved_strands_'+side))
    attempts.append((proposal,operation))
   except ValueError:continue
  if not attempts:return candidate,report,card
  def rank(value):
   r=value[0]['report'];return (r['numerical_checks_passed'],min(r['visible_mask_coverage'].values()))
  proposal,operation=max(attempts,key=rank)
  return proposal,dict(report,status='visible_strands_recovered',strand_recovery=operation),dict(operation='strand_recovery',parameters=dict(reference=operation['reference']))
 def menu(self,candidate,attempted):
  actions,rows=super().menu(candidate,attempted)
  self.rejected_menu_cards=[]
  for aid,card in list(actions.items()):
   if card['operation']=='occluder_polygons' and self.occlusion_requests>=2:
    del actions[aid]
   elif card['operation']=='topology_paths' and not self.valid_topology(self.topology[aid]):
    self.rejected_menu_cards.append(dict(action_id=aid,reason='unsupported_sharp_crossing_turn'))
    del actions[aid]
  for side in ['left','right']:
   card=dict(operation='strand_recovery',parameters=dict(reference=side),
    effect='Recover all observed strands without welding crossing pixels; retain unresolved branches and use stereo for depth. Existing geometry remains preserved.')
   if self.action_key(card) not in attempted:actions['strands.'+side]=card
  if candidate and getattr(self,'allow_model_identity_cleanup',False):
   from coincident_stitch import proposals
   from thread_visibility_repairs_v5 import load_parts,load_inputs
   _,_,_,_,_,K,b=load_inputs(self.case,self.pipeline)
   for pair in proposals(load_parts(candidate['fragment_path']),K,b):
    card=dict(operation='coincident_endpoint_stitch',parameters=dict(first=pair['first'],second=pair['second']),
     effect='Join only numerically coincident, direction-consistent endpoints in both views; preserve every source point and add zero hidden-curve samples. Wrong crossing turns and competing endpoints remain ineligible.')
    if self.action_key(card) not in attempted:actions['numeric_stitch.'+'.'.join(map(str,pair['first']+pair['second']))]=card
  if self.needle_path and candidate and getattr(self,'allow_model_identity_cleanup',False):
   from needle_thread_identity import duplicate_runs
   from thread_visibility_repairs_v5 import load_parts,load_inputs
   with np.load(self.needle_path,allow_pickle=False) as d:needle={k:d[k].copy() for k in d.files}
   _,_,masks,_,_,K,b=load_inputs(self.case,self.pipeline)
   runs=duplicate_runs(load_parts(candidate['fragment_path']),needle,K,b,masks['left'].shape)
   if runs:
    card=dict(operation='needle_duplicate_trim',parameters={},effect='After reviewing both images, remove long thread samples that duplicate observed needle geometry in both stereo projections AND 3D; preserve crossings and swage vicinity; reconnect remaining fragments.')
    if self.action_key(card) not in attempted:actions['object_identity.needle_duplicate_trim']=card
  return actions,rows
 def valid_topology(self,row):
  directions={(p['zone'],p['port']):np.asarray(p['tangent_xy'],float) for p in row.get('ports',[])}
  return all(1.+float(directions[p['zone'],p['ports'][0]]@directions[p['zone'],p['ports'][1]])<=.6 for p in row.get('pairing_hypothesis',[]))
 def execute_repair(self,aid,card,candidate,out,selection_evidence=None):
  if card['operation']=='coincident_endpoint_stitch':
   self.assert_unchanged(candidate)
   if not selection_evidence or selection_evidence.get('controller')!='agent' or not {'image.left','image.right'}.issubset(selection_evidence.get('evidence_ids',[])):
    raise ValueError('Actual stereo-reviewed agent topology selection required')
   from coincident_stitch import stitch
   from thread_visibility_repairs_v5 import load_parts,load_inputs,save_parts,overlays,endpoint_table
   from complex_case_diagnostics import write_json
   _,images,masks,_,_,K,b=load_inputs(self.case,self.pipeline)
   params=card['parameters']
   if set(params)!={'first','second'}:raise ValueError('Explicit current endpoint pair required')
   parts,operation=stitch(load_parts(candidate['fragment_path']),tuple(params['first']),tuple(params['second']),K,b)
   if parts is None:raise ValueError(operation)
   out=Path(out);out.mkdir(parents=True,exist_ok=False);save_parts(out/'fragments.npz',parts);overlays(parts,images,K,b,out)
   report=dict(candidate['report'],fragment_count=len(parts),remaining_fragment_count=len(parts),
    endpoints=endpoint_table(parts,K,b,masks['left'].shape),coincident_endpoint_stitch=operation,
    selection_evidence=selection_evidence,topology_is_hypothesis=True,physical_collision_validated=False)
   write_json(out/'stitch_receipt.json',operation)
   proposal=self.audit_geometry(self.package(out/'fragments.npz',report,out))
   return proposal,operation
  if card['operation']=='needle_duplicate_trim':
   self.assert_unchanged(candidate)
   from needle_thread_identity import execute_identity_cleanup
   return execute_identity_cleanup(self,candidate,out,selection_evidence)
  if card['operation']=='occluder_polygons':
   if self.occlusion_requests>=2:raise ValueError('Two occlusion requests already attempted; preserve feedback and remaining connection budget')
   self.occlusion_requests+=1
  if card['operation']=='strand_recovery':
   self.assert_unchanged(candidate)
   if set(card['parameters'])!={'reference'} or card['parameters']['reference'] not in ['left','right']:raise ValueError('Explicit stereo reference required')
   return self.recover_strands(card['parameters']['reference'],out)
  if card['operation']=='topology_paths':
   row=self.topology[aid]
   directions={(p['zone'],p['port']):np.asarray(p['tangent_xy'],float) for p in row.get('ports',[])}
   for pair in row.get('pairing_hypothesis',[]):
    a,b=pair['ports'];zone=pair['zone']
    if 1.+float(directions[zone,a]@directions[zone,b])>.6:
     raise ValueError('Crossing pairing makes an unsupported sharp turn; preserve separate strands and inspect both views')
  if card['operation'] in ['smooth_connection','inferred_connection'] and candidate:
   from thread_visibility_repairs_v5 import join_proposals,load_parts
   parts=load_parts(candidate['fragment_path']);proposals=join_proposals(self.case,self.pipeline,parts)
   selected=next((p for p in proposals if p['proposal_id']==card['parameters']['proposal_id']),None)
   if selected is None:raise ValueError('Connection proposal changed; request fresh endpoint inventory')
   guard=inferred_connection_guard(selected['_curve'],[p['points_m'] for p in parts])
   if not guard['allowed']:raise ValueError(guard['reason'])
  return super().execute_repair(aid,card,candidate,out,selection_evidence=selection_evidence)
