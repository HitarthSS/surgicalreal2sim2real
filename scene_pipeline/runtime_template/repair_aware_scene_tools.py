"""Shared depth-first repair and stereo-reviewed endpoint-cluster assembly."""
from pathlib import Path
from copy import deepcopy
import json,cv2,numpy as np
from repair_base import RepairAwareSceneTools as Base
from reconstruction_tools import fingerprints,sha256
from compact_agent_protocol import digest
from complex_case_diagnostics import write_json
from thread_visibility_repairs_v5 import load_inputs,load_parts,save_parts,overlays,endpoint_table
ROOT=Path(__file__).parent

class RepairAwareSceneTools(Base):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.paths.update({n:ROOT/n for n in ('repair_base.py','cluster_topology.py','stereo_depth_recovery.py','repair_aware_scene_tools.py')});self.expected=fingerprints(self.paths)
 def action_key(self,card):
  key=super().action_key(card)
  if card['operation'] in ('cluster_topology','stereo_depth_repair'):
   return digest([key,self.geometry_version,{k:sha256(v) for k,v in self.case['inputs'].items() if k.endswith('_mask') or k=='depth_npy'}])
  return key
 def menu(self,candidate,attempted):
  cards,rows=super().menu(candidate,attempted)
  failures=(candidate or {}).get('report',{}).get('failures',[])
  if candidate is None or any('depth' in f for f in failures):
   card=dict(operation='stereo_depth_repair',parameters={},effect='Recover independent mask-supported rectified stereo correspondences from both original images; reject ambiguous or non-mutual matches, mark unsupported depth missing, relift strands and retain only verified improvements before binding.')
   if self.action_key(card) not in attempted:cards['depth.mutual_stereo']=card
  if candidate and candidate['report'].get('numerical_checks_passed') and getattr(self,'allow_model_identity_cleanup',False):
   from cluster_topology import candidates
   _,_,masks,_,_,K,b=load_inputs(self.case,self.pipeline);parts=load_parts(candidate['fragment_path'])
   reserved=getattr(self,'reserved_material_points',[])
   for row in candidates(parts,K,b):
    points=[parts[f]['points_m'][0 if e==0 else -1] for pair in row['pairs'] for f,e in pair]
    if any(np.linalg.norm(p-q)<1e-8 for p in points for q in reserved):continue
    card=dict(operation='cluster_topology',parameters={'candidate_id':row['candidate_id'],'pairs':row['pairs']},effect='Stereo-reviewed one-to-one ordering at coincident crossing endpoints. Fixed matching preserves every material sample, adds zero bridge points, keeps crossing chains separate and rejects sharp turns, reused endpoints and closed cycles. Topology remains inferred.')
    if self.action_key(card) not in attempted:cards['cluster.'+row['candidate_id']]=card
  return cards,rows
 def depth_repair(self,candidate,out):
  from stereo_depth_recovery import recover
  out=Path(out);out.mkdir(parents=True,exist_ok=False)
  images={s:cv2.cvtColor(cv2.imread(self.case['inputs'][s]),cv2.COLOR_BGR2RGB) for s in ('left','right')};masks={s:cv2.imread(self.case['inputs']['thread_'+s+'_mask'],0)>0 for s in images};K=np.loadtxt(self.case['inputs']['K'],delimiter=',');depth=np.load(self.case['inputs']['depth_npy'])
  fixed,pairs,receipt=recover(images['left'],images['right'],masks['left'],masks['right'],depth,K,self.case['baseline_m'])
  receipt['source_fingerprints']=fingerprints(self.case['inputs']);write_json(out/'stereo_receipt.json',receipt);np.savez_compressed(out/'correspondences.npz',left_xy=pairs[:,:2],right_xy=pairs[:,2:4],depth_m=pairs[:,4],score=pairs[:,5])
  if len(pairs)<2:raise ValueError('Fewer than two unambiguous stereo matches; original geometry preserved')
  np.save(out/'depth_candidate.npy',fixed)
  self.case['inputs']['depth_npy']=str((out/'depth_candidate.npy').resolve());self.version+=1;self.working_path=out/'derived_case.json';write_json(self.working_path,self.case)
  self.register_derived({str(p):str(p) for p in out.iterdir() if p.is_file()})
  proposals=[];errors=[]
  for side in ('left','right'):
   try:
    proposal,audit=self.recover_strands(side,out/('reference_'+side));proposals.append(proposal)
   except ValueError as error:errors.append(dict(reference=side,error=str(error)))
  receipt['lifting_errors']=errors;write_json(out/'stereo_receipt.json',receipt)
  # A receipt is finalized before its protected fingerprint is registered.
  self.register_derived({str(out/'stereo_receipt.json'):str(out/'stereo_receipt.json')})
  if not proposals:raise ValueError('Stereo matches found but no strand passes metric lifting; source preserved')
  proposal=max(proposals,key=lambda p:(p['report']['numerical_checks_passed'],min(p['report']['visible_mask_coverage'].values()),-len(p['report']['failures'])))
  proposal['report']['stereo_depth_repair']=dict(receipt=str(out/'stereo_receipt.json'),depth_is_estimated=True)
  write_json(proposal['protected_paths']['report'],proposal['report'])
  proposal['protected_paths'].update({'recovered_depth':str(out/'depth_candidate.npy'),'stereo_correspondences':str(out/'correspondences.npz'),'stereo_receipt':str(out/'stereo_receipt.json')});proposal['fingerprints']=fingerprints(proposal['protected_paths'])
  return proposal,receipt
 def reconstruct_first(self,out):
  candidate,report,card=super().reconstruct_first(out)
  if candidate and candidate['report'].get('numerical_checks_passed'):return candidate,report,card
  snapshot={k:deepcopy(getattr(self,k)) for k in ('case','version','working_path','routed','last_audit','automatic_keys')}
  try:
   proposal,receipt=self.depth_repair(candidate,Path(out)/'stereo_depth_before_binding')
   from refinement_policy import scene_promotes
   promoted,why=scene_promotes(candidate,proposal)
   if promoted:return proposal,dict(report,depth_repair=receipt,promotion=why),dict(operation='stereo_depth_repair',parameters={})
  except ValueError:pass
  for key,value in snapshot.items():setattr(self,key,value)
  return candidate,report,card
 def execute_repair(self,aid,card,candidate,out,selection_evidence=None):
  if card['operation']=='stereo_depth_repair':
   self.assert_unchanged(candidate)
   if card['parameters']!={}:raise ValueError('Fixed stereo recovery contract')
   return self.depth_repair(candidate,out)
  if card['operation']=='cluster_topology':
   self.assert_unchanged(candidate)
   if not candidate or not candidate['report']['numerical_checks_passed']:raise ValueError('Recover depth/visible geometry before binding')
   if not selection_evidence or selection_evidence.get('controller')!='agent' or not {'image.left','image.right'}<=set(selection_evidence.get('evidence_ids',[])):raise ValueError('Actual stereo-reviewed topology decision required')
   from cluster_topology import candidates,apply
   from thread_pair_assembly_v7 import separate_coverage
   _,images,masks,_,_,K,b=load_inputs(self.case,self.pipeline);parts=load_parts(candidate['fragment_path'])
   rows=candidates(parts,K,b);params=card['parameters'];row=next((r for r in rows if r['candidate_id']==params.get('candidate_id') and r['pairs']==params.get('pairs')),None)
   if row is None:raise ValueError('Stale cluster matching')
   endpoints=[parts[f]['points_m'][0 if e==0 else -1] for pair in row['pairs'] for f,e in pair]
   if any(np.linalg.norm(p-q)<1e-8 for p in endpoints for q in getattr(self,'reserved_material_points',[])):raise ValueError('Reserved attachment terminal')
   result=apply(parts,row['pairs'],K,b);out=Path(out);out.mkdir(parents=True,exist_ok=False);save_parts(out/'fragments.npz',result);overlays(result,images,K,b,out)
   coverage=separate_coverage(result,masks,K,b,8.)
   report=dict(candidate['report'],fragment_count=len(result),remaining_fragment_count=len(result),visible_mask_coverage=coverage,endpoints=endpoint_table(result,K,b,masks['left'].shape),cluster_topology=row,selection_evidence=selection_evidence,topology_is_hypothesis=True,physical_collision_validated=False)
   report['hypothesis_lineage']=list(report.get('hypothesis_lineage',[]))+[dict(mode='coincident_cluster_topology',candidate_id=row['candidate_id'],selection_evidence=selection_evidence)]
   report['connected_hypothesis_only']=True
   return self.audit_geometry(self.package(out/'fragments.npz',report,out)),row
  return super().execute_repair(aid,card,candidate,out,selection_evidence)
