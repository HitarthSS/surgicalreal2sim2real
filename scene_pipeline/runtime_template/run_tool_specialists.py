"""Coded orchestrator -> perception -> SAM3/fitting tools -> independent review.

No generated code is executed. API budget shared across roles; original baseline
geometry is immutable. Every tool ID is routed, including absent baseline tools.
"""
from pathlib import Path
import sys,os,json,shutil,subprocess,fcntl,time,struct,traceback,hashlib
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'executed_sources'),str(ROOT)]
from openai_scene_backend import OpenAISceneModel
from tool_agent_roles_v2 import Specialists
from tool_agent_protocol import promotion,keep_missing_approximate,prefer_supported_approximate
from geometry_repairs import tool_mask_diagnostics
import cv2
SAM3='sha256:901ee73c7e4b7904e4c52d93224635f471bb14fd1ad536cd207971ff530e14b2'
TRACKER='sha256:7436a2116753bbcf1d45194df17922748210afc523d23860d1d59dfafc9c9e00'
PHYSICS='sha256:bd675f584cb274db81af062e8fc0934fdfbdf87b228f57a6ff99e6a8529bb12e'
BASE=ROOT
CHECKPOINT='/data/xliang/ismr_2027/sam3_pipeline10_20261019_v1/model/sam3_human27_best_epoch2.pt'
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def status(**v):save(ROOT/'job_status.json',dict(pid=os.getpid(),**v))
def execute(cmd,log,timeout=900):
 with Path(log).open('w') as f:
  p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
 if p.returncode:raise RuntimeError('Tool execution failed: '+str(log))
def docker(script,args=(),image=PHYSICS,gpu=False):
 cmd=['docker','run','--rm','--cpus','4','--memory','16g','--user','1009:1009','-e','HOME=/tmp','-e','USER=ismr','-e','LOGNAME=ismr','-e','PYTHONDONTWRITEBYTECODE=1','-e','MPLBACKEND=Agg','-e','OPENBLAS_NUM_THREADS=1','-e','OMP_NUM_THREADS=4','-e','TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_ismr','-e','TORCH_EXTENSIONS_DIR='+str(ROOT/'cache/extensions'),'-v','/home/hitarth:/home/hitarth:ro','-v',str(ROOT)+':'+str(ROOT),'--network','none']
 if gpu:cmd+=['--gpus','device=0','--shm-size','2g']
 return cmd+['--entrypoint','python3',image,str(script)]+list(map(str,args))
def gpu(cmd,log,case,stage):
 from gpu_resources import execute_gpu
 return execute_gpu(sys.modules[__name__],cmd,log,case,stage)
def brief(row):
 if not row:return dict(baseline_available=False)
 checks=row['verification'].get('distal_checks',{})
 return dict(baseline_available=True,views=row['verification']['views'],distal_checks=checks,repair_required=bool(checks and any(not v['reconstruction_visible'] or v['distal_coverage']<.8 for v in checks.values())),numerical_pass=row.get('accepted_tool_hypothesis',False))
def run_case(short):
 cid='snapshot_20261015_'+short;folder=ROOT/cid/'tool_agent';folder.mkdir(exist_ok=False);casepath=BASE/cid/'case.json';case=json.loads(casepath.read_text())
 original=json.loads((BASE/cid/'tools/complete_tools.json').read_text());baseline=original['tools'].get('heuristic',{});selected=json.loads(json.dumps(baseline));events=[]
 protected={p:sha(p) for p in list(case['inputs'].values())+[str(casepath)]+[x['mesh'] for x in baseline.values()]}
 budget=json.loads((ROOT/'pilot_manifest.json').read_text())['cases'][short]
 backend=OpenAISceneModel(folder/'model_requests',max_calls=budget['max_new_calls']);roles=Specialists(backend,folder/'roles')
 sizes={s:list(struct.unpack('>II',Path(case['inputs'][s]).read_bytes()[16:24])) for s in ['left','right']}
 images=[('image.'+s,case['inputs'][s]) for s in ['left','right']]
 evidence=images+[(n+'.baseline.'+s,p) for n,r in baseline.items() for s,p in r['verification']['feedback_images'].items()]
 try:
  plan=roles.call('orchestrator',dict(case_id=cid,tools={n:brief(baseline.get(n)) for n in ['tool_A','tool_B']},remaining_calls=backend.max_calls),evidence,reserve_calls=4)
  tasks=plan['tasks']
 except Exception as e:
  events.append(dict(stage='orchestrator',error=str(e),fallback='host schedules missing/failed tools'))
  tasks=[dict(instance_id=n,action='resegment_and_refit' if not baseline.get(n,{}).get('accepted_tool_hypothesis') else 'keep_baseline',gross_mismatch=False,reason='Fallback follows recorded numerical/missing failures') for n in ['tool_A','tool_B']]
 # Both tools must receive the perception/keypoint specialist even when router keeps the silhouette baseline.
 events.append(dict(stage='mandatory_keypoint_assessment',tools=['tool_A','tool_B'],original_router_tasks=tasks))
 tasks=[dict(instance_id=n,action='resegment_and_refit',gross_mismatch=False,reason='Required native jaw-tip assessment and separate candidate fit; preserve verified baseline unless improved') for n in ['tool_A','tool_B']]
 for index,task in enumerate(tasks):
  name=task['instance_id']
  if task['action']=='keep_baseline':events.append(dict(instance=name,status='baseline_kept_by_router'));continue
  # Reserve one perception and one reviewer call; do not exhaust calls on first instrument.
  if backend.max_calls-backend.index<2:events.append(dict(instance=name,status='budget_exhausted_baseline_retained'));continue
  out=folder/name;out.mkdir();old=baseline.get(name);prior=brief(old)
  try:
   last_feedback=None;feedback_images=[];base_out=out
   for attempt in range(1,3):
    remaining_tasks=sum(t['action']=='resegment_and_refit' for t in tasks[index+1:])
    if backend.max_calls-backend.index<2+2*remaining_tasks:break
    out=base_out if attempt==1 else base_out/'retry_02'
    out.mkdir(exist_ok=True)
    status(state='running',case=short,stage=name+'/perception_specialist',gpu=0)
    request=roles.call('perception',dict(instance_id=name,image_sizes=sizes,baseline=prior,last_feedback=last_feedback,diagnostic='Ignore unsupported heuristic bounds; find visible instrument directly in full stereo.'),images+[(name+'.baseline.'+s,p) for s,p in (old or {}).get('verification',{}).get('feedback_images',{}).items()]+feedback_images,reserve_calls=1+2*remaining_tasks)
    samreq=dict(objects={name:{s:{k:v for k,v in request['views'][s].items() if k in ('prompt','confidence_threshold','boxes')} for s in ['left','right']}})
    save(out/'sam3_request.json',samreq)
    segmentation=out/'sam3'
    gpu(docker(ROOT/'segment_tools.py',['--case',casepath,'--request',out/'sam3_request.json','--checkpoint',CHECKPOINT,'--pipeline',ROOT/'pipeline-tools','--out',segmentation],image=SAM3,gpu=True),out/'sam3.log',short,name+'/guided_sam3')
    masks_for_check={s:cv2.imread(str(segmentation/(s+'_'+name+'_mask.png')),0)>0 for s in ['left','right']}
    thread_for_check={s:cv2.imread(case['inputs']['thread_'+s+'_mask'],0)>0 for s in ['left','right']}
    semantic_check=tool_mask_diagnostics(masks_for_check,thread_for_check,entry_border='left' if name=='tool_A' else 'right')
    save(out/'mask_semantic_check.json',semantic_check)
    if semantic_check['requires_resegmentation']:
     events.append(dict(instance=name,status='mask_identity_repair_required',diagnostic=semantic_check))
     last_feedback=dict(reason='Stereo instance masks are incompatible; independently relocalize this instrument in both FULL views.',diagnostic=semantic_check)
     feedback_images=[(name+'.failed_mask.'+s,str(segmentation/(s+'_'+name+'_mask.png'))) for s in ['left','right']]
     continue
    save(out/'keypoint_evidence.json',dict(views={s:{k:request['views'][s][k] for k in ('jaw_tips_xy','jaw_tip_visibility','jaw_tip_reason')} for s in ('left','right')},native_tip_frames=[2,3],input_pixel_units='original_full_frame_pixels',preserve_baseline_until_verified=True))
    cfg=dict(out=str(out/'fit'),case=str(casepath),repo='/home/hitarth/online_dvrk_tracking_clean_extracted/online_dvrk_tracking_clean',instance_id=name,views=request['views'],masks={s:str(segmentation/(s+'_'+name+'_mask.png')) for s in ['left','right']},baseline_mesh=old['mesh'] if old else None)
    save(out/'fit_config.json',cfg)
    execute(docker(ROOT/'fit_candidate.py',['seed',out/'fit_config.json'],image=TRACKER),out/'seed.log')
    native=out/'fit'
    gpu(docker(ROOT/'native_candidate_worker.py',['--repo',cfg['repo'],'--run',native,'--K',case['inputs']['K'],'--request',native/'native_request.json'],image=TRACKER,gpu=True),out/'native.log',short,name+'/fresh_native_fit')
    import numpy as np
    with np.load(native/'robot/online_tracker/tracker_result.npz',allow_pickle=False) as tracked:
     expected=np.asarray(request['views']['left']['jaw_tips_xy'],float).reshape(-1,2)
     assert np.array_equal(tracked['seed_keypoints_xy'],expected.astype('float32')),'Native optimizer did not consume supplied tips'
     assert bool(tracked['jaw_tip_evidence_available'])==bool(len(expected))
     assert bool(tracked['keypoint_loss_enabled'])==bool(len(expected)),'Native keypoint loss was silently disabled'
    status(state='running',case=short,stage=name+'/stereo_verification',gpu=0)
    execute(docker(ROOT/'fit_candidate.py',['evaluate',out/'fit_config.json'],image=TRACKER),out/'evaluate.log',400)
    evaluation=json.loads((native/'evaluation.json').read_text());candidate=evaluation['candidate'];reports=candidate['report']
    rev_evidence=images+[(name+'.candidate.'+s,p) for s,p in reports['feedback_images'].items()]
    if evaluation['baseline']:rev_evidence += [(name+'.baseline_common_masks.'+s,p) for s,p in evaluation['baseline']['feedback_images'].items()]
    view=roles.call('verifier',dict(instance_id=name,baseline=evaluation['baseline']['views'] if evaluation['baseline'] else None,candidate=reports['views'],candidate_numerical_pass=reports['accepted_numerical_hypothesis'],mask_source='native SAM3 conditioned by specialist boxes; not ground truth'),rev_evidence,reserve_calls=2*remaining_tasks)
    promoted,reason=promotion(evaluation['baseline'],reports,view)
    if promoted:
     selected[name]=dict(mesh=candidate['mesh'],verification=reports,accepted_tool_hypothesis=True,numerical_recovery_passed=True,agent_review_status='specialist_verified_improvement',selection_origin='agent_segmented_refit',baseline_preserved=True,visual_instance=name,psm_identity=None)
    approximate,approximate_reason=prefer_supported_approximate(evaluation['baseline'],reports,view)
    if not promoted and approximate:
     selected[name]=dict(mesh=candidate['mesh'],verification=reports,accepted_tool_hypothesis=False,numerical_recovery_passed=False,
         agent_review_status='visual_improvement_unverified',selection_origin='agent_segmented_refit_approximate',
         baseline_preserved=True,baseline_mesh=old['mesh'],visual_instance=name,psm_identity=None,
         uncertainty_warnings=reports.get('failures',[]),approximate_selection_reason=approximate_reason)
    if not promoted and keep_missing_approximate(evaluation['baseline'],reports,view):
     selected[name]=dict(mesh=candidate['mesh'],verification=reports,accepted_tool_hypothesis=False,numerical_recovery_passed=False,agent_review_status='approximate_requires_review',selection_origin='agent_segmented_refit_approximate',baseline_preserved=True)
    events.append(dict(instance=name,attempt=attempt,promoted=promoted,reason=reason,candidate_mesh=candidate['mesh'],evaluation=str(native/'evaluation.json'),review=view))
    if promoted:break
    last_feedback=dict(reason=reason,review_reason=view['reason'],numerical_failures=reports.get('failures',[]))
    feedback_images=[(name+'.previous_candidate.'+s,p) for s,p in reports['feedback_images'].items()]
  except Exception as e:
   events.append(dict(instance=name,status='repair_failed_baseline_retained',error=f'{type(e).__name__}: {e}'));save(out/'failure.json',events[-1])
  save(folder/'progress.json',dict(events=events,actual_api_calls=backend.index,selected_tools=list(selected)))
 assert all(sha(p)==h for p,h in protected.items()),'Protected baseline/input changed'
 assert budget['previous_calls']+backend.index<=23
 result=dict(case_id=cid,tools=dict(heuristic=baseline,agent=selected),events=events,actual_api_calls=backend.index,combined_api_calls=budget['previous_calls']+backend.index,api_cost_estimate_usd=backend.estimated_api_cost_usd,original_sha256=protected,independent_accuracy_measured=False,physics_validated=False)
 save(folder/'results.json',result)
 old_complete=json.loads((BASE/cid/'tools/complete_tools.json').read_text())
 save(BASE/cid/'tools/complete_tools.json',dict(old_complete,tools=result['tools'],specialist_events=events,actual_feedback_calls=backend.index,estimated_api_cost_usd=backend.estimated_api_cost_usd,errors=old_complete['errors']+[e for e in events if e.get('status')=='repair_failed_baseline_retained']))
 return dict(case_id=cid,calls=backend.index,cost_estimate=backend.estimated_api_cost_usd,promoted=[e['instance'] for e in events if e.get('promoted')],failures=[e for e in events if e.get('status')=='repair_failed_baseline_retained'])
def main():
 key=sys.stdin.readline().strip();assert key;os.environ['OPENAI_API_KEY']=key;del key
 rows=[]
 for short in json.loads((ROOT/'pilot_manifest.json').read_text())['cases']:
  rows.append(run_case(short));save(ROOT/'pilot_results.json',rows)
 source=json.loads((ROOT/'source_manifest.json').read_text());assert all(sha(ROOT/p)==h for p,h in source.items()),'Pilot code changed'
 status(state='complete',results=rows,source_hashes_verified=True)
if __name__=='__main__':
 try:main()
 except Exception as e:traceback.print_exc();status(state='stopped_for_review',error=str(e))
