"""One isolated full scene; a failed component never discards the baseline."""
from pathlib import Path
import json,os,sys,shutil,time,traceback,runpy
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
runpy.run_path(str(P/'transport_hook/sitecustomize.py'))
from contract import read,save,require_test,api_attempts,sha
from execution import execute,cpu,gpu_execute
from evidence_preservation import needs_perception_feedback
ROOT=Path(sys.argv[1]);CID=sys.argv[2];F=ROOT/CID;SHORT=CID.rsplit('_',1)[1]
sys.path[:0]=[str(ROOT),str(ROOT/'executed_sources'),str(ROOT/'perception_runtime')]
import coordinator as frozen
import run_perception as per
import run_tool_specialists as specialists
KEY=sys.stdin.readline().strip();assert KEY;os.environ['OPENAI_API_KEY']=KEY
os.environ['ISMR_API_SCENE_DIR']=str(F.resolve())
from api_resilience import outcomes as api_outcomes
ERRORS=[]
def status(**value):save(F/'active_stage.json',dict(updated=time.time(),**value))
frozen.status=status;specialists.status=status
frozen.gpu=lambda cmd,log,case,stage:gpu_execute(frozen,cmd,log,case,stage)
specialists.gpu=lambda cmd,log,case,stage:gpu_execute(specialists,cmd,log,case,stage)
def dispatch(cmd,log,timeout=900,key=None):
 cmd=[str(ROOT/'recovery_runtime/export_complete_v2.py') if str(v)==str(ROOT/'export_complete.py') else v for v in cmd]
 return execute(cmd,log,timeout,key)
frozen.execute=dispatch;specialists.execute=lambda cmd,log,timeout=900:execute(cmd,log,timeout)
per.execute=lambda cmd,log,timeout=600:execute(cmd,log,timeout)
def guided(case,request,pipeline,destination):
 return gpu_execute(per,per.command('segment_guided.py',['--case',str(case),'--request',str(request),'--checkpoint',per.CHECKPOINT,'--pipeline',str(pipeline),'--out',str(destination)],gpu=True),destination.parent/(destination.name+'_stdout.log'),CID,'perception_repair')
per.gpu_request=guided
def stage(name,action,scope=None):
 receipt=F/'stages'/(name+'.json')
 if read(receipt).get('state') in ['complete','failed']:
  previous=read(receipt)
  if previous['state']=='failed':ERRORS.append(dict(stage=name,error=previous.get('error','Preserved prior failure')))
  return previous['state']=='complete'
 if scope and any((p.parent.parent/'backend_manifest.json').exists() for p in scope.rglob('input.json')):
  error='Interrupted API stage not replayed; preserve charged attempts and best baseline'
  save(receipt,dict(state='failed',error=error));ERRORS.append(dict(stage=name,error=error));return False
 began=time.time();status(stage=name,state='running');save(receipt,dict(state='running',started=began))
 try:action();save(receipt,dict(state='complete',started=began,finished=time.time(),elapsed_seconds=time.time()-began));return True
 except Exception as e:
  error=(type(e).__name__+': '+str(e)).replace(KEY,'[REDACTED]');traceback.print_exc()
  ERRORS.append(dict(stage=name,error=error));save(receipt,dict(state='failed',error=error,started=began,finished=time.time(),elapsed_seconds=time.time()-began,scientific_outputs_retained=True));return False
def contexts(baseline,perception):
 for method in ['heuristic','agent']:
  folder=F/method;folder.mkdir(exist_ok=True);summary=read(folder/'summary.json')
  if not summary:
   old=read(F/'heuristic/summary.json') if method=='agent' else None
   summary=dict(old or dict(candidate=baseline.get('thread',{}).get('candidate'),attachment=None,completion=dict(all_current_endpoints_assessed=False),budget=dict(model_calls=0,tool_calls=0)),controller=method,status='preserved_baseline_after_component_failure',framework_failure=True)
   save(folder/'summary.json',summary)
  context=ROOT/'export_context'/method/CID;context.mkdir(parents=True,exist_ok=True)
  path=Path(baseline['case_path']);needle=baseline['needle']['common_needle']
  if method=='agent' and perception.get('accepted'):path=Path(perception['selected_case']);needle=perception['selected_needle']
  needle=summary.get('needle_provenance',{}).get('path',needle)
  path=Path(summary.get('selected_case_path',path))
  if not (context/'case.json').exists():shutil.copy2(path,context/'case.json')
  if not (context.parent/'preregistration.json').exists():shutil.copy2(ROOT/'preregistration.json',context.parent/'preregistration.json')
  if not (context/method).exists():(context/method).symlink_to(folder,target_is_directory=True)
  if needle and not (context/'common_needle.npz').exists():shutil.copy2(needle,context/'common_needle.npz')
def tool_fallback():
 path=F/'tools/complete_tools.json'
 if path.exists():return
 baseline=read(F/'tools/shared_broad_results.json').get('heuristic',{})
 for p in (F/'tools/heuristic').glob('*/final_selection.json'):baseline[p.parent.name]=read(p)
 save(path,dict(tools=dict(heuristic=baseline,agent=baseline),errors=ERRORS,actual_feedback_calls=0,estimated_api_cost_usd=0,baseline_preserved=True))
def main():
 require_test(P,CID);F.mkdir(exist_ok=True);marker=read(ROOT/'ready'/(CID+'.json'));assert marker.get('status')=='ready'
 save(F/'ready.json',marker)
 if not (F/'tools/complete_tools.json').exists():
  stage('heuristic_tools',lambda:frozen.heuristic_tools(F,Path(marker['cache'])))
 baseline_path=F/'perception/baseline/assessment.json'
 if not baseline_path.exists():
  from needle_tool_handoff import prepare
  working=F/'initial_tool_observations/case.json'
  if working.is_file():
   initial_case=read(working);requires_visual=False
  else:
   initial_case,requires_visual,working=prepare(read(F/'case.json'),None,F,'heuristic',F/'initial_tool_observations')
  assert not requires_visual
  confidence=read(F/'class_confidence_state.json')
  assert confidence and all(initial_case['inputs'][k]==read(F/'case.json')['inputs'][k] for k in confidence['bindings'])
  confidence['case_sha256']=sha(working)
  save(Path(working).parent/'class_confidence_state.json',confidence)
  ok=stage('initial_assessment',lambda:per.assess(F/'case.json',ROOT/'pipeline-tools',F/'perception/baseline',working=Path(working)))
  if not ok:raise RuntimeError('Initial fixed assessment failed; other scenes will continue')
 baseline=read(baseline_path);out=F/'perception';perception=read(out/'result.json')
 if not perception:
  if not needs_perception_feedback(baseline):
   perception=dict(status='original_needle_usable',accepted=False,model_calls=0,estimated_api_cost_usd=0,selected_case=None,selected_needle=None)
   save(out/'result.json',perception)
  else:
   stage('needle_perception_recovery',lambda:per.run(F/'case.json',ROOT/'pipeline-tools',out,baseline=baseline),scope=out)
   perception=read(out/'result.json')
   if not perception:perception=dict(status='perception_failed',accepted=False,model_calls=len([x for x in api_attempts(F) if x['stage']=='perception']),estimated_api_cost_usd=0);save(out/'result.json',perception)
 if not (F/'scene_stage_results.json').exists():stage('scene_controllers',lambda:dispatch(cpu(ROOT/'scene_stage.py',[F],ROOT,network=True),F/'scene_stage.log',key=KEY),scope=F/'agent_model_requests')
 contexts(baseline,perception)
 if not (F/'tools/complete_tools.json').exists():stage('heuristic_tools',lambda:frozen.heuristic_tools(F,Path(marker['cache'])))
 # Reserve one of the existing 23 API slots for fixed-size needle feedback.
 tool_fallback();calls=len(api_attempts(F));save(ROOT/'pilot_manifest.json',dict(cases={SHORT:dict(previous_calls=calls,max_new_calls=min(5,max(0,22-calls)))}))
 if not (F/'tool_agent/results.json').exists() and calls<22:stage('tool_specialists',lambda:specialists.run_case(SHORT),scope=F/'tool_agent')
 # Reuse the completed tool stage before giving up on an interior thread gap.
 stage('post_tool_connections',lambda:dispatch(cpu(ROOT/'connection_finalize.py',['--scene',F,'--pipeline',ROOT/'pipeline-tools'],ROOT),F/'post_tool_connections.log'))
 stage('fixed_size_needle_attachment',lambda:dispatch(cpu(ROOT/'needle_size_stage.py',['--scene',F],ROOT,network=True),F/'needle_size_stage.log',key=KEY),scope=F/'needle_size_refinement')
 stage('rigid_needle_export_contract',lambda:dispatch(cpu(ROOT/'needle_size_stage.py',['--scene',F,'--audit-export'],ROOT),F/'rigid_export_contract.log'))
 stage('bind_selected_export_context',lambda:dispatch(cpu(ROOT/'sync_export_contexts.py',[F],ROOT),F/'export_context_binding.log'))
 # Package failures do not invalidate surviving component outputs or halt queue.
 if not (F/'export_complete.json').exists():stage('portable_export',lambda:dispatch(cpu(ROOT/'export_complete.py',[F],ROOT),F/'advisor_export.log'))
 if not (F/'all_geometry_export.log.execution.json').exists():stage('geometry_export',lambda:dispatch(cpu(ROOT/'all_geometry_export/export_all_geometry.py',[SHORT],ROOT),F/'all_geometry_export.log'))
 from report_components import report
 report(F,ROOT)
 attempts=api_attempts(F);api=api_outcomes(F);save(F/'API_execution.json',api)
 if api['agent_work_pending']:ERRORS.append(dict(stage='API_execution',error='Agent decisions incomplete; provider failures/deferred work are not completed agent decisions',details=api))
 save(F/'case_result.json',dict(case_id=CID,status='agent_pending_provider' if api['agent_work_pending'] else 'attempt_complete' if not ERRORS else 'complete_with_component_failures',API_execution=api,errors=ERRORS,model_attempts=len(attempts),API_calls=attempts,independent_accuracy_measured=False,physics_validated=False,failed_components_preserved=True))
if __name__=='__main__':
 try:main()
 except Exception as e:
  traceback.print_exc();F.mkdir(exist_ok=True);save(F/'case_result.json',dict(case_id=CID,status='pipeline_failed',error=(type(e).__name__+': '+str(e)).replace(KEY,'[REDACTED]'),model_attempts=len(api_attempts(F)),failed_components_preserved=True));sys.exit(1)
