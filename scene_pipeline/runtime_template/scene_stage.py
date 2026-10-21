"""Preserve original heuristic scene; agent repairs are separate candidates."""
from pathlib import Path
import json,os,sys,shutil
R=Path(__file__).resolve().parent
sys.path[:0]=[str(R/'scene_sources'),str(R/'executed_sources')]
from run_scene_orchestrator_v7 import SceneOrchestrator,RoleModels,Budget
from reconstruction_tools import fingerprints

def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
 folder=Path(sys.argv[1]);baseline=json.loads((folder/'perception/baseline/assessment.json').read_text());per=json.loads((folder/'perception/result.json').read_text())
 key=sys.stdin.readline().strip();assert key;os.environ['OPENAI_API_KEY']=key;del key
 cache=Path(json.loads((folder/'ready.json').read_text())['cache']);rows=[];source=None;old=None
 for method in ['heuristic','agent']:
  summary={}
  case_path=Path(baseline['case_path']);needle=baseline['needle']['common_needle']
  if method=='agent' and per.get('accepted'):
   case_path=Path(per['selected_case']);needle=per['selected_needle']
  dest=folder/method
  try:
   models=RoleModels(folder/'agent_model_requests') if method=='agent' else None
   run=SceneOrchestrator(case_path,R/'pipeline-tools',Path(needle) if needle else None,dest,controller=method,models=models,cache=cache,budget=Budget(max_tool_calls=10))
   try:summary=run.run(baseline=(source,old) if method=='agent' and source is not None else None)
   except Exception as error:
    if method=='agent' and old is not None:
     summary=dict(old,controller='agent',status='baseline_returned_after_framework_failure',refinement_error=f'{type(error).__name__}: {error}',selection_origin='heuristic_baseline',model_accounting=models.accounting(),budget=dict(max_model_calls=run.budget.max_model_calls,max_tool_calls=run.budget.max_tool_calls,model_calls=run.budget.model_calls,tool_calls=run.budget.tool_calls));save(dest/'summary.json',summary)
    else:raise
   if method=='heuristic':source,old=run,summary
   row=dict(controller=method,status=summary['status'],summary=str(dest/'summary.json'))
  except Exception as error:
   dest.mkdir(exist_ok=True);save(dest/'framework_failure.json',dict(error=f'{type(error).__name__}: {error}'))
   row=dict(controller=method,status='framework_failed',error=f'{type(error).__name__}: {error}')
  needle=summary.get('needle_provenance',{}).get('path',needle)
  case_path=Path(summary.get('selected_case_path',case_path))
  context=R/'export_context'/method/folder.name;context.mkdir(parents=True,exist_ok=False)
  shutil.copy2(case_path,context/'case.json');shutil.copy2(R/'preregistration.json',context.parent/'preregistration.json');(context/method).symlink_to(dest,target_is_directory=True)
  if needle:shutil.copy2(needle,context/'common_needle.npz')
  save(context/'preprocessing_cache.json',dict(input_fingerprints=fingerprints(json.loads(case_path.read_text())['inputs'])))
  rows.append(row)
 save(folder/'scene_stage_results.json',rows)
if __name__=='__main__':main()
