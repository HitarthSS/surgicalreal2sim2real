"""Ten earlier development cases, five pairs. Multipart tool masks + retained occlusion evidence. Heuristic-first scene and tool specialists.

No scheduled jobs, old queue resume, repeated completed stage or generated code.
"""
from pathlib import Path
import json,os,sys,time,subprocess,shutil,traceback,fcntl,hashlib
R=Path(__file__).resolve().parent
ROOT=R
TRACKER='sha256:7436a2116753bbcf1d45194df17922748210afc523d23860d1d59dfafc9c9e00'
PHYSICS='sha256:bd675f584cb274db81af062e8fc0934fdfbdf87b228f57a6ff99e6a8529bb12e'
def save(p,v):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def status(**v):save(R/'job_status.json',dict(pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**v))
def execute(cmd,log,timeout=900,key=None):
 with Path(log).open('w') as f:
  proc=subprocess.Popen(cmd,stdin=subprocess.PIPE if key else subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
  if key:proc.stdin.write((key+'\n').encode());proc.stdin.close()
  try:code=proc.wait(timeout)
  except subprocess.TimeoutExpired:
   proc.terminate()
   try:proc.wait(timeout=30)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
   raise RuntimeError('Stage timeout; log preserved: '+str(log))
 if code:raise RuntimeError('Stage failed; log preserved: '+str(log))
def docker(script,args=(),tracker=False,gpu=False,network=False):
 cmd=['docker','run','--rm','--cpus','4','--memory','16g','--user','1009:1009','-e','HOME=/tmp','-e','USER=ismr','-e','LOGNAME=ismr','-e','PYTHONDONTWRITEBYTECODE=1','-e','MPLBACKEND=Agg','-e','OPENBLAS_NUM_THREADS=1','-e','OMP_NUM_THREADS=4','-e','TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_ismr','-e','TORCH_EXTENSIONS_DIR='+str(R/'cache/extensions'),'-v','/home/hitarth:/home/hitarth:ro','-v',str(R)+':'+str(R),'--network','bridge' if network else 'none']
 if gpu:cmd+=['--gpus','device=0','--shm-size','2g']
 if network:cmd+=['-i']
 return cmd+['--entrypoint','python3',TRACKER if tracker or gpu else PHYSICS,str(script)]+list(map(str,args))
def gpu(cmd,log,case,stage):
 from gpu_resources import execute_gpu
 return execute_gpu(sys.modules[__name__],cmd,log,case,stage)
def preprocess(batch,shorts):
 import queue_worker_host as work
 with (R/'gpu0.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  markers=[R/'ready'/('snapshot_20261015_'+c+'.json') for c in shorts]
  if all(p.exists() for p in markers):
   reg=json.loads((R/'preregistration.json').read_text())
   for c,p in zip(shorts,markers):
    assert json.loads(p.read_text())['status']=='ready'
    report=json.loads((R/('snapshot_20261015_'+c)/'preprocessing.json').read_text())
    assert report['status']=='preprocessed' and report['input_sha256']==reg['input_sha256'][c]
   return
  assert not any(p.exists() for p in markers),'Partial preprocessing requires review before reuse'
  while True:
   free=work.gpu_free();available=[g for g in [0,1] if free.get(g,0)>=6144]
   if available:break
   status(state='waiting_for_gpu',stage='preprocessing',cases=shorts,free_mib=free);time.sleep(10)
  chosen=available[0];status(state='preprocessing',batch=batch,cases=shorts,gpu=chosen)
  work.process_chunk(chosen,batch,shorts)
def heuristic_tools(folder,cache):
 stage=folder/'tools';shutil.copytree(R/'tool_template',stage,symlinks=True)
 save(stage/'pilot_config.json',dict(case=str(folder/'case.json'),K=str(folder/'K.csv'),repo='/home/hitarth/online_dvrk_tracking_clean_extracted/online_dvrk_tracking_clean',cache=str(cache/folder.name/'occluder.sam3.grasper'),case_id=folder.name,model='gpt-6.1-sol',scale_provisional=True))
 execute(docker(stage/'prepare_heuristic_only.py'),stage/'initialization.log');fits=[]
 request=stage/'heuristic/initialization.json'
 if request.exists():
  for item in json.loads(request.read_text())['instances']:
   name=item['instance_id'];out=stage/'heuristic'/name
   try:
    gpu(docker(stage/'native_worker.py',['--repo','/home/hitarth/online_dvrk_tracking_clean_extracted/online_dvrk_tracking_clean','--run',out,'--K',folder/'K.csv','--request',out/'native_request.json'],gpu=True),stage/(name+'_native.log'),folder.name,name+'/heuristic_native')
    execute(docker(stage/'verify_tools.py',['--method','heuristic','--instance',name]),stage/(name+'_verify.log'))
    fits.append(dict(instance=name,status='native_fit_complete'))
   except Exception as e:fits.append(dict(instance=name,status='failed',error=str(e)))
   save(stage/'native_fit_results.json',fits)
 execute(docker(stage/'complete_heuristic.py',tracker=True),stage/'heuristic_broad.log',1800)
 return json.loads((stage/'complete_tools.json').read_text())
def case_job(short,key):
 folder=R/('snapshot_20261015_'+short);marker=json.loads((R/'ready'/(folder.name+'.json')).read_text());save(folder/'ready.json',marker)
 if marker['status']!='ready':raise RuntimeError('Initial preprocessing failed; inspect segmentation/stereo logs')
 sys.path.insert(0,str(R/'perception_runtime'));import run_perception as per
 def perception_gpu(case_path,request_path,pipeline,destination):
  gpu(per.command('segment_guided.py',['--case',str(case_path),'--request',str(request_path),'--checkpoint',per.CHECKPOINT,'--pipeline',str(pipeline),'--out',str(destination)],gpu=True),destination.parent/(destination.name+'_stdout.log'),short,'thread_needle_mask_repair');return 0
 per.gpu_request=perception_gpu
 out=folder/'perception';out.mkdir()
 status(state='running',case=short,stage='initial_needle_thread_assessment')
 baseline=per.assess(folder/'case.json',R/'pipeline-tools',out/'baseline')
 if baseline['needle']['common_needle']:
  perception=dict(status='original_needle_usable',accepted=False,model_calls=0,estimated_api_cost_usd=0,selected_case=None,selected_needle=None,reason='Preserve original heuristic; agent scene has shared cached mask repair tools.')
  per.save(out/'result.json',perception)
 else:
  status(state='running',case=short,stage='needle_perception_recovery')
  perception=per.run(folder/'case.json',R/'pipeline-tools',out,baseline=baseline)
 status(state='running',case=short,stage='heuristic_then_agent_scene')
 execute(docker(R/'scene_stage.py',[folder],network=True),folder/'scene_stage.log',1800,key)
 scene=json.loads((folder/'scene_stage_results.json').read_text());summary_path=folder/'agent/summary.json'
 summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
 calls=int(perception['model_calls'])+int((summary.get('model_accounting') or {}).get('attempted',0))
 status(state='running',case=short,stage='heuristic_tool_baseline')
 baseline_tools=heuristic_tools(folder,Path(marker['cache']))
 status(state='running',case=short,stage='tool_specialists')
 save(R/'pilot_manifest.json',dict(cases={short:dict(previous_calls=calls,max_new_calls=min(6,23-calls))}))
 import run_tool_specialists as specialist
 specialist.run_case(short)
 result=json.loads((folder/'tools/complete_tools.json').read_text())
 total=calls+int(result['actual_feedback_calls']);assert total<=23
 status(state='running',case=short,stage='advisor_export')
 execute(docker(R/'export_complete.py',[folder]),folder/'advisor_export.log')
 execute(docker(R/'all_geometry_export/export_all_geometry.py',[short]),folder/'all_geometry_export.log')
 return dict(case_id=folder.name,status='attempt_complete',scene_results=scene,tools={m:{n:dict(accepted=r['accepted_tool_hypothesis'],selection_origin=r.get('selection_origin'),review_status=r['agent_review_status']) for n,r in rows.items()} for m,rows in result['tools'].items()},tool_errors=result['errors'],total_model_calls=total,estimated_api_cost_usd=float(perception.get('estimated_api_cost_usd',0))+float((summary.get('model_accounting') or {}).get('estimated_api_cost_usd',0))+result.get('estimated_api_cost_usd',0),independent_accuracy_measured=False,physics_validated=False)
def main():
 key=sys.stdin.readline().strip();assert key,'Memory-only API credential required';os.environ['OPENAI_API_KEY']=key
 with (R/'job.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (R/'case_progress.json').exists(),'Existing attempts must be reviewed, never automatically repeated'
  manifest=json.loads((R/'job_manifest.json').read_text());rows=[]
  for batch,shorts in enumerate(manifest['batches'],1):
   if batch==1:
    while True:
     p=R/'preprocess_first_status.json';state=json.loads(p.read_text()) if p.exists() else {}
     if state.get('state')=='complete':break
     if state.get('state')=='failed':raise RuntimeError('First pair preprocessing failed')
     status(state='waiting_for_first_pair_preprocessing');time.sleep(10)
   else:preprocess(batch,shorts)
   for short in shorts:
    try:row=case_job(short,key)
    except Exception as e:
     traceback.print_exc();row=dict(case_id='snapshot_20261015_'+short,status='pipeline_failed',error=f'{type(e).__name__}: {e}');save(R/row['case_id']/'pipeline_failure.json',row)
    rows.append(row);save(R/'case_progress.json',rows)
    status(state='running',completed_cases=len(rows),total_cases=len(manifest['cases']),latest=row)
    if row['status']=='pipeline_failed':status(state='stopped_for_failure_review',completed_cases=len(rows),failure=row);return
  status(state='finalizing',completed_cases=len(rows),total_cases=len(manifest['cases']))
  execute(docker(R/'export_complete.py',['--finalize']),R/'finalize.log')
  execute(docker(R/'all_geometry_export/export_all_geometry.py',['--finalize']),R/'all_geometry_finalize.log')
  source=json.loads((R/'runtime_manifest.json').read_text());assert all(sha(R/p)==h for p,h in source.items()),'Runtime source changed'
  raw_ok=all(sha(Path(manifest['raw_root'])/('snapshot_20261015_'+c)/n)==h for c in manifest['cases'] for n,h in manifest['raw_input_sha256'][c].items());assert raw_ok
  save(R/'verification.json',dict(attempted_cases=len(rows),expected_cases=len(manifest['cases']),expected_packages=2*len(manifest['cases']),sources_unchanged=True,original_data_unchanged=raw_ok,api_budgets_verified=all(x['total_model_calls']<=23 for x in rows),independent_accuracy_measured=False,physics_validated=False))
  status(state='complete',completed_cases=len(rows),total_cases=len(manifest['cases']),advisor_archive=str(R/'all_geometry_export/advisor_overnight_remaining33_all_geometry.zip'))
if __name__=='__main__':
 try:main()
 except Exception as e:status(state='failed',error=f'{type(e).__name__}: {e}');raise
