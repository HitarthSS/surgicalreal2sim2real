"""Run only unattempted cases with the frozen scientific case function and durable execution."""
from pathlib import Path
import json,sys,os,time,fcntl,traceback,hashlib,subprocess,importlib.util
BASE=Path(__file__).resolve().parent
from overnight_execution import save,read,execute
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()
def audit(root,job,c):
 assert all(sha(root/n)==h for n,h in read(root/'runtime_manifest.json').items()),'Protected source changed'
 assert all(sha(Path(job['raw_root'])/('snapshot_20261015_'+c)/n)==h for n,h in job['raw_input_sha256'][c].items()),'Raw input changed'
 F=root/('snapshot_20261015_'+c);counts={};calls=0
 for p in F.rglob('input.json'):
  if not (p.parent.parent/'backend_manifest.json').exists():continue
  x=read(p);assert x['model']['name']=='gpt-6.1-sol' and x['model']['retries']==0
  stage=p.relative_to(F).parts[0];counts[stage]=counts.get(stage,0)+1;calls+=1
 assert calls<=23 and counts.get('perception',0)<=5 and counts.get('agent_model_requests',0)<=12 and counts.get('tool_agent',0)<=6,'API budget exceeded'
 return calls
def main():
 root=Path(sys.argv[1]).resolve();key=sys.stdin.readline().strip();assert key
 os.environ['OPENAI_API_KEY']=key
 sys.path[:0]=[str(BASE/'transport_hook'),str(root),str(root/'perception_runtime')]
 spec=importlib.util.spec_from_file_location('ismr_overnight_transport',BASE/'transport_hook/sitecustomize.py');hook=importlib.util.module_from_spec(spec);spec.loader.exec_module(hook)
 import coordinator as frozen
 frozen.execute=lambda cmd,log,timeout=900,key=None:execute(root,cmd,log,timeout,key)
 import run_perception as perception
 perception.execute=lambda cmd,log,timeout=600:execute(root,cmd,log,timeout)
 with (root/'job.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  job=read(root/'job_manifest.json');rows=read(root/'case_progress.json',[]);attempted={x['case_id'] for x in rows}
  assert len(rows)==len(attempted),'Duplicate prior attempts'
  failures=0;last_failure=None
  for batch,shorts in enumerate(job['batches'],1):
   for c in shorts:
    cid='snapshot_20261015_'+c
    if cid in attempted:continue
    audit(root,job,c)
    save(root/'active_case.json',dict(case=c,case_id=cid,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),driver_pid=os.getpid()))
    try:row=frozen.case_job(c,key);failures=0;last_failure=None
    except Exception as error:
     traceback.print_exc();row=dict(case_id=cid,status='pipeline_failed',error=type(error).__name__+': '+str(error),failure_preserved=True,retried=False)
     save(root/cid/'pipeline_failure.json',row)
     signature=str(error).replace(str(root/cid),'CASE')
     failures=failures+1 if signature==last_failure else 1;last_failure=signature
    row['audited_model_attempts']=audit(root,job,c);rows.append(row);attempted.add(cid);save(root/'case_progress.json',rows)
    frozen.status(state='running',completed_cases=len(rows),total_cases=len(job['cases']),latest=row)
    subprocess.run([sys.executable,str(BASE/'overnight_pack.py'),'--dashboard'],check=True)
    if failures>=3:
     frozen.status(state='stopped_for_failure_review',reason='Three consecutive identical infrastructure/stage failures; all attempts retained',completed_cases=len(rows));return 2
  assert attempted=={'snapshot_20261015_'+c for c in job['cases']},'Coverage mismatch'
  frozen.status(state='finalizing',completed_cases=len(rows),total_cases=len(job['cases']))
  if all(r['status']=='attempt_complete' for r in rows):
   execute(root,frozen.docker(root/'export_complete.py',['--finalize']),root/'finalize.log')
   execute(root,frozen.docker(root/'all_geometry_export/export_all_geometry.py',['--finalize']),root/'all_geometry_finalize.log')
  else:
   save(root/'partial_finalization.json',dict(failed_cases=[r['case_id'] for r in rows if r['status']!='attempt_complete'],available_packages_preserved=True))
  save(root/'verification.json',dict(attempted_cases=len(rows),expected_cases=len(job['cases']),expected_packages=2*len(job['cases']),sources_unchanged=True,original_data_unchanged=True,api_budgets_verified=True,failed_pipeline_attempts=sum(r['status']!='attempt_complete' for r in rows),independent_accuracy_measured=False,physics_validated=False))
  frozen.status(state='complete' if all(r['status']=='attempt_complete' for r in rows) else 'complete_with_failures',completed_cases=len(rows),total_cases=len(job['cases']))
  subprocess.run([sys.executable,str(BASE/'overnight_pack.py'),'--dashboard'],check=True)
 return 0
if __name__=='__main__':raise SystemExit(main())
