"""Four-scene finite pilot with private RAM credentials and honest API status."""
from pathlib import Path
import json,os,sys,socket,threading,time,subprocess,fcntl
C=Path(__file__).resolve().parent;D=Path('/data/xliang/ismr_2027/sam3_pipeline10_20261019_v1')
sys.path.insert(0,str(C))
from contract import read,save,sha,api_attempts
from api_resilience import outcomes
KEY=None;event=threading.Event()
def receive():
 global KEY
 p=D/'.credential.sock';p.unlink(missing_ok=True)
 with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as s:
  s.bind(str(p));os.chmod(p,0o600);s.listen(1);c,_=s.accept()
  with c:
   data=b''
   while not data.endswith(b'\n') and len(data)<4096:
    v=c.recv(4096)
    if not v:break
    data+=v
   KEY=data.decode().strip();assert len(KEY)>20;event.set();c.sendall(b'accepted\n')
 p.unlink(missing_ok=True)
def audit():
 assert sha(read(C/'model.json')['checkpoint'])==read(C/'model.json')['sha256'],'Selected SAM3 checkpoint changed'
 assert all(sha(p)==v for p,v in read(D/'source_manifest.json').items()),'Recovery sources changed'
 assert all(sha(p)==v for p,v in read(D/'original_run_source_manifest.json').items()),'Original frozen run changed'
def metrics():
 rows=[]
 for cid in read(C/'recovery_scope.json')['cases']:
  f=C/cid;r=read(f/'case_result.json');s=read(f/'agent/summary.json');candidate=s.get('candidate') or {};report=candidate.get('report') or {}
  if not r:continue
  tools=read(f/'tools/complete_tools.json').get('tools',{}).get('agent',{})
  rows.append(dict(case_id=cid,status=r['status'],fragments=report.get('fragment_count'),thread_checked=report.get('numerical_checks_passed'),
   needle_available=(C/'export_context/agent'/cid/'common_needle.npz').is_file(),attachment_status=(s.get('attachment') or {}).get('status'),
   tool_meshes=sum(bool(t.get('mesh')) for t in tools.values()),accepted_tools=sum(bool(t.get('accepted_tool_hypothesis')) for t in tools.values()),
   API_execution=outcomes(f),model_attempts=len(api_attempts(f)),errors=r.get('errors',[])))
 save(D/'results_summary.json',dict(rows=rows,total=10,API_physical_attempts=sum(r['model_attempts'] for r in rows),
  independent_accuracy_measured=False,physics_validated=False));return rows
assert os.environ.get('SLURM_JOB_ID') and os.environ.get('APPTAINER_CONTAINER')
with (D/'queue.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);audit()
 save(D/'started.json',dict(slurm_job_id=os.environ['SLURM_JOB_ID'],started=time.time()))
 threading.Thread(target=receive,daemon=True).start();save(D/'job_status.json',dict(state='waiting_for_RAM_credential',total=10))
 if not event.wait(3600):raise RuntimeError('Credential unavailable')
 cases=read(C/'recovery_scope.json')['cases']
 for i,cid in enumerate(cases):
  f=C/cid
  if (f/'case_result.json').exists():continue
  try:
   from execution import execute
   execute([sys.executable,str(C/'fresh_preprocessing.py'),cid],f/'fresh_preprocessing.log',timeout=1800)
  except Exception as error:
   save(f/'case_result.json',dict(case_id=cid,status='fresh_preprocessing_failed',errors=[dict(stage='fresh_SAM3',error=str(error))],model_attempts=0))
   continue
  save(D/'job_status.json',dict(state='running',active_case=cid,completed=i,total=10,updated=time.time()))
  with (f/'worker.log').open('w') as log:
   p=subprocess.Popen([sys.executable,'-u',str(C/'case_worker.py'),str(C),cid],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   p.stdin.write((KEY+'\n').encode());p.stdin.close()
   try:code=p.wait(timeout=5400)
   except subprocess.TimeoutExpired:
    import psutil
    parent=psutil.Process(p.pid)
    for x in parent.children(recursive=True)+[parent]:
     try:x.kill()
     except psutil.NoSuchProcess:pass
    code=p.wait()
  if not (f/'case_result.json').exists():save(f/'case_result.json',dict(case_id=cid,status='worker_failed',errors=[dict(stage='worker',exit_code=code)]))
  rows=metrics();provider=read(D/'API_state/provider_state.json')
  if provider.get('blocked'):
   save(D/'pending_work.json',dict(remaining_cases=cases[i+1:],repair_cases=[r['case_id'] for r in rows if r['API_execution']['agent_work_pending']],provider=provider))
   save(D/'job_status.json',dict(state='blocked_provider',completed=i+1,total=10,error_code=provider.get('error_code')));sys.exit(2)
 rows=metrics();audit();assert sum(r['model_attempts'] for r in rows)<=230
 save(D/'verification.json',dict(source_hashes_verified=True,original_run_unchanged=True,physical_API_attempts=sum(r['model_attempts'] for r in rows),
  raw_attempt_budgets_verified=True,completed_responses=sum(r['API_execution']['completed_responses'] for r in rows),
  failed_attempts=sum(r['API_execution']['failed_attempts'] for r in rows),physics_validated=False))
 save(D/'job_status.json',dict(state='complete_with_failures' if any(r['errors'] or r['status']!='attempt_complete' or r['API_execution']['agent_work_pending'] for r in rows) else 'complete',completed=4,total=10,updated=time.time()))
