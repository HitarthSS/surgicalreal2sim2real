"""One detached, finite server queue for all remaining current-pipeline scenes."""
from pathlib import Path
import json,sys,os,time,fcntl,subprocess,hashlib,traceback
R=Path(__file__).resolve().parent
from overnight_execution import save,read
def status(state,**details):
 save(R/'overnight_status.json',dict(state=state,pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**details))
def verify(root):
 source=read(root/'runtime_manifest.json');assert source
 assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in source.items()),'Protected source mismatch'
def wait(root):
 while True:
  previous=read(root/'job_status.json');name=previous.get('state')
  if name in ['complete','complete_with_failures']:
   assert read(root/'verification.json')['api_budgets_verified'];verify(root);return
  if name in ['failed','stopped_for_failure_review','blocked_by_predecessor_failure']:
   raise RuntimeError('Previous scientific run needs review: '+str(root))
  status('waiting_for_predecessor',predecessor=str(root),previous_status=previous,previous_attempts=len(read(root/'case_progress.json',[])),queue_timeout_seconds=None)
  subprocess.run([sys.executable,str(R/'overnight_pack.py'),'--dashboard'],check=True)
  time.sleep(30)
def run(root,key):
 verify(root)
 with (root/'overnight_driver_stdout.log').open('w') as log:
  p=subprocess.Popen([sys.executable,str(R/'overnight_driver.py'),str(root)],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
  old=read(root/'job_pid.json');save(root/'overnight_previous_dispatch.json',old)
  save(root/'job_pid.json',dict(pid=p.pid,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),execution_owner=str(R),resumed_existing_attempts=False))
  p.stdin.write((key+'\n').encode());p.stdin.close()
  while True:
   try:code=p.wait(timeout=30);break
   except subprocess.TimeoutExpired:
    status('running',active_root=str(root),driver_pid=p.pid,active_status=read(root/'job_status.json'),attempts=len(read(root/'case_progress.json',[])),queue_timeout_seconds=None)
  if code:raise RuntimeError('Scientific driver stopped for review: '+str(root))
  assert read(root/'job_status.json')['state'] in ['complete','complete_with_failures']
def main():
 key=sys.stdin.readline().strip();assert key
 with (R/'overnight_master.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  config=read(R/'overnight_manifest.json');Q=Path(config['existing_queue']);P=Path(config['active_predecessor']);ownership=read(R/'ownership.json')
  if ownership.get('replaced_waiting_gate'):
   with (Q/'dependency_gate.lock').open('a') as gate:
    fcntl.flock(gate,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (Q/'case_progress.json').exists(),'Refuse to repeat existing queued attempts'
    wait(P)
   run(Q,key)
  else:wait(Q)
  run(R,key);del key
  status('packaging_all63',attempts=len(read(R/'case_progress.json',[])))
  p=subprocess.run([sys.executable,str(R/'overnight_pack.py'),'--finalize'],capture_output=True,text=True)
  (R/'overnight_pack_stdout.log').write_text(p.stdout+p.stderr)
  if p.returncode:raise RuntimeError('Final package audit failed; inspect overnight_pack_stdout.log')
  status('complete',coverage=read(R/'all63_verification.json'),queue_timeout_seconds=None,independent_accuracy_measured=False,physics_validated=False)
if __name__=='__main__':
 try:main()
 except Exception as error:
  traceback.print_exc();status('stopped_for_review',error=type(error).__name__+': '+str(error),scientific_attempts_preserved=True);raise
