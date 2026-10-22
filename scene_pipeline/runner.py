"""Durable sequential scene runner; failed attempts are retained, never replayed."""
from pathlib import Path
import fcntl,json,os,signal,subprocess,sys,time,hashlib

def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def run(command,log,env,timeout,key=None):
 log=Path(log);log.parent.mkdir(parents=True,exist_ok=True);start=time.monotonic()
 with log.open('w') as out:
  p=subprocess.Popen(command,env=env,stdin=subprocess.PIPE if key else subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
  if key:p.stdin.write((key+'\n').encode());p.stdin.close()
  try:code=p.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise TimeoutError('Scene process group exceeded watchdog; outputs retained')
 if code:raise RuntimeError(f'Stage exited {code}: {log}')
 return time.monotonic()-start

def verify_sources(root):
 manifest=Path(root)/'source_hashes.json'
 if not manifest.exists():return
 for name,expected in json.loads(manifest.read_text()).items():
  assert hashlib.sha256((Path(root)/name).read_bytes()).hexdigest()==expected,'Runtime source changed: '+name

def main(root):
 root=Path(root).resolve();verify_sources(root);scope=json.loads((root/'scope.json').read_text());assert os.environ.get('SLURM_JOB_ID') and os.environ.get('APPTAINER_CONTAINER'),'Run inside Slurm/Apptainer'
 cfg=json.loads((root/'config.json').read_text())
 key=sys.stdin.readline().strip();assert key,'Credential required on stdin, never in arguments';env=os.environ.copy();env.pop('OPENAI_API_KEY',None);env['ISMR_API_STATE_DIR']=str(root/'API_state');env['ISMR_FFS_ROOT']=cfg['ffs_root'];env['ISMR_FFS_CHECKPOINT']=cfg['ffs_checkpoint'];rows=[]
 with (root/'queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for case in scope['scenes']:
   cid=case['case_id'];C=root/'gpt'/('day'+cid.split('_')[1]);F=C/cid;done=F/'run_result.json';started=F/'run_started.json'
   if done.exists():rows.append(json.loads(done.read_text()));continue
   if started.exists():
    row=dict(case_id=cid,state='interrupted_preserved',processing_completed=False,error='A prior scene attempt exists; no automatic scientific replay');save(done,row);rows.append(row);continue
   save(started,dict(started=time.time(),job_id=os.environ['SLURM_JOB_ID']));begin=time.monotonic();stage='preprocessing';row=dict(case_id=cid,difficulty=case['difficulty'],processing_completed=False)
   env['ISMR_API_SCENE_DIR']=str(F);env['PYTHONPATH']=os.pathsep.join([str(C/'transport_hook'),str(C),str(C/'executed_sources'),'/opt/ismr_sam3'])
   try:
    pre=run([sys.executable,str(C/'prepare_scene.py'),cid],F/'run_preprocessing.log',env,1800)
    stage='full_pipeline';elapsed=run([sys.executable,str(C/'case_worker.py'),str(C),cid],F/'run_pipeline.log',env,3600,key)
    result=json.loads((F/'case_result.json').read_text());row.update(state=result['status'],processing_completed=True,model_attempts=result['model_attempts'],component_errors=result.get('errors',[]),preprocessing_seconds=pre,pipeline_seconds=elapsed)
   except Exception as e:row.update(state='failed_preserved',stage=stage,error=str(e).replace(key,'[REDACTED]'))
   if cfg.get('background',{}).get('enabled'):
    # Optional background failure never discards or reruns foreground geometry.
    began_background=time.monotonic()
    try:
     from .background_stage import run_case as build_background
     os.environ['ISMR_API_STATE_DIR']=str(root/'API_state')
     background=build_background(F/'case.json',F/'background',cfg['background'],key)
     row['background']=dict(state=background['status'],selection_source=background['selection_source'])
    except Exception as error:row['background']=dict(state='failed_preserved',error_type=type(error).__name__)
    row['background_seconds']=time.monotonic()-began_background
   row.update(elapsed_seconds=time.monotonic()-begin,scientific_conversion_not_implied=True);save(done,row);rows.append(row);save(root/'progress.json',dict(completed=len(rows),total=len(scope['scenes']),rows=rows));print(json.dumps(row),flush=True)
  verify_sources(root)
  save(root/'queue_complete.json',dict(completed=len(rows),total=len(scope['scenes']),rows=rows,all_processing_terminal=True,independent_accuracy_measured=False))
 key=''
