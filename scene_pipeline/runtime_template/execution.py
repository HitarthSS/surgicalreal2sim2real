"""Run legacy stage commands inside the enclosing Slurm Apptainer allocation.

No Docker execution, scheduler bypass, GPU selection, or API retry occurs here.
"""
from pathlib import Path
import fcntl,json,os,signal,subprocess,sys,time
from contract import read,save
R=Path(__file__).resolve().parent

def normalize(cmd):
    cmd=list(map(str,cmd));env=os.environ.copy()
    if cmd[:2]==['docker','run']:
        # Legacy builders consistently end with --entrypoint python3 IMAGE SCRIPT.
        i=cmd.index('--entrypoint')
        assert cmd[i+1] in ('python3','python') and cmd[i+2].startswith('sha256:')
        for j,arg in enumerate(cmd[:i]):
            if arg=='-e':
                name,value=cmd[j+1].split('=',1)
                if name not in ('HOME','USER','LOGNAME','TORCH_EXTENSIONS_DIR','TORCHINDUCTOR_CACHE_DIR'):
                    env[name]=value
        cmd=[sys.executable]+cmd[i+3:]
    assert cmd and cmd[0] not in ('docker','apptainer','srun','sbatch')
    env['PYTHONPATH']=os.pathsep.join([str(R/'transport_hook'),str(R),env.get('PYTHONPATH',''),'/opt/ismr_sam3'])
    return cmd,env

def execute(cmd,log,timeout=900,key=None):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('APPTAINER_CONTAINER'),'A Slurm Apptainer job is required'
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True);receipt=Path(str(log)+'.execution.json')
    previous=read(receipt)
    if previous.get('state')=='complete' and previous.get('exit_code')==0:return 0
    if receipt.exists():
        history=log.parent/'execution_history';history.mkdir(exist_ok=True);stamp=str(time.time_ns())
        for p in (receipt,log):
            if p.exists():p.rename(history/(stamp+'_'+p.name))
    command,env=normalize(cmd);began=time.monotonic();limit=min(max(timeout,2700 if key else 1200),2700 if key else 1800)
    meta=dict(state='running',command=command,credential_in_arguments=False,slurm_job_id=os.environ['SLURM_JOB_ID'],
              container=os.environ['APPTAINER_CONTAINER'],scientific_retry=False,stage_wall_limit_seconds=limit)
    save(receipt,meta)
    with log.open('w') as out:
        p=subprocess.Popen(command,env=env,stdin=subprocess.PIPE if key else subprocess.DEVNULL,
                           stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        if key:p.stdin.write((key+'\n').encode());p.stdin.close()
        while True:
            try:code=p.wait(timeout=20);break
            except subprocess.TimeoutExpired:
                elapsed=time.monotonic()-began;meta.update(elapsed_seconds=elapsed,child_pid=p.pid,updated=time.time());save(receipt,meta)
                if elapsed>limit:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                    code=124;break
    meta.update(state='complete' if code==0 else 'failed',exit_code=code,elapsed_seconds=time.monotonic()-began);save(receipt,meta)
    if code:raise RuntimeError('Stage failed; outputs retained: '+str(log))
    return 0

def gpu_execute(runtime,cmd,log,case,stage):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') is not None
    with (R/'gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        save(R/'gpu_status.json',dict(state='running',case=case,stage=stage,slurm_job_id=os.environ['SLURM_JOB_ID'],
                                      assigned_device_mask=os.environ['CUDA_VISIBLE_DEVICES'],selection='Slurm',process_cap_gib=8))
        try:return execute(cmd,log)
        finally:save(R/'gpu_status.json',dict(state='released',case=case,stage=stage))

def cpu(script,args=(),root=None,tracker=False,network=False):
    return [sys.executable,str(script)]+list(map(str,args))
