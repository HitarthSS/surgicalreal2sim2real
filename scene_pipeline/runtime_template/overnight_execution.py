"""Durable stage execution: RAM credentials, heartbeats, owned-container cleanup, no host wall cutoff."""
from pathlib import Path
import json,subprocess,time,os,hashlib
BASE=Path(__file__).resolve().parent
def save(p,x):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def read(p,default=None):
 try:return json.loads(Path(p).read_text())
 except FileNotFoundError:return {} if default is None else default
def annotate(cmd,log):
 cmd=list(map(str,cmd));cid=None
 if cmd[:2]==['docker','run']:
  folder=BASE/'container_receipts';folder.mkdir(exist_ok=True)
  token=hashlib.sha256((str(log)+str(os.getpid())+str(time.time_ns())).encode()).hexdigest()[:20]
  cid=folder/(token+'.cid');hook=str(BASE/'transport_hook');found=False
  for i,item in enumerate(cmd):
   if item.startswith('PYTHONPATH='):
    cmd[i]='PYTHONPATH='+hook+':'+item.split('=',1)[1];found=True
  extra=['--label','ismr.overnight='+BASE.name,'--cidfile',str(cid)]
  if not found:extra+=['-e','PYTHONPATH='+hook]
  cmd[2:2]=extra
 return cmd,cid
def cleanup(cid):
 if cid is None or not cid.exists():return
 identifier=cid.read_text().strip()
 p=subprocess.run(['docker','inspect','--format','{{json .}}',identifier],text=True,capture_output=True,timeout=30)
 if p.returncode:return
 data=json.loads(p.stdout)
 assert data['Config']['Labels'].get('ismr.overnight')==BASE.name,'Refuse another job container'
 if data['State']['Running']:
  p=subprocess.run(['docker','stop','--time','20',identifier],text=True,capture_output=True,timeout=35)
  if p.returncode:subprocess.run(['docker','kill',identifier],capture_output=True,timeout=30)
def execute(root,cmd,log,requested_timeout=None,key=None):
 root=Path(root);log=Path(log);log.parent.mkdir(parents=True,exist_ok=True);command,cid=annotate(cmd,log);began=time.monotonic()
 receipt=log.with_suffix(log.suffix+'.execution.json')
 metadata=dict(log=str(log),requested_legacy_timeout_seconds=requested_timeout,host_wall_timeout_seconds=None,HTTP_timeout_seconds=900,container_receipt=str(cid) if cid else None,credential_in_arguments=False,scientific_retry=False)
 save(receipt,metadata)
 with log.open('w') as stream:
  p=subprocess.Popen(command,stdin=subprocess.PIPE if key else subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
  if key:p.stdin.write((key+'\n').encode());p.stdin.close()
  while True:
   try:code=p.wait(timeout=30);break
   except subprocess.TimeoutExpired:
    metadata.update(state='running',child_pid=p.pid,elapsed_seconds=round(time.monotonic()-began,1),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save(receipt,metadata)
    status=read(root/'job_status.json');status.update(stage_elapsed_seconds=metadata['elapsed_seconds'],heartbeat_utc=metadata['updated_utc'],execution_receipt=str(receipt));save(root/'job_status.json',status)
 metadata.update(state='complete' if code==0 else 'failed',exit_code=code,elapsed_seconds=round(time.monotonic()-began,1));save(receipt,metadata)
 cleanup(cid)
 if code:raise RuntimeError('Stage failed; log preserved: '+str(log))
 return 0
