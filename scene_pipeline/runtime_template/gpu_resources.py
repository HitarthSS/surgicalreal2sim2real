"""One serialized GPU tool at a time. Prefer GPU0; use GPU1 if0 lacks6GiB."""
from pathlib import Path
import subprocess,fcntl,time,json

def execute_gpu(runtime,cmd,log,case,stage):
 with (runtime.ROOT/'gpu0.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  while True:
   rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
   free={int(a):int(b) for a,b in (l.split(',') for l in rows.splitlines())}
   available=[i for i in [0,1] if free.get(i,0)>=6144]
   if available:break
   runtime.status(state='waiting_for_gpu',case=case,stage=stage,free_mib=free);time.sleep(10)
  chosen=available[0]
  command=[f'device={chosen}' if x in ('device=0','device=1') else x for x in cmd]
  with (runtime.ROOT/'gpu_allocations.jsonl').open('a') as f:f.write(json.dumps(dict(case=case,stage=stage,gpu=chosen,free_mib=free,required_mib=6144,process_cap_gib=8,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))+'\n')
  runtime.status(state='running',case=case,stage=stage,gpu=chosen);runtime.execute(command,log,1200)
