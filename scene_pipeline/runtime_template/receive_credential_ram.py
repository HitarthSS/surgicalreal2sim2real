"""Deliver one encrypted-stdin credential from server RAM when the queued job starts."""
from pathlib import Path
import json,os,socket,sys,time
ROOT=Path('/data/xliang/ismr_2027/sam3_pipeline10_20261019_v1')
key=sys.stdin.buffer.readline().strip();assert 20<len(key)<4096
pid=os.fork()
if pid:
 print(json.dumps(dict(state='RAM_relay_started',pid=pid,credential_written_to_disk=False)),flush=True)
 sys.exit(0)
os.setsid()
fd=os.open('/dev/null',os.O_RDWR)
for target in (0,1,2):os.dup2(fd,target)
if fd>2:os.close(fd)
status=ROOT/'credential_relay_status.json'
def save(state,**kw):
 temp=status.with_suffix('.tmp');temp.write_text(json.dumps(dict(state=state,pid=os.getpid(),credential_written_to_disk=False,updated=time.time(),**kw),indent=2));temp.replace(status)
save('waiting_for_job_socket')
try:
 deadline=time.monotonic()+24*3600
 while time.monotonic()<deadline:
  path=ROOT/'.credential.sock'
  if path.exists():
   try:
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as s:
     s.settimeout(10);s.connect(str(path));s.sendall(key+b'\n');reply=s.recv(128)
     assert reply.strip()==b'accepted'
    key=b'';save('delivered_to_job');os._exit(0)
   except (ConnectionRefusedError,FileNotFoundError):pass
  time.sleep(5)
 save('expired_without_job_socket')
except BaseException as error:save('failed',error_type=type(error).__name__)
finally:key=b''
