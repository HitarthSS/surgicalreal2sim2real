"""Keep a typed credential in server RAM until the queued job accepts it."""
from pathlib import Path
import getpass,json,os,socket,time

def main(root):
 root=Path(root).resolve();marker=root/'credential_relay.json'
 if marker.exists():raise ValueError('Existing relay must be inspected before launching another')
 key=getpass.getpass('OpenAI API key (hidden; retained only in server RAM): ').encode();assert 20<len(key)<4096
 # Create marker before fork so a second invocation cannot silently overlap it.
 with marker.open('x') as f:json.dump(dict(state='starting',credential_written_to_disk=False),f)
 pid=os.fork()
 if pid:print('RAM relay started:',pid);return
 os.setsid();fd=os.open('/dev/null',os.O_RDWR)
 for target in (0,1,2):os.dup2(fd,target)
 if fd>2:os.close(fd)
 def status(state):marker.write_text(json.dumps(dict(state=state,pid=os.getpid(),credential_written_to_disk=False),indent=2))
 status('waiting_for_job_socket')
 try:
  deadline=time.monotonic()+7*86400
  while time.monotonic()<deadline:
   p=root/'.credential.sock'
   if p.exists():
    try:
     with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as s:
      s.settimeout(10);s.connect(str(p));s.sendall(key+b'\n');assert s.recv(128).strip()==b'accepted'
     key=b'';status('delivered');os._exit(0)
    except (FileNotFoundError,ConnectionRefusedError):pass
   time.sleep(5)
  status('expired')
 except BaseException:status('failed')
 finally:key=b''
 os._exit(1)
