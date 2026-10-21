"""Receive a credential through a private RAM socket, then invoke the runner."""
from pathlib import Path
import socket,os,subprocess,sys
root=Path(sys.argv[1]).resolve();path=root/'.credential.sock'
assert not path.exists(),'Existing credential socket must be inspected rather than replaced'
with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
 server.bind(str(path));os.chmod(path,0o600);server.listen(1);server.settimeout(3600);conn,_=server.accept()
 with conn:
  data=b''
  while not data.endswith(b'\n'):
   piece=conn.recv(4096)
   if not piece:raise RuntimeError('Credential transport closed')
   data+=piece
   if len(data)>4096:raise ValueError('Invalid credential payload')
  conn.sendall(b'accepted\n')
path.unlink()
result=subprocess.run([sys.executable,'-m','scene_pipeline','run',str(root)],input=data);data=b'';raise SystemExit(result.returncode)
