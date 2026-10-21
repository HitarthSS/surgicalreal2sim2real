"""Private Unix-socket transport for full-view images and local VLM decisions.

SSH forwards the socket between the two authorized hosts. No listening TCP port,
cloud API, credentials, generation retries or geometry edits are involved.
"""
import base64
import hashlib
import json
import socket
import struct
from pathlib import Path

MAX_BYTES=48*1024*1024

def read_exact(sock,n):
    chunks=[]
    while n:
        part=sock.recv(min(n,1024*1024))
        if not part:raise ConnectionError('Incomplete model transport frame')
        chunks.append(part);n-=len(part)
    return b''.join(chunks)

def send(sock,value):
    data=json.dumps(value,allow_nan=False,separators=(',',':')).encode()
    if len(data)>MAX_BYTES:raise ValueError('Model transport size limit exceeded')
    sock.sendall(struct.pack('!I',len(data))+data)

def receive(sock):
    n=struct.unpack('!I',read_exact(sock,4))[0]
    if not 0<n<=MAX_BYTES:raise ValueError('Invalid model transport size')
    return json.loads(read_exact(sock,n))

def bundle(messages):
    copied=json.loads(json.dumps(messages));assets={}
    for message in copied:
        for part in message['content']:
            if part['type']!='image':continue
            data=Path(part['url']).read_bytes();key=hashlib.sha256(data).hexdigest()
            assets[key]=base64.b64encode(data).decode();part['url']=key
    return copied,assets

def restore(messages,assets,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    for key,encoded in assets.items():
        data=base64.b64decode(encoded,validate=True)
        if hashlib.sha256(data).hexdigest()!=key:raise ValueError('Image payload hash mismatch')
        path=root/(key+'.png')
        if path.exists():
            if path.read_bytes()!=data:raise ValueError('Existing image asset changed')
        else:path.write_bytes(data)
    for message in messages:
        for part in message['content']:
            if part['type']=='image':
                if part['url'] not in assets:raise ValueError('Undeclared image asset')
                part['url']=str(root/(part['url']+'.png'))
    return messages

class Client:
    def __init__(self,path,label):self.path=str(path);self.label=label
    def request(self,operation,**payload):
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as sock:
            sock.settimeout(1200);sock.connect(self.path)
            send(sock,dict(operation=operation,label=self.label,**payload));reply=receive(sock)
        if not reply.get('ok'):raise RuntimeError('Local VLM: '+reply.get('error','unknown failure'))
        return reply
    def stats(self):return self.request('stats')['accounting']

class RPCBackend:
    """Stage-local counter plus a server-enforced global 23-attempt boundary."""
    def __init__(self,client,stage,max_calls):
        self.client=client;self.stage=stage;self.max_calls=max_calls
        self.index=0;self.last_usage=None;self.estimated_api_cost_usd=0.
    def __call__(self,messages):
        if self.index>=self.max_calls:raise ValueError('Stage model budget exhausted')
        self.index+=1;self.last_usage=None
        packed,assets=bundle(messages)
        reply=self.client.request('infer',role=self.stage,messages=packed,assets=assets)
        self.last_usage=reply['usage'];return reply['raw']

class SceneModels:
    def __init__(self,client):self.client=client
    def infer(self,role,messages,budget,charged=False):
        if not charged:budget.consume('model')
        return RPCBackend(self.client,'scene.'+role,1)(messages)
    def accounting(self):return self.client.stats()
