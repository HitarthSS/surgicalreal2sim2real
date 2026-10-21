"""Actual model transport for existing scene controllers; no model-written code.

Accepts the existing text/image message interface and records exact observations,
API responses, token usage, model identity, and source/image hashes per call.
"""
import base64
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
from api_resilience import (admission,record_failure,record_success,classify_http,retry_delay,reserve_input,ProviderBlocked,scrub)


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class OpenAISceneModel:
    def __init__(self, out, key_file=None, model='gpt-6.1-sol', max_calls=4):
        if model != 'gpt-6.1-sol':
            raise ValueError('This pilot backend is frozen to gpt-6.1-sol')
        if type(max_calls) is not int or not 1 <= max_calls <= 6:
            raise ValueError('Model call budget must be 1–6')
        self.root = Path(out)
        self.root.mkdir(parents=True, exist_ok=False)
        self.key = os.environ.get('OPENAI_API_KEY')
        if not self.key and key_file:
            self.key = Path(key_file).read_text().strip()
        if not self.key:
            raise ValueError('OpenAI API credentials unavailable')
        self.model, self.max_calls, self.index = model, max_calls, 0
        self.logical_index = 0
        self.last_usage = None
        self.estimated_api_cost_usd = 0.
        self.manifest = dict(backend='OpenAI Responses API', name=model, reasoning_effort='low',
            image_detail='original', max_output_tokens=2048, max_model_calls=max_calls,
            retries=2, retries_count_as_physical_attempts=True, max_transport_attempts_per_decision=3, store=False, source_sha256=hash_file(__file__),
            cost_basis='Estimated standard short-context rates: $2/M input, $10/M output; invoice unverified.')
        self.write(self.root / 'backend_manifest.json', self.manifest)

    @staticmethod
    def write(path, value):
        Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

    def __call__(self, messages):
        if self.index >= self.max_calls:raise ValueError('Model attempt budget exhausted')
        self.logical_index += 1
        inputs, images = [], {}
        for message in messages:
            content = []
            for part in message['content']:
                if part['type'] == 'text':content.append(dict(type='input_text', text=part['text']))
                elif part['type'] == 'image':
                    path=Path(part['url']).resolve()
                    if not path.is_file():raise ValueError('Missing declared image artifact')
                    mime=mimetypes.guess_type(path.name)[0]
                    if mime not in ('image/png','image/jpeg'):raise ValueError('Original PNG/JPEG images required')
                    images[str(path)]=hash_file(path)
                    content.append(dict(type='input_image',detail='original',image_url='data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode('ascii')))
                else:raise ValueError('Unsupported message item')
            inputs.append(dict(role=message['role'],content=content))
        logical_start=time.monotonic()
        for transport_attempt in range(1,4):
            if self.index>=self.max_calls:raise ValueError('Model attempt budget exhausted')
            try:admission()
            except ProviderBlocked as error:
                self.write(self.root/f'provider_deferred_{self.logical_index:03d}.json',dict(status='pending_provider',error_code=scrub(str(error),self.key),network_attempted=False))
                raise
            self.index+=1;folder=self.root/f'request_{self.index:03d}'
            try:reserve_input(folder,dict(messages=messages,image_sha256=images,model=self.manifest,logical_decision=self.logical_index,transport_attempt=transport_attempt))
            except Exception as error:
                self.index-=1
                self.write(self.root/f'provider_deferred_{self.logical_index:03d}.json',dict(status='pending_budget',error_type=type(error).__name__,network_attempted=False))
                raise
            start=time.perf_counter();result=dict(status='failed',actual_model_used=False,logical_decision=self.logical_index,transport_attempt=transport_attempt)
            failure=None
            try:
                payload=dict(model=self.model,store=False,reasoning={'effort':'low'},max_output_tokens=2048,input=inputs)
                req=urllib.request.Request('https://api.openai.com/v1/responses',data=json.dumps(payload).encode(),method='POST',headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
                with urllib.request.urlopen(req,timeout=900) as response:response=json.load(response)
                self.write(folder/'api_response.json',response)
                raw='\n'.join(p['text'] for item in response.get('output',[]) if item.get('type')=='message' for p in item.get('content',[]) if p.get('type')=='output_text')
                (folder/'raw_response.txt').write_text(raw+'\n')
                usage=response.get('usage') or {}
                self.last_usage=dict(input_tokens_total=usage.get('input_tokens'),output_tokens=usage.get('output_tokens'),reasoning_tokens=usage.get('output_tokens_details',{}).get('reasoning_tokens'),cached_tokens=usage.get('input_tokens_details',{}).get('cached_tokens'),accounting='actual_api_response',model_returned=response.get('model'),api_usage=usage)
                if usage.get('input_tokens') is not None and usage.get('output_tokens') is not None:
                    cost=(usage['input_tokens']*2+usage['output_tokens']*10)/1e6
                    self.last_usage['estimated_api_cost_usd']=cost;self.estimated_api_cost_usd+=cost
                result.update(actual_model_used=True,usage=self.last_usage,response_id=response.get('id'),api_status=response.get('status'))
                if {p:hash_file(p) for p in images}!=images:raise ValueError('Image changed during inference')
                if response.get('model')!=self.model or response.get('status')!='completed' or not raw.strip():raise ValueError('Unexpected model identity or incomplete response')
                result['status']='completed';record_success();return raw
            except urllib.error.HTTPError as error:
                failure=classify_http(error.code,error.read(32768).decode('utf8','replace'),dict(error.headers),self.key)
                result.update(failure,exception_type='HTTPError');record_failure(failure)
            except Exception as error:
                result.update(error_type=type(error).__name__,error_message=scrub(str(error),self.key),http_status=getattr(error,'code',None))
                raise
            finally:
                result['runtime_seconds']=time.perf_counter()-start;self.write(folder/'result.json',result)
            delay=retry_delay(failure,transport_attempt,time.monotonic()-logical_start,self.max_calls-self.index)
            if delay is None:raise ProviderBlocked(failure.get('error_code') or failure['classification'])
            self.write(folder/'retry_policy.json',dict(wait_seconds=delay,next_attempt_counts_towards_budget=True))
            time.sleep(delay)
# Timed comparison provider adapter. The original GPT implementation above stays unchanged.
if __import__('os').environ.get('COMPARE_PROVIDER')=='gemma':
 import json as _json,time as _time
 from pathlib import Path as _Path
 from model_bridge import Client as _Client,RPCBackend as _RPC
 class OpenAISceneModel:
  def __init__(self,out,key_file=None,max_calls=1,**kwargs):
   self.root=_Path(out);self.root.mkdir(parents=True,exist_ok=False)
   self.max_calls=max_calls;self.index=0;self.last_usage=None;self.estimated_api_cost_usd=0.
   self.manifest=dict(name='google/gemma-4-31B-it',revision='842da3794eaa0b77d5f08bae87a17459d91ff475',retries=0,
    backend='resident local model via private Unix socket',max_model_calls=max_calls,full_images=True)
   (self.root/'backend_manifest.json').write_text(_json.dumps(self.manifest,indent=2))
   self.rpc=_RPC(_Client(__import__('os').environ['ISMR_LOCAL_MODEL_SOCKET'],'gemma'),str(self.root),max_calls)
  def __call__(self,messages):
   if self.index>=self.max_calls:raise ValueError('Stage model budget exhausted')
   self.index+=1;folder=self.root/f'request_{self.index:03d}';start=_time.perf_counter()
   from api_resilience import reserve_input
   reserve_input(folder,dict(messages=messages,model=self.manifest))
   result=dict(status='failed',actual_model_used=False)
   try:
    raw=self.rpc(messages);self.last_usage=self.rpc.last_usage
    (folder/'raw_response.txt').write_text(raw+'\n')
    result.update(status='completed',actual_model_used=True,usage=self.last_usage);return raw
   except Exception as e:result.update(error=type(e).__name__+': '+str(e));raise
   finally:
    result['runtime_seconds']=_time.perf_counter()-start
    (folder/'result.json').write_text(_json.dumps(result,indent=2))
