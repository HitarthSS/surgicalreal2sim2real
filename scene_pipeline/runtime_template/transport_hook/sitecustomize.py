"""Execution-only allowance for the existing OpenAI transport; no retries or payload changes."""
import urllib.request
import os
from urllib.parse import urlsplit
TIMEOUT_SECONDS=900
def install():
 original=urllib.request.urlopen
 if getattr(original,'_ismr_overnight_transport',False):return
 def request(url,*args,**kwargs):
  address=getattr(url,'full_url',url)
  if isinstance(address,str) and urlsplit(address).hostname=='api.openai.com':
   if os.environ.get('ISMR_NO_API')=='1':raise RuntimeError('External API disabled for baseline-only precomputation')
   if len(args)>=2:
    values=list(args);value=values[1];values[1]=max(value,TIMEOUT_SECONDS) if isinstance(value,(int,float)) else TIMEOUT_SECONDS;args=tuple(values)
   else:
    value=kwargs.get('timeout');kwargs['timeout']=max(value,TIMEOUT_SECONDS) if isinstance(value,(int,float)) else TIMEOUT_SECONDS
  return original(url,*args,**kwargs)
 request._ismr_overnight_transport=True
 urllib.request.urlopen=request
install()
