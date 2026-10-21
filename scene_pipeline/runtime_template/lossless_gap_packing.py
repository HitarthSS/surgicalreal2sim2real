"""Lossless compact record transport for large gap states. No evidence selection."""
from collections import Counter
import json


def dump(value):return json.dumps(value,separators=(',',':'),allow_nan=False)


def pack(value):
 counts=Counter();originals={};strings=set()
 def count(x):
  token=dump(x)
  if len(token)>=6:counts[token]+=1;originals[token]=x
  if isinstance(x,str):strings.add(x)
  elif isinstance(x,dict):
   for v in x.values():count(v)
  elif isinstance(x,list):
   for v in x:count(v)
 count(value);schemas={};refs={};values={};tags={};used=set()
 def tag(prefix,index):
  key=(prefix,index)
  if key in tags:return tags[key]
  alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!#$%&()*+,-./:;<=>?[]^_`{|}~'
  n=0
  while True:
   number=n;name=''
   while True:
    name=alphabet[number%len(alphabet)]+name;number=number//len(alphabet)-1
    if number<0:break
   n+=1
   if name not in strings and name not in used:break
  tags[key]=name;used.add(name);return name
 def encode(x,expand=False):
  token=dump(x);n=counts[token];alias=tag('@',len(refs))
  # Account for reference uses, dictionary key, punctuation and stored value.
  gain=(n-1)*len(token)-(n+1)*len(dump(alias))-2
  if not expand and n>1 and gain>0:
   if token not in refs:
    refs[token]=alias;values[alias]=None;values[alias]=encode(x,True)
   return refs[token]
  if isinstance(x,dict):
   keys=tuple(x)
   if keys not in schemas:schemas[keys]=tag('~',len(schemas))
   return [schemas[keys]]+[encode(v) for v in x.values()]
  if isinstance(x,list):return [encode(v) for v in x]
  return x
 obs=encode(value)
 result=dict(encoding='Lists beginning with a schemas key are dict records: remaining elements match that ordered field list. Other lists are literal. String values matching refs keys expand recursively, preserving original types. All fields, candidates and rounded diagnostic numbers are retained exactly. Expand IDs before returning them; never output transport aliases.',schemas={name:list(keys) for keys,name in schemas.items()},refs=values,observation=obs)
 if unpack(result)!=value:raise ValueError('Lossless state round-trip failed')
 return result


def unpack(packed):
 def decode(x):
  if isinstance(x,str) and x in packed['refs']:return decode(packed['refs'][x])
  if isinstance(x,list):
   if x and isinstance(x[0],str) and x[0] in packed['schemas']:
    fields=packed['schemas'][x[0]]
    if len(fields)!=len(x)-1:raise ValueError('Record width mismatch')
    return dict(zip(fields,[decode(v) for v in x[1:]]))
   return [decode(v) for v in x]
  return x
 return decode(packed['observation'])
