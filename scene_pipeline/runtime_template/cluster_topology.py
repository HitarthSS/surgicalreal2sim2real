"""Stereo-reviewed endpoint-cluster matchings; no synthetic bridge or multiway weld."""
import hashlib,json,itertools
import numpy as np

def project(points,K,b=0.):
 q=np.asarray(points).copy();q[:,0]-=b;v=q@K.T;return v[:,:2]/v[:,2,None]
def tangent(part,end):
 p=part['points_m'] if end==0 else part['points_m'][::-1];s=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];j=min(int(np.searchsorted(s,.002)),len(p)-1);v=p[j]-p[0];n=np.linalg.norm(v);return v/n if n>1e-10 else None

def candidates(parts,K,baseline,max_ports=8):
 ports=[(i,e) for i in range(len(parts)) for e in (0,1)];positions={p:parts[p[0]]['points_m'][0 if p[1]==0 else -1] for p in ports};remaining=set(ports);groups=[]
 while remaining:
  seed=min(remaining);group={p for p in remaining if np.linalg.norm(positions[p]-positions[seed])<=1e-6};remaining-=group
  # Require every pair, rather than transitive chaining across a distance limit.
  if not all(np.linalg.norm(positions[a]-positions[b])<=1e-6 for a,b in itertools.combinations(group,2)):continue
  if 2<len(group)<=max_ports:groups.append(sorted(group))
 rows=[]
 for group in groups:
  compatible={}
  for a,b in itertools.combinations(group,2):
   if a[0]==b[0]:continue
   if any(np.linalg.norm(project(np.vstack([positions[a],positions[b]]),K,off)[0]-project(np.vstack([positions[a],positions[b]]),K,off)[1])>.01 for off in (0.,baseline)):continue
   u,v=tangent(parts[a[0]],a[1]),tangent(parts[b[0]],b[1])
   if u is None or v is None:continue
   turn=float(np.arccos(np.clip(-u@v,-1.,1.)))
   if turn<=.2:compatible[a,b]=turn
  def match(rest):
   if not rest:yield [];return
   a=rest[0]
   for j,b in enumerate(rest[1:],1):
    if (a,b) in compatible:
     for tail in match(rest[1:j]+rest[j+1:]):yield [(a,b)]+tail
   if len(rest)%2:
    for tail in match(rest[1:]):yield tail
  combinations=list(match(group))
  if not combinations:combinations=[[(a,b)] for a,b in compatible]
  seen=set()
  for pairs in combinations:
   if not pairs:continue
   params=[[list(a),list(b)] for a,b in pairs];key=json.dumps(params)
   if key in seen:continue
   seen.add(key)
   try:apply(parts,params,K,baseline)
   except ValueError:continue
   cid=hashlib.sha256(key.encode()).hexdigest()[:12]
   rows.append(dict(candidate_id=cid,pairs=params,cluster=[list(p) for p in group],turns_rad=[compatible[a,b] for a,b in pairs],
     unresolved_ports=[list(p) for p in group if all(p not in pair for pair in pairs)],new_curve_samples=0,topology_is_hypothesis=True))
 return rows

def apply(parts,pairs,K,baseline):
 edges={};parent=list(range(len(parts)))
 def root(i):
  while parent[i]!=i:i=parent[i]
  return i
 for aa,bb in pairs:
  a,b=tuple(aa),tuple(bb)
  if a in edges or b in edges or a[0]==b[0]:raise ValueError('Endpoint reused or same-fragment cycle')
  if a[1] not in (0,1) or b[1] not in (0,1):raise ValueError('Invalid endpoint')
  ra,rb=root(a[0]),root(b[0])
  if ra==rb:raise ValueError('Closed cycle would replace the open-thread hypothesis')
  pa=parts[a[0]]['points_m'][0 if a[1]==0 else -1];pb=parts[b[0]]['points_m'][0 if b[1]==0 else -1]
  if np.linalg.norm(pa-pb)>1e-6:raise ValueError('Endpoints are not numerically coincident')
  for off in (0.,baseline):
   xy=project(np.vstack([pa,pb]),K,off)
   if np.linalg.norm(xy[0]-xy[1])>.01:raise ValueError('Stereo endpoints differ')
  u,v=tangent(parts[a[0]],a[1]),tangent(parts[b[0]],b[1])
  if u is None or v is None or np.arccos(np.clip(-u@v,-1.,1.))>.2:raise ValueError('Unsupported sharp crossing turn')
  parent[rb]=ra;edges[a]=b;edges[b]=a
 output=[];visited=set();required=('points_m','depth_inferred','geometry_inferred','observed_left','observed_right')
 for node in range(len(parts)):
  if node in visited:continue
  component=[i for i in range(len(parts)) if root(i)==root(node)]
  ends=[(i,e) for i in component for e in (0,1) if (i,e) not in edges]
  if len(ends)!=2:raise ValueError('Component must retain two distinct material terminals')
  current=min(ends);sequence=[]
  while True:
   i,entry=current
   if i in visited:raise ValueError('Repeated material strand')
   visited.add(i);sequence.append((i,entry));exit=(i,1-entry)
   if exit not in edges:break
   current=edges[exit]
  item={key:np.concatenate([parts[i][key] if entry==0 else parts[i][key][::-1] for i,entry in sequence]) for key in required}
  item['reference_xy']=project(item['points_m'],K,baseline);output.append(item)
 if len(visited)!=len(parts):raise ValueError('Source strand lost')
 return output
