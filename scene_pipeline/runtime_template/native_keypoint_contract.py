"""Native dVRK jaw-endpoint semantics; no guessed mesh vertex correspondence."""
import math,numpy as np
NAMES=('jawright_tip','jawleft_tip')
OFFSETS={'initialization':np.array([[0.,.0004,.009],[0.,-.0004,.009]]),'tracking':np.array([[0.,.0004,.0096],[0.,-.0004,.0096]])}

def validate_view(view,size):
 required={'jaw_tips_xy','jaw_tip_visibility','jaw_tip_reason'}
 if not required<=set(view):raise ValueError('Explicit jaw-tip visibility, coordinates and reason required in every view')
 tips=view['jaw_tips_xy'];status=view['jaw_tip_visibility'];w,h=size
 if status not in ('both_visible','not_both_visible'):raise ValueError('Unknown jaw visibility')
 if not isinstance(view['jaw_tip_reason'],str) or not 1<=len(view['jaw_tip_reason'])<=300:raise ValueError('Explain visible pair or unavailable landmarks')
 if not isinstance(tips,list) or len(tips)!=(2 if status=='both_visible' else 0):raise ValueError('Both visible tips require2points; unavailable pair must not invent tips')
 for p in tips:
  if not isinstance(p,list) or len(p)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in p) or not 0<=p[0]<w or not 0<=p[1]<h:raise ValueError('Finite ORIGINAL full-frame pixel coordinates required')
 return tips

def body_landmarks(rotations,translations,phase='tracking'):
 if phase not in OFFSETS:raise ValueError('Unknown native phase')
 r,t=np.asarray(rotations),np.asarray(translations)
 if r.shape!=(4,3,3) or t.shape!=(4,3):raise ValueError('Four native FK body frames required')
 return np.asarray([OFFSETS[phase][j]@r[j+2].T+t[j+2] for j in range(2)])
def camera_landmarks(body,pose):return np.asarray(body)@np.asarray(pose)[:3,:3].T+np.asarray(pose)[:3,3]
def project(points,K,baseline=0.):
 p=np.array(points,float,copy=True);p[:,0]-=baseline
 if p.shape!=(2,3) or not np.isfinite(p).all() or (p[:,2]<=0).any():raise ValueError('Invalid projected native jaw tips')
 q=p@np.asarray(K).T;return q[:,:2]/q[:,2,None]
def pair_error(predicted,observed):
 a,b=np.asarray(predicted,float),np.asarray(observed,float)
 if a.shape!=(2,2) or b.shape!=(2,2) or not np.isfinite([a,b]).all():raise ValueError('Exactly two finite 2D tips per set')
 direct=np.linalg.norm(a-b,axis=1);reverse=np.linalg.norm(a-b[::-1],axis=1);dist=direct if direct.sum()<=reverse.sum() else reverse
 return dist,float(dist.sum()+np.linalg.norm(a.mean(0)-b.mean(0)))
