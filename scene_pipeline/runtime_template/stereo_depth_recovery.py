"""Mask-supported mutual stereo patch correspondence; no depth/network rerun."""
import numpy as np,cv2
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

def recover(left,right,left_mask,right_mask,original_depth,K,baseline,min_z=.03,max_z=.5,patch_radius=4,min_ncc=.70,min_margin=.05):
 if left.shape!=right.shape or left.shape[:2]!=left_mask.shape or left_mask.shape!=right_mask.shape or original_depth.shape!=left_mask.shape:raise ValueError('Stereo shape mismatch')
 if not 0<baseline or not 0<min_z<max_z:raise ValueError('Invalid stereo scale/range')
 h,w=left_mask.shape;r=patch_radius
 def features(image,mask):
  gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) if image.ndim==3 else image
  gray=gray.astype(np.float32);sk=skeletonize(mask>0);rc=np.argwhere(sk)
  rc=rc[(rc[:,0]>=r)&(rc[:,0]<h-r)&(rc[:,1]>=r)&(rc[:,1]<w-r)]
  patches=np.asarray([gray[y-r:y+r+1,x-r:x+r+1].ravel() for y,x in rc],np.float32)
  if not len(rc):return rc,np.empty((0,(2*r+1)**2),np.float32),np.empty(0,bool)
  centered=patches-patches.mean(1,keepdims=True);norm=np.linalg.norm(centered,axis=1)
  valid=patches.std(1)>=2.;centered/=np.maximum(norm[:,None],1e-9)
  return rc,centered,valid
 lr,lf,lv=features(left,left_mask);rr,rf,rv=features(right,right_mask)
 edges=[];byrow={int(y):np.flatnonzero(rr[:,0]==y) for y in np.unique(rr[:,0])}
 dmin=K[0,0]*baseline/max_z;dmax=K[0,0]*baseline/min_z
 for i,(y,x) in enumerate(lr):
  if not lv[i]:continue
  ids=np.concatenate([byrow.get(int(yy),np.empty(0,int)) for yy in range(y-1,y+2)])
  if not len(ids):continue
  disparity=x-rr[ids,1];ids=ids[(disparity>=dmin)&(disparity<=dmax)&rv[ids]]
  if not len(ids):continue
  scores=rf[ids]@lf[i]-.04*np.abs(rr[ids,0]-y)
  edges.extend((i,int(j),float(score)) for j,score in zip(ids,scores))
 left_edges={};right_edges={}
 for i,j,score in edges:left_edges.setdefault(i,[]).append((score,j));right_edges.setdefault(j,[]).append((score,i))
 accepted=[];rejections={'low_ncc':0,'ambiguous':0,'non_mutual':0}
 for i,options in left_edges.items():
  options.sort(reverse=True);score,j=options[0];reverse=sorted(right_edges[j],reverse=True)
  if score<min_ncc:rejections['low_ncc']+=1;continue
  # Neighboring pixels on one strand are not distinct topology hypotheses.
  other=[s for s,k in options[1:] if np.linalg.norm(rr[k]-rr[j])>2.]
  reverse_other=[s for s,k in reverse[1:] if np.linalg.norm(lr[k]-lr[i])>2.]
  if (other and score-max(other)<min_margin) or (reverse_other and score-max(reverse_other)<min_margin):rejections['ambiguous']+=1;continue
  if reverse[0][1]!=i:rejections['non_mutual']+=1;continue
  disparity=float(lr[i,1]-rr[j,1]);z=float(K[0,0]*baseline/disparity)
  if min_z<=z<=max_z:accepted.append((i,j,z,score))
 depth=original_depth.copy();depth[left_mask>0]=np.nan
 anchor=np.zeros((h,w),bool);values=np.full((h,w),np.nan,np.float32)
 for i,j,z,score in accepted:y,x=lr[i];anchor[y,x]=True;values[y,x]=z
 if anchor.any():
  dist,indices=distance_transform_edt(~anchor,return_indices=True);support=(left_mask>0)&(dist<=3.)
  depth[support]=values[indices[0][support],indices[1][support]]
 else:support=np.zeros((h,w),bool)
 report=dict(schema='mutual_stereo_depth_repair_v1',status='candidate_only',left_skeleton_samples=len(lr),right_skeleton_samples=len(rr),
   accepted_mutual_correspondences=len(accepted),supported_thread_pixels=int(support.sum()),thread_pixels=int((left_mask>0).sum()),rejections=rejections,
   normalized_correlation_threshold=min_ncc,distinct_match_margin=min_margin,epipolar_tolerance_px=1,propagation_radius_px=3,
   depth_range_m=[min_z,max_z],background_depth_preserved=True,unsupported_thread_depth_marked_missing=True,
   source_depth_modified=False,depth_is_estimated=True,independent_accuracy_measured=False)
 pairs=np.asarray([[*lr[i][::-1],*rr[j][::-1],z,score] for i,j,z,score in accepted],np.float64).reshape(-1,6)
 return depth,pairs,report
