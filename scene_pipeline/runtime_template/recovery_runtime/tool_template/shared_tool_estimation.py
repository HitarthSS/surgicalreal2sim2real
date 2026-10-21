"""Common instance-mask construction, stereo verification and pose refinement."""
from copy import deepcopy
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from tool_protocol import LIMITS,validate_initialization,assess_fit
from geometry_repairs import tool_coverage


def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def masks_from_request(request,sizes):
    request=validate_initialization(request,sizes);masks={}
    for item in request['instances']:
        masks[item['instance_id']]={}
        for side,view in item['views'].items():
            w,h=sizes[side];mask=np.zeros((h,w),'uint8')
            cv2.fillPoly(mask,[np.rint(view['polygon']).astype('int32')],1)
            if not 256<=int(mask.sum())<=w*h*.30:raise ValueError('Per-tool mask area outside shared bounds')
            edge=mask[:,0] if item['entry_border']=='left' else mask[:,-1]
            if not edge.any():raise ValueError('Declared instrument must meet its frame boundary')
            masks[item['instance_id']][side]=mask>0
    for side in sizes:
        if len(masks)<2:continue
        a,b=[v[side] for v in masks.values()]
        if (a&b).sum()>min(a.sum(),b.sum())*.15:raise ValueError('Tool instance masks overlap excessively')
    return masks


def heuristic_initialization(images,cached_masks):
    """Independent instance extraction; no jaw-tip observations are invented."""
    instances=[];sources={};sizes={s:[im.shape[1],im.shape[0]] for s,im in images.items()}
    for name,border in [('tool_A','left'),('tool_B','right')]:
        views={}
        try:
            for side,image in images.items():
                cached=cached_masks.get(side);h,w=image.shape[:2]
                hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
                proposals=[('cached_sam3',np.asarray(cached,bool))] if cached is not None else []
                proposals.append(('fixed_dark_low_saturation',(hsv[:,:,2]<85)|(hsv[:,:,1]<100)))
                chosen=None
                for origin,mask in proposals:
                    mask=cv2.morphologyEx(mask.astype('uint8'),cv2.MORPH_OPEN,np.ones((5,5),'uint8'))
                    _,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
                    ids=set(labels[:,0 if border=='left' else -1])-{0}
                    ids=[i for i in ids if 256<=stats[i,cv2.CC_STAT_AREA]<=h*w*.30 and not (labels[:,w-1 if border=='left' else 0]==i).any()]
                    if not ids:continue
                    region=(labels==max(ids,key=lambda i:stats[i,cv2.CC_STAT_AREA])).astype('uint8')
                    contours,_=cv2.findContours(region,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    contour=max(contours,key=cv2.contourArea);epsilon=1.
                    polygon=cv2.approxPolyDP(contour,epsilon,True)
                    while len(polygon)>16:epsilon+=1.;polygon=cv2.approxPolyDP(contour,epsilon,True)
                    points=polygon.reshape(-1,2).tolist()
                    # Preserve actual observed boundary support after contour simplification.
                    ys=np.flatnonzero(region[:,0 if border=='left' else -1])
                    if len(points)<3 or not ys.size:continue
                    if not any(x==(0 if border=='left' else w-1) for x,y in points):continue
                    chosen=dict(polygon=points,jaw_tips_xy=[]);sources[name+'.'+side]=origin;break
                if chosen is None:raise ValueError(side+': no usable '+border+' silhouette')
                views[side]=chosen
            item=dict(instance_id=name,entry_border=border,views=views,reason='Shared fixed silhouette extraction; jaw tips unavailable, keypoint loss disabled.')
            masks_from_request(dict(instances=[item]),sizes)
            masks_from_request(dict(instances=instances+[item]),sizes)
            instances.append(item)
        except ValueError as e:sources[name+'.failure']=str(e)
    if not instances:raise ValueError('Neither visual tool has a usable silhouette: '+str(sources))
    return validate_initialization(dict(instances=instances),sizes),sources


def project(vertices,K,baseline=0):
    p=vertices.copy();p[:,0]-=baseline
    return p[:,:2]/np.maximum(p[:,2,None],1e-6)*[K[0,0],K[1,1]]+K[:2,2]


def rasterize(vertices,faces,K,baseline,shape):
    """Front-surface z-buffer at original resolution, not back-vertex depth."""
    h,w=shape;xy=project(vertices,K,baseline);z=vertices[:,2]
    depth=np.full(shape,np.inf)
    for face in faces:
        if np.any(z[face]<=.005):continue
        triangle=xy[face];lo=np.maximum(np.floor(triangle.min(0)).astype(int),0);hi=np.minimum(np.ceil(triangle.max(0)).astype(int),[w-1,h-1])
        if np.any(hi<lo):continue
        a,b,c=triangle;den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-9:continue
        yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        u=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den
        v=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den;t=1-u-v
        inside=(u>=-1e-6)&(v>=-1e-6)&(t>=-1e-6)
        zz=1/np.maximum(u/z[face[0]]+v/z[face[1]]+t/z[face[2]],1e-9)
        region=depth[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        np.minimum(region,np.where(inside,zz,np.inf),out=region)
    return np.isfinite(depth),depth


def verify(mesh,masks,images,K,baseline,estimated_depth,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);rows={};artifacts={}
    # FFS is left-referenced. Reproject it into the right view with a z-buffer.
    h,w=estimated_depth.shape;yy,xx=np.indices((h,w));good=np.isfinite(estimated_depth)&(estimated_depth>.03)&(estimated_depth<.5)
    right=np.full((h,w),np.inf);yr=yy[good];xr=np.rint(xx[good]-K[0,0]*baseline/estimated_depth[good]).astype(int)
    inside=(xr>=0)&(xr<w);np.minimum.at(right,(yr[inside],xr[inside]),estimated_depth[good][inside])
    for side in ('left','right'):
        rendered,z=rasterize(mesh['vertices'],mesh['faces'],K,baseline if side=='right' else 0,masks[side].shape)
        target=masks[side];depth=estimated_depth if side=='left' else right
        supported=rendered&target&np.isfinite(depth)&(depth>.03)&(depth<.5)
        error=np.abs(z[supported]-depth[supported]);union=(rendered|target).sum()
        rows[side]=dict(mask_iou=float((rendered&target).sum()/max(1,union)),depth_samples=int(supported.sum()),
            median_absolute_depth_difference_m=float(np.median(error)) if len(error) else None,
            depth_source='FFS estimated depth; reprojected to right' if side=='right' else 'FFS estimated depth',
            comparison_is_not_independent_accuracy=True)
        overlay=images[side].copy();overlay[target]=(overlay[target]*.6+np.array([0,170,255])*.4).astype('uint8')
        contour,_=cv2.findContours(rendered.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay,contour,-1,(220,80,220),2)
        path=out/(side+'_tool_overlay.png');cv2.imwrite(str(path),overlay);artifacts[side]=str(path)
    distal_checks={}
    for side,target in masks.items():
        border='left' if target[:,:2].any() else 'right'
        rendered,_=rasterize(mesh['vertices'],mesh['faces'],K,baseline if side=='right' else 0,target.shape)
        distal_checks[side]=tool_coverage(rendered,target,border)
    pose=mesh['frame4_to_camera'];q=mesh['distal_joints_rad']
    valid=bool(np.isfinite(pose).all() and np.allclose(pose[3],[0,0,0,1]) and
        np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-4) and np.linalg.det(pose[:3,:3])>.99 and
        .03<=pose[2,3]<=.5 and q.shape==(4,) and np.isfinite(q).all() and
        np.all(q>=[-1.5707,-1.3963,0,0]) and np.all(q<=[1.5707,1.3963,1.5707,1.5707]))
    report=dict(views=rows,**assess_fit(rows,valid),limits=LIMITS,feedback_images=artifacts,
        mask_source_is_proposed_not_ground_truth=True,depth_is_estimated=True)
    report['distal_checks']=distal_checks
    for side,row in distal_checks.items():
        if not row['reconstruction_visible'] or row['distal_coverage']<.8 or row['tip_region_coverage']<.8:report['failures'].append(side+'_distal_shape_mismatch')
    report['accepted_numerical_hypothesis']=not report['failures']
    save(out/'report.json',report);return report


def refine_stereo_pose(mesh,masks,K,baseline,depth,out):
    """Bounded camera-frame SE(3) refinement, preserving native articulation."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    center=mesh['frame4_to_camera'][:3,3];vertices=mesh['vertices'];subsets={};target_boundaries={};distances={}
    for side in ('left','right'):
        xy=project(vertices,K,baseline if side=='right' else 0);h,w=masks[side].shape
        good=(vertices[:,2]>.03)&(xy[:,0]>=-20)&(xy[:,0]<w+20)&(xy[:,1]>=-20)&(xy[:,1]<h+20)
        ids=np.where(good)[0][::max(1,int(good.sum()/500))][:500]
        if len(ids)<20:raise ValueError('Insufficient projected native vertices for bounded refinement')
        subsets[side]=ids;distances[side]=distance_transform_edt(~masks[side])
        boundary=masks[side]&~cv2.erode(masks[side].astype('uint8'),np.ones((3,3),'uint8')).astype(bool)
        y,x=np.nonzero(boundary);p=np.c_[x,y];target_boundaries[side]=p[::max(1,len(p)//120)][:120]
    def moved(parameters):return (vertices-center)@Rotation.from_rotvec(parameters[:3]).as_matrix().T+center+parameters[3:]
    def residual(parameters):
        current=moved(parameters);values=[]
        for side in ('left','right'):
            xy=project(current[subsets[side]],K,baseline if side=='right' else 0);h,w=masks[side].shape
            clipped=np.clip(np.rint(xy).astype(int),[0,0],[w-1,h-1]);d=distances[side][clipped[:,1],clipped[:,0]]
            values.extend((d+np.linalg.norm(xy-clipped,axis=1))/10)
            values.extend(cKDTree(xy).query(target_boundaries[side])[0]/10)
        # Unchanged estimated depth acts as a soft local constraint, not truth.
        ids=subsets['left'];xy=project(current[ids],K);h,w=depth.shape;pix=np.clip(np.rint(xy).astype(int),[0,0],[w-1,h-1])
        target=depth[pix[:,1],pix[:,0]];ok=np.isfinite(target)&(target>.03)&(target<.5)&masks['left'][pix[:,1],pix[:,0]]
        values.extend(np.where(ok,(current[ids,2]-np.where(ok,target,0))/.005,0))
        values.extend(parameters[:3]/.262*.1);values.extend(parameters[3:]/.025*.1)
        return np.asarray(values)
    bound=np.array([LIMITS['max_rotation_refinement_rad']]*3+[LIMITS['max_translation_refinement_m']]*3)
    result=least_squares(residual,np.zeros(6),bounds=(-bound,bound),loss='soft_l1',max_nfev=80,diff_step=None,
                         x_scale=np.r_[np.full(3,.1),np.full(3,.01)])
    refined={k:v.copy() for k,v in mesh.items()};refined['vertices']=moved(result.x)
    pose=mesh['frame4_to_camera'].copy();pose[:3,:3]=Rotation.from_rotvec(result.x[:3]).as_matrix()@pose[:3,:3];pose[:3,3]+=result.x[3:]
    refined['frame4_to_camera']=pose
    np.savez_compressed(out/'tool_camera_m.npz',**refined)
    save(out/'operation.json',dict(operation='refine_stereo_pose',delta_rotvec_rad=result.x[:3].tolist(),delta_translation_m=result.x[3:].tolist(),
        native_distal_joints_unchanged=True,optimizer_success=bool(result.success),evaluations=result.nfev,
        residual_before=float(np.mean(residual(np.zeros(6))**2)),residual_after=float(np.mean(residual(result.x)**2)),limits=LIMITS,
        independent_accuracy_measured=False))
    return refined
