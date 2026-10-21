"""Fit whole fixed arcs jointly to stereo masks; never resize or deform a needle."""
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

def triangulated_seed(masks,K,baseline):
    """Use two-view needle correspondences to initialize its 3D plane, not just depth.

    This free circle is an initialization only. Every delivered candidate still
    has a fixed catalog radius/span and the original rigid/stereo verification.
    """
    from skimage.morphology import skeletonize
    sk={s:skeletonize(masks[s]) for s in ('left','right')};pairs=[]
    for y in range(sk['left'].shape[0]):
        left=np.flatnonzero(sk['left'][y]);right=np.flatnonzero(sk['right'][y])
        # An arc has at most two intersections with a row. Preserve arm order;
        # a merged/branched mask is not an initialization correspondence.
        if not 1<=len(left)==len(right)<=2:continue
        for a,b in zip(left,right):
            disparity=float(a-b)
            if disparity<=1:continue
            z=K[0,0]*baseline/disparity
            if .03<=z<=.5:pairs.append([(a-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z])
    if len(pairs)<12:return None
    p=np.asarray(pairs);center=np.median(p,axis=0)
    _,_,axes=np.linalg.svd(p-center,full_matrices=False)
    u,v=axes[:2];normal=np.cross(u,v);rotation=np.column_stack([u,v,normal])
    xy=(p-center)@rotation[:,:2]
    matrix=np.c_[2*xy,np.ones(len(xy))];a,b,c=np.linalg.lstsq(matrix,np.sum(xy*xy,axis=1),rcond=None)[0]
    radius=np.sqrt(max(c+a*a+b*b,1e-10))
    fit=least_squares(lambda q:np.linalg.norm(xy-q[:2],axis=1)-q[2],[a,b,radius],
        bounds=([-1,-1,.0001],[1,1,.2]),loss='soft_l1',f_scale=.0005,max_nfev=60)
    center=center+rotation[:,:2]@fit.x[:2]
    return np.r_[center,Rotation.from_matrix(rotation).as_rotvec()],float(fit.x[2])

def stereo_seed(masks,K,baseline):
    """Initialize depth from paired needle rows rather than background depth."""
    from geometry_repairs import _circle2d
    disparities=[]
    for y in range(masks['left'].shape[0]):
        a=np.flatnonzero(masks['left'][y]);b=np.flatnonzero(masks['right'][y])
        if len(a) and len(b):disparities.append(float(np.median(a)-np.median(b)))
    disparities=np.asarray(disparities);disparities=disparities[np.isfinite(disparities)&(disparities>1.)]
    if len(disparities)<10:return None
    z=float(K[0,0]*baseline/np.median(disparities))
    if not .03<=z<=.5:return None
    xy=np.argwhere(masks['left'])[:,::-1];center,_=_circle2d(xy)
    return np.r_[(center[0]-K[0,2])*z/K[0,0],(center[1]-K[1,2])*z/K[1,1],z,[0.,0.,0.]]

def refine_arc_rows(case,original,parts,out,existing,max_evaluations=90):
    from fixed_needle_fitter import (inputs,mask_points,_circle_seed,_angle_interval,distance_fields,
        sample_distances,sample,from_pose,evaluate,sha,CATALOG,audit_rigid)
    from geometry_repairs import project
    from fixed_needle_fitter import thread_endpoints
    images,masks,thread,occ,K,depth,b=inputs(case);observed={s:mask_points(masks[s],192) for s in ('left','right')}
    if min(map(len,observed.values()))<6:return []
    fields=distance_fields(masks,thread);ends=thread_endpoints(parts)
    seeds=[]
    if original is not None:seeds.append(('legacy_rigid_pose_seed',_circle_seed(original,masks,K,depth)))
    stereo=stereo_seed(masks,K,b)
    if stereo is not None:seeds.append(('needle_stereo_disparity_seed',stereo))
    triangulated=triangulated_seed(masks,K,b)
    if triangulated is not None:seeds.append(('needle_triangulated_plane_seed',triangulated[0]))
    if not seeds:return []
    # Keep only the most promising two radii from the original catalog, plus
    # the closest fixed size to a stereo-derived projected radius. No new size.
    ranks={r:max((x['report']['score'] for x in existing if x['radius_mm']==r),default=-999) for r in CATALOG['radius_mm']}
    # When the native fit is missing, its two highest-scoring failed radii are
    # not a reliable size shortlist. Search the unchanged complete catalog.
    radii=list(CATALOG['radius_mm']) if original is None else sorted(ranks,key=ranks.get,reverse=True)[:2]
    if triangulated is not None:radii.append(min(CATALOG['radius_mm'],key=lambda x:abs(x-triangulated[1]*1000)))
    if original is not None:
        radius=float(original.get('fit_radius_m',np.nan))*1000
        if np.isfinite(radius):radii.append(min(CATALOG['radius_mm'],key=lambda x:abs(x-radius)))
    radii=sorted(set(radii));rows=[];out=Path(out)
    for radius in radii:
        for span in CATALOG['span_degrees']:
            unit_theta=np.linspace(0,np.deg2rad(span),128)
            for label,seed in seeds:
                circle_theta=np.linspace(0,2*np.pi,360,endpoint=False)
                rotation=Rotation.from_rotvec(seed[3:]).as_matrix()
                unit=np.c_[np.cos(circle_theta),np.sin(circle_theta),np.zeros(len(circle_theta))]
                full=seed[:3]+radius/1000.*(unit@rotation.T);angles=[]
                for side in ('left','right'):
                    xy=project(full,K,b if side=='right' else 0)
                    angles.extend(circle_theta[cKDTree(xy).query(observed[side])[1]])
                start,visible=_angle_interval(np.asarray(angles));phase=start-(np.deg2rad(span)-visible)/2
                q0=np.r_[seed,phase]
                def points(q):
                    theta=unit_theta+q[6];unit=np.c_[np.cos(theta),np.sin(theta),np.zeros(len(theta))]
                    return q[:3]+radius/1000.*(unit@Rotation.from_rotvec(q[3:6]).as_matrix().T)
                def residual(q):
                    p=points(q);values=[]
                    for side in ('left','right'):
                        xy=project(p,K,b if side=='right' else 0)
                        values.extend(cKDTree(xy).query(observed[side])[0]/3.)
                        distance=sample_distances(fields['needle'][side],xy)
                        hidden,in_frame=sample(occ[side],xy)
                        values.extend(np.minimum(np.where(hidden,0.,distance)[::2],100.)/4.)
                    values.extend(np.maximum(.03-p[:,2],0.)/.001)
                    return np.asarray(values)
                lower=np.r_[-.5,-.5,.03,[-2*np.pi]*3,-4*np.pi];upper=np.r_[.5,.5,.5,[2*np.pi]*3,4*np.pi]
                q0=np.clip(q0,lower+1e-8,upper-1e-8)
                fit=least_squares(residual,q0,bounds=(lower,upper),loss='soft_l1',f_scale=2.,max_nfev=max_evaluations,
                    diff_step=1e-4,x_scale=np.r_[[.02]*3,[1.]*4])
                # Preserve both initial and optimized poses: a lower objective
                # may still explain less observed evidence under the verifier.
                for phase_name,q in [('seed',q0),('optimized',fit.x)]:
                    artifact=from_pose(radius/1000.,span,Rotation.from_rotvec(q[3:6]).as_matrix(),q[:3],q[6])
                    report,inferred=evaluate(artifact['points_m'],masks,occ,thread,K,b,ends,fields=fields)
                    contract=audit_rigid(artifact,CATALOG,require_pose=True);assert contract['usable']
                    report['rigid_contract']=contract
                    artifact.update(inferred=inferred,visible_stereo_support=~inferred,reliability=(~inferred).astype(float),
                        tube_radius_m=np.asarray(CATALOG['tube_radius_m']),size_prior_used=np.asarray(True))
                    cid=f'joint_r{radius:g}_s{span:g}_{label}_{phase_name}'
                    path=out/(cid+'.npz');np.savez_compressed(path,**artifact)
                    rows.append(dict(candidate_id=cid,needle_path=str(path.resolve()),needle_sha256=sha(path),radius_mm=radius,
                        span_degrees=span,report=report,whole_arc_joint_stereo_refinement=True,initialization=label,
                        optimizer_evaluations=int(fit.nfev),optimizer_success=bool(fit.success)))
    return rows
