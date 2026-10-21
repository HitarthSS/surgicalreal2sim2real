"""Broader deterministic stereo pose search on saved automatic observations.

No API calls. No change to dimensions, physical joint limits, or final checks.
Both controllers receive the same seed generators, search bounds and budget.
"""
import json
from pathlib import Path
import time
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
from tool_repair import (BASE, NativeTool, refine, masks_from_request, verify,
                         promotion, save, sha, silhouette, tip_vertices)

ROOT=Path(__file__).resolve().parent
PREVIOUS=Path('/home/hitarth/integrated_surgical_scene_outputs/ismr_tool_repair_20261016_v2')
SEARCH=dict(rotation_component_rad=float(np.pi/2),translation_m=.05,
            joint_delta_rad=float(np.pi/4),max_evaluations=2200,max_seconds=240)


def fit_shaft_line(mask,border):
    """Fit the proximal shaft midline, using columns near its entering border."""
    y,x=np.nonzero(mask)
    if len(x)<256:raise ValueError('Insufficient tool mask support')
    width=int(x.max()-x.min()+1)
    span=max(25,int(width*.35))
    columns=range(max(0,mask.shape[1]-span),mask.shape[1]) if border=='right' else range(min(span,mask.shape[1]))
    centers=[]
    for column in columns:
        rows=np.where(mask[:,column])[0]
        if len(rows)>=4:centers.append([column,(rows.min()+rows.max())*.5])
    if len(centers)<15:raise ValueError('Proximal shaft line unsupported')
    vx,vy,cx,cy=np.asarray(cv2.fitLine(np.asarray(centers,dtype=np.float32),cv2.DIST_L1,0,.01,.01)).reshape(4)
    line=np.array([-vy,vx,vy*cx-vx*cy],float)
    return line/np.linalg.norm(line[:2]),dict(columns=len(centers),proximal_span_px=span)


def stereo_axis(left,right,K,baseline):
    """Intersect the two back-projected image-line planes in camera coordinates."""
    normals=np.stack([K.T@left,K.T@right])
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    direction=np.cross(*normals)
    conditioning=float(np.linalg.norm(direction))
    if conditioning<1e-4:raise ValueError('Stereo shaft planes nearly parallel')
    direction/=conditioning
    # right camera center is [baseline,0,0] in the rectified left frame.
    rhs=np.array([0.,normals[1,0]*baseline])
    point=np.linalg.lstsq(normals,rhs,rcond=None)[0]
    return direction,point,conditioning


def align_axis(source,target):
    source=source/np.linalg.norm(source);target=target/np.linalg.norm(target)
    v=np.cross(source,target);s=np.linalg.norm(v);c=np.clip(source@target,-1,1)
    if s<1e-10:
        if c>0:return np.eye(3)
        axis=np.eye(3)[np.argmin(np.abs(source))];axis=np.cross(source,axis);axis/=np.linalg.norm(axis)
        return Rotation.from_rotvec(axis*np.pi).as_matrix()
    return Rotation.from_rotvec(v/s*np.arctan2(s,c)).as_matrix()


def stereo_tip_points(views,K,baseline):
    left=np.asarray(views['left']['jaw_tips_xy'],float)
    right=np.asarray(views['right']['jaw_tips_xy'],float)
    if left.shape!=(2,2) or right.shape!=(2,2):raise ValueError('Jaw tips unobserved; skip tip-based seed and retain silhouette/shaft seeds')
    cost=np.abs(left[:,None,1]-right[None,:,1])
    _,order=linear_sum_assignment(cost);right=right[order]
    disparity=left[:,0]-right[:,0]
    if (np.abs(left[:,1]-right[:,1])>8).any() or (disparity<=0).any():
        raise ValueError('Jaw-tip stereo correspondence unsupported')
    z=K[0,0]*baseline/disparity
    if (z<.03).any() or (z>.5).any():raise ValueError('Jaw-tip depth outside camera workspace')
    points=np.c_[(left[:,0]-K[0,2])*z/K[0,0],((left[:,1]+right[:,1])*.5-K[1,2])*z/K[1,1],z]
    return points,dict(epipolar_residual_px=np.abs(left[:,1]-right[:,1]).tolist(),
                       disparity_px=disparity.tolist(),right_pair_order=order.tolist())


def pose_seed(native,original,masks,views,K,baseline,border='right'):
    line_left,info_l=fit_shaft_line(masks['left'],border)
    line_right,info_r=fit_shaft_line(masks['right'],border)
    axis,point,conditioning=stereo_axis(line_left,line_right,K,baseline)
    body=native.body(original['distal_joints_rad'])
    shaft=body[native.slices[0]];center=shaft.mean(axis=0)
    _,_,v=np.linalg.svd(shaft-center,full_matrices=False);native_axis=v[0]
    current_axis=original['frame4_to_camera'][:3,:3]@native_axis
    if current_axis@axis<0:axis=-axis
    r=align_axis(current_axis,axis)@original['frame4_to_camera'][:3,:3]
    # Position native shaft line on the observed 3D line near the old wrist.
    near_native=center-native_axis*(center@native_axis)
    origin=original['frame4_to_camera'][:3,3]
    closest=point+axis*((origin-point)@axis)
    t=closest-r@near_native
    pose=np.eye(4);pose[:3,:3]=r;pose[:3,3]=t
    metadata=dict(left=info_l,right=info_r,plane_conditioning=conditioning,
        shaft_direction_camera=axis.tolist(),source='automatic stereo mask midlines')
    return pose,metadata


def delta_for(original,pose,q):
    return np.r_[Rotation.from_matrix(pose[:3,:3]@np.linalg.inv(original['frame4_to_camera'][:3,:3])).as_rotvec(),
                 pose[:3,3]-original['frame4_to_camera'][:3,3],q-original['distal_joints_rad']]


def load(path):
    with np.load(path,allow_pickle=False) as data:return {k:data[k].copy() for k in data.files}


def main():
    assert not (ROOT/'results.json').exists(),'Completed diagnostic must not be rerun'
    config=json.loads((BASE/'pilot_config.json').read_text())
    case=json.loads(Path(config['case']).read_text())
    images={s:cv2.imread(case['inputs'][s]) for s in ['left','right']}
    sizes={s:[im.shape[1],im.shape[0]] for s,im in images.items()}
    K=np.loadtxt(case['inputs']['K'],delimiter=',');depth=np.load(case['inputs']['depth_npy'])
    protected={p:sha(p) for p in case['inputs'].values() if Path(p).is_file()}
    for method in ['agent','heuristic']:
        for name in ['tool_A','tool_B']:
            p=str(BASE/method/name/'mesh/tool_camera_m.npz');protected[p]=sha(p)
    protected.update({str(p):sha(p) for p in PREVIOUS.rglob('*.json')})
    save(ROOT/'protected_inputs.json',protected)
    save(ROOT/'source_manifest.json',{p.name:sha(p) for p in ROOT.glob('*.py')})
    native=NativeTool(config['repo']);results={}
    for method in ['agent','heuristic']:
        out=ROOT/method;out.mkdir(exist_ok=False)
        original=load(BASE/method/'tool_B/mesh/tool_camera_m.npz')
        native.validate_source(original)
        old_obs=json.loads((BASE/method/'initialization.json').read_text())
        observation_path=PREVIOUS/'agent_feedback/repair_02/observations.json' if method=='agent' else BASE/method/'initialization.json'
        observations=json.loads(observation_path.read_text());save(out/'observations.json',observations)
        masks=masks_from_request(observations,sizes)['tool_B'];frozen=masks_from_request(old_obs,sizes)['tool_B']
        views=next(x['views'] for x in observations['instances'] if x['instance_id']=='tool_B')
        reference_before=verify(original,frozen,images,K,case['baseline_m'],depth,out/'before_frozen')
        revised_before=verify(original,masks,images,K,case['baseline_m'],depth,out/'before_revised')
        previous_path=PREVIOUS/'agent_feedback/repair_02/optimizer/tool_camera_m.npz' if method=='agent' else PREVIOUS/'diagnostic/heuristic/optimizer/tool_camera_m.npz'
        previous=load(previous_path)
        seeds=[('previous_local_fit',delta_for(original,previous['frame4_to_camera'],previous['distal_joints_rad']))]
        seed_info=[]
        try:
            pose,info=pose_seed(native,original,masks,views,K,case['baseline_m']);seed_info.append(info)
            seeds.append(('stereo_shaft_axis',delta_for(original,pose,original['distal_joints_rad'])))
            points,info=stereo_tip_points(views,K,case['baseline_m']);seed_info.append(info)
            ids,_=tip_vertices(native,original,views,K,case['baseline_m'])
            body=native.body(original['distal_joints_rad'])
            pose_tips=pose.copy();pose_tips[:3,3]=points.mean(axis=0)-pose[:3,:3]@body[ids].mean(axis=0)
            seeds.append(('stereo_shaft_and_tips',delta_for(original,pose_tips,original['distal_joints_rad'])))
        except ValueError as error:
            seed_info.append(dict(seed_failure=str(error)))
        # Equal count limit, no fabricated stereo correspondences on failure.
        if len(seeds)<3:seeds.append(('original_native_pose',np.zeros(10)))
        save(out/'seed_provenance.json',dict(seeds=[dict(name=n,delta=d.tolist()) for n,d in seeds],details=seed_info,search=SEARCH))
        attempts=[]
        for index,(name,initial) in enumerate(seeds[:3]):
            folder=out/f'attempt_{index+1:02d}'
            save(ROOT/'status.json',dict(state='reinitializing',controller=method,seed=name,index=index+1,completed_controllers=len(results)))
            candidate=refine(native,original,masks,views,K,case['baseline_m'],folder,reference_masks=frozen,search=SEARCH,initial=initial)
            after_frozen=verify(candidate,frozen,images,K,case['baseline_m'],depth,folder/'frozen_verification')
            after_revised=verify(candidate,masks,images,K,case['baseline_m'],depth,folder/'revised_verification')
            operation=json.loads((folder/'operation.json').read_text())
            tip_gate=all(max(values)<=20 and np.isfinite(values).all() for values in operation['jaw_tip_residuals_px'].values())
            passed=bool(promotion(reference_before,after_frozen) and promotion(revised_before,after_revised) and tip_gate)
            record=dict(seed=name,candidate=str(folder/'tool_camera_m.npz'),numerical_pass=passed,jaw_tip_gate=tip_gate,
                frozen_report=after_frozen,revised_report=after_revised,operation=operation)
            attempts.append(record);save(out/'attempts.json',attempts)
        # Preserve every attempt, choose by explicit equal-view frozen consistency.
        eligible=[x for x in attempts if x['numerical_pass']]
        selected=max(eligible,key=lambda x:sum(v['mask_iou'] for v in x['frozen_report']['views'].values())) if eligible else None
        results[method]=dict(attempts=attempts,selected=selected,before_frozen=reference_before,
            source_observations=str(observation_path),new_model_calls=0,actual_agent_selected_reinitialization=False,
            independent_accuracy_measured=False,physics_validated=False)
        save(ROOT/'results.json',results)
    unchanged=all(sha(p)==h for p,h in protected.items())
    save(ROOT/'verification.json',dict(protected_inputs_unchanged=unchanged,
        sources_unchanged=all(sha(ROOT/n)==h for n,h in json.loads((ROOT/'source_manifest.json').read_text()).items()),
        actual_model_calls=0,physical_dimensions_unchanged=True,physical_joint_limits_unchanged=True,
        final_acceptance_gates_unchanged=True,independent_accuracy_measured=False,physics_validated=False))
    assert unchanged
    save(ROOT/'status.json',dict(state='complete',controllers=2,new_model_calls=0))


if __name__=='__main__':main()
