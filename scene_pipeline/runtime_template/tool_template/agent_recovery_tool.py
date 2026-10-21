"""Explicit recovery operation for an agent or heuristic controller.

The host supplies arrays/calibration/native model and output paths. The model can
select a search strategy, never write executable code, transform calibration,
scale geometry or declare the result visually accepted.
"""
import numpy as np
from reinitialize import SEARCH, pose_seed, stereo_tip_points, delta_for
from tool_repair import refine, tip_vertices, verify, promotion, save

TOOL_CARD = dict(action_id='reinitialize_stereo_tool',
    purpose='Recover from a local tool fit that reaches its search bound or retains a stereo shaft mismatch.',
    parameters=dict(instance_id=['tool_A','tool_B'],strategy=['multi_start','stereo_geometry','warm_start']),
    search=SEARCH,maximum_starts=3,
    acceptance='Unchanged two-view numerical gates, original/revised mask nonregression and all jaw-tip residuals <=20px; visual review remains separate.',
    requires=['image.left','image.right'],changes_dimensions=False,changes_calibration=False)


def validate(request,version):
    if set(request)!={'scene_version','action_id','parameters','evidence_ids','reason'}:
        raise ValueError('Exact recovery request fields required')
    if request['scene_version']!=version:raise ValueError('Stale scene version')
    if request['action_id']!='reinitialize_stereo_tool':raise ValueError('Unknown recovery action')
    p=request['parameters']
    if set(p)!={'instance_id','strategy'} or p['instance_id'] not in ['tool_A','tool_B'] or p['strategy'] not in TOOL_CARD['parameters']['strategy']:
        raise ValueError('Only declared tool instance and automatic seed strategy may be selected')
    if not {'image.left','image.right'}<=set(request['evidence_ids'])<=set(['image.left','image.right','fit.left','fit.right']):
        raise ValueError('Registered full stereo evidence required')
    if not isinstance(request['reason'],str) or not 1<=len(request['reason'])<=500:
        raise ValueError('Concise bounded reason required')
    return p


def execute(request,context):
    """Shared host callback; context is trusted application data, not model JSON.

    context: scene_version, remaining_recoveries, native, original, warm,
    masks, reference_masks, views, K, baseline_m, images, depth, out.
    """
    from pathlib import Path
    p=validate(request,context['scene_version'])
    if context['remaining_recoveries']<1:raise ValueError('Recovery operation budget exhausted')
    context['remaining_recoveries']-=1
    native,original=context['native'],context['original']
    native.validate_source(original)
    out=Path(context['out']);out.mkdir(parents=True,exist_ok=False)
    save(out/'request.json',request)
    warm=context['warm']
    seeds=[];failures=[]
    if p['strategy']!='stereo_geometry':
        seeds.append(('warm_start',delta_for(original,warm['frame4_to_camera'],warm['distal_joints_rad'])))
    if p['strategy']!='warm_start':
        try:
            pose,info=pose_seed(native,original,context['masks'],context['views'],context['K'],context['baseline_m'],border=context['entry_border'])
            seeds.append(('stereo_shaft_axis',delta_for(original,pose,original['distal_joints_rad'])))
            points,info=stereo_tip_points(context['views'],context['K'],context['baseline_m'])
            ids,_=tip_vertices(native,original,context['views'],context['K'],context['baseline_m'])
            body=native.body(original['distal_joints_rad'])
            pose=pose.copy();pose[:3,3]=points.mean(axis=0)-pose[:3,:3]@body[ids].mean(axis=0)
            seeds.append(('stereo_shaft_and_tips',delta_for(original,pose,original['distal_joints_rad'])))
        except ValueError as error:failures.append(str(error))
    before={}
    for kind,mask in [('reference',context['reference_masks']),('current',context['masks'])]:
        before[kind]=verify(original,mask,context['images'],context['K'],context['baseline_m'],context['depth'],out/('before_'+kind))
    records=[]
    for index,(name,initial) in enumerate(seeds[:3]):
        folder=out/f'start_{index+1:02d}'
        candidate=refine(native,original,context['masks'],context['views'],context['K'],context['baseline_m'],folder,
            reference_masks=context['reference_masks'],search=SEARCH,initial=initial)
        reports={}
        for kind,mask in [('reference',context['reference_masks']),('current',context['masks'])]:
            reports[kind]=verify(candidate,mask,context['images'],context['K'],context['baseline_m'],context['depth'],folder/kind)
        import json
        operation=json.loads((folder/'operation.json').read_text())
        tips=operation['jaw_tip_residuals_px']
        passed=bool(all(promotion(before[k],reports[k]) for k in reports) and
            all(np.isfinite(v).all() and max(v)<=20 for v in tips.values()))
        records.append(dict(seed=name,mesh=str(folder/'tool_camera_m.npz'),numerical_pass=passed,reports=reports,operation=operation))
    eligible=[r for r in records if r['numerical_pass']]
    selected=max(eligible,key=lambda r:sum(v['mask_iou'] for v in r['reports']['reference']['views'].values())) if eligible else None
    result=dict(selected=selected,attempts=records,seed_failures=failures,
        requires_visual_review=True,independent_accuracy_measured=False,physics_validated=False)
    save(out/'result.json',result)
    return result
