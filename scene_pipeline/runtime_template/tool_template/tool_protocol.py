"""Shared bounded tool initialization and stereo-fit acceptance contracts."""
from copy import deepcopy
import hashlib
import json
import math

LIMITS=dict(max_instances=2,max_model_attempts=6,max_native_fits=2,max_stereo_refinements=2,
            min_iou_per_view=.35,max_median_depth_error_m=.010,min_depth_samples=100,
            max_translation_refinement_m=.05,max_rotation_refinement_rad=1.5707963267948966)


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_initialization(value,sizes):
    if not isinstance(value,dict) or set(value)!={'instances'} or not isinstance(value['instances'],list) or not 1<=len(value['instances'])<=2:
        raise ValueError('Require one or two independently identified visual instances')
    result=deepcopy(value);seen=set()
    for item in result['instances']:
        if set(item)!={'instance_id','entry_border','views','reason'}:raise ValueError('Unexpected tool initialization fields')
        if item['instance_id'] not in ('tool_A','tool_B') or item['instance_id'] in seen:raise ValueError('Unique tool_A/tool_B IDs required')
        seen.add(item['instance_id'])
        if item['entry_border'] not in ('left','right'):raise ValueError('Declare observed boundary entry, not PSM identity')
        if not isinstance(item['reason'],str) or not 1<=len(item['reason'])<=400:raise ValueError('Reason <=400 characters')
        if not isinstance(item['views'],dict) or set(item['views'])!={'left','right'}:raise ValueError('Both full stereo views required')
        for side,row in item['views'].items():
            if set(row)!={'polygon','jaw_tips_xy'}:raise ValueError('Require polygon and jaw_tips_xy; use [] when tips are not observed')
            polygon=row['polygon'];tips=row['jaw_tips_xy'];w,h=sizes[side]
            if not isinstance(polygon,list) or not 3<=len(polygon)<=16:raise ValueError('Polygon requires 3–16 vertices')
            if not isinstance(tips,list) or len(tips) not in (0,2):raise ValueError('Provide two supported tips or an empty list for silhouette-only fitting')
            for point in polygon+tips:
                if not isinstance(point,list) or len(point)!=2 or any(type(x) not in (int,float) or not math.isfinite(x) for x in point):raise ValueError('Finite [x,y] coordinates required')
                if not 0<=point[0]<=w-1 or not 0<=point[1]<=h-1:raise ValueError('Use original pixel centers within width-1/height-1')
            if tips and math.dist(tips[0],tips[1])<1:row['jaw_tips_xy']=[]
    if len({x['entry_border'] for x in result['instances']})!=len(result['instances']):raise ValueError('Distinct visual entries required')
    return result


def assess_fit(views,pose_valid=True):
    failures=[]
    if set(views)!={'left','right'}:failures.append('both_views_required')
    if not pose_valid:failures.append('invalid_pose_or_joint_bounds')
    for side,row in views.items():
        if not math.isfinite(row['mask_iou']) or row['mask_iou']<LIMITS['min_iou_per_view']:failures.append(side+'_silhouette_mismatch')
        if row['depth_samples']<LIMITS['min_depth_samples']:failures.append(side+'_insufficient_stereo_depth_support')
        error=row['median_absolute_depth_difference_m']
        if error is None or not math.isfinite(error) or error>LIMITS['max_median_depth_error_m']:failures.append(side+'_stereo_depth_mismatch')
    return dict(accepted_numerical_hypothesis=not failures,failures=failures,
                independent_accuracy_measured=False,robot_arm_identity_established=False,
                physics_validated=False,training_eligible=False)
