"""Conservative, measured-consistency promotion. Does not claim independent accuracy."""
import math
from copy import deepcopy

def tool_promotes(old,new):
    """Fixed original masks/calibration; improve IoU without degrading stereo depth."""
    if not new or not new.get('accepted_numerical_hypothesis'):return False,'candidate_failed_verification'
    if not old:return True,'first_verified_candidate'
    if set(new.get('views',{}))!={'left','right'}:return False,'missing_stereo_verification'
    improved=False
    for side in ['left','right']:
        a=old.get('views',{}).get(side,{});b=new['views'][side]
        x,y=a.get('mask_iou'),b.get('mask_iou')
        if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (x,y)):return False,'invalid_iou'
        if y<x-1e-6:return False,side+'_overlap_regression'
        improved|=y>x+1e-4
        x,y=a.get('median_absolute_depth_difference_m'),b.get('median_absolute_depth_difference_m')
        if y is None or not math.isfinite(y):return False,side+'_invalid_depth'
        if x is not None and math.isfinite(x):
            if y>x+1e-6:return False,side+'_depth_regression'
            improved|=y<x-1e-5
        if b.get('depth_samples',0)<a.get('depth_samples',0):return False,side+'_depth_support_regression'
    return (True,'verified_strict_improvement') if improved else (False,'no_strict_improvement')

def scene_promotes(old,new):
    if new is old:return False,'same_candidate'
    if not new or not new.get('report',{}).get('numerical_checks_passed'):return False,'candidate_failed_verification'
    if old is None:return True,'first_verified_candidate'
    receipt=new['report'].get('object_identity_cleanup')
    if receipt:
        from pathlib import Path
        import json,hashlib
        r=json.loads(Path(receipt).read_text())
        source=r['protected_sources']['source_thread']
        if (r.get('controller')=='agent' and r.get('runs') and source['path']==old.get('fragment_path')
            and hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
            and r.get('derived_fragment_sha256')==hashlib.sha256(Path(new['fragment_path']).read_bytes()).hexdigest()):
            return True,'stereo_and_3D_needle_duplicate_removed_with_original_preserved'

    a=old.get('report',{});b=new['report']
    if not a.get('numerical_checks_passed'):return True,'verification_recovered'
    af,bf=a.get('fragment_count'),b.get('fragment_count')
    if not isinstance(af,int) or not isinstance(bf,int):return False,'missing_fragment_metrics'
    if bf>af:return False,'fragment_count_regression'
    improved=bf<af
    for side in ['left','right']:
        x=a.get('visible_mask_coverage',{}).get(side);y=b.get('visible_mask_coverage',{}).get(side)
        if x is None or y is None:return False,'missing_fixed_reference_coverage'
        if not all(math.isfinite(v) for v in (x,y)):return False,'invalid_coverage'
        if y<x-1e-6:return False,side+'_visible_support_regression'
        improved|=y>x+1e-4
    return (True,'verified_strict_improvement') if improved else (False,'no_strict_improvement')

def selected_or_baseline(baseline,proposed,kind):
    promote,reason=(tool_promotes if kind=='tool' else scene_promotes)(baseline,proposed)
    return deepcopy(proposed if promote else baseline),dict(promoted=promote,reason=reason,independent_accuracy_measured=False)
