"""Object-class ambiguity audit; original masks and physical gates stay fixed."""
from pathlib import Path
import cv2
import numpy as np
from skimage.morphology import skeletonize
from reconstruction_tools import fingerprints
from complex_case_diagnostics import write_json
from thread_visibility_repairs_v5 import load_inputs,load_parts,project,RepairLimits

SHORT_COMPONENT_PIXELS=30 # Existing reconstruction minimum.
NEEDLE_BOUNDARY_TOLERANCE_PX=2. # Existing strict stereo/mask scale.


def classify_skeleton(mask,needle_mask,needle_points,needle_inferred,K,baseline):
    skeleton=skeletonize(mask);count,labels=cv2.connectedComponents(skeleton.astype('uint8'),connectivity=8)
    sizes=np.bincount(labels.ravel(),minlength=count)
    short=skeleton&(sizes[labels]<SHORT_COMPONENT_PIXELS)
    near_mask=(cv2.distanceTransform((~needle_mask).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        <=NEEDLE_BOUNDARY_TOLERANCE_PX) if needle_mask.any() else np.zeros_like(mask)
    canvas=np.zeros(mask.shape,'uint8')
    if needle_points is not None:
        p=np.asarray(needle_points);flags=np.asarray(needle_inferred)
        if p.ndim!=2 or p.shape[1]!=3 or flags.shape!=(len(p),) or flags.dtype!=bool or not np.isfinite(p).all():
            raise ValueError('Unchanged fitted needle and boolean support flags required')
        xy=project(p,K,baseline)
        for i in range(len(xy)-1):
            if not flags[i] and not flags[i+1]:
                cv2.line(canvas,tuple(np.rint(xy[i]).astype(int)),tuple(np.rint(xy[i+1]).astype(int)),1,1)
    near_arc=(cv2.distanceTransform(1-canvas,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)<=RepairLimits().mask_check_px
        if canvas.any() else np.zeros_like(mask))
    ambiguous=short&near_mask&near_arc
    return skeleton,ambiguous,dict(original_skeleton_pixels=int(skeleton.sum()),short_feature_pixels=int(short.sum()),
        ambiguous_needle_adjacent_skeleton_pixels=int(ambiguous.sum()),
        unambiguous_thread_skeleton_pixels=int((skeleton&~ambiguous).sum()),
        original_skeleton_pixels_removed=0,physical_object_identity_measured=False)


def evaluate_thread_object_evidence(case,pipeline,fragment_path,needle_path=None,out=None):
    sources={**case['inputs'],'thread_fragments':str(fragment_path),'verifier':str(Path(__file__).resolve())}
    points=flags=None
    if needle_path is not None:
        sources['unchanged_needle']=str(needle_path)
        with np.load(needle_path,allow_pickle=False) as data:points=data['points_m'].copy();flags=data['inferred'].copy()
    before=fingerprints(sources);base,images,masks,_,_,K,baseline=load_inputs(case,pipeline)
    parts=load_parts(fragment_path);views={};overlays={}
    for side,mask in masks.items():
        needle=base.load_mask(case['inputs']['needle_'+side+'_mask'],mask.shape) if 'needle_'+side+'_mask' in case['inputs'] else np.zeros_like(mask)
        skeleton,ambiguous,metrics=classify_skeleton(mask,needle,points,flags,K,baseline if side=='right' else 0.)
        canvas=np.zeros(mask.shape,'uint8')
        for part in parts:
            xy=project(part['points_m'],K,baseline if side=='right' else 0.)
            cv2.polylines(canvas,[np.rint(xy).astype('int32')],False,1,1)
        distance=cv2.distanceTransform(1-canvas,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        covered=distance<=RepairLimits().mask_check_px;required=skeleton&~ambiguous
        metrics.update(raw_original_mask_coverage=float(covered[skeleton].mean()) if skeleton.any() else 0.,
            object_aware_thread_coverage=float(covered[required].mean()) if required.any() else 0.,
            unexplained_thread_skeleton_pixels=int((required&~covered).sum()),ambiguous_uncovered_pixels=int((ambiguous&~covered).sum()),
            ambiguous_pixel_xy=np.argwhere(ambiguous)[:,::-1].tolist(),
            unexplained_thread_pixel_xy=np.argwhere(required&~covered)[:,::-1].tolist())
        views[side]=metrics
        image=cv2.cvtColor(images[side],cv2.COLOR_RGB2BGR)
        image[required&covered]=(65,220,65);image[required&~covered]=(35,50,240);image[ambiguous]=(220,40,220);overlays[side]=image
    failures=[]
    if any(v['unambiguous_thread_skeleton_pixels']==0 for v in views.values()):failures.append('no_unambiguous_thread_evidence')
    if any(v['object_aware_thread_coverage']<RepairLimits().min_visible_coverage for v in views.values()):failures.append('original_visible_support_lost')
    report=dict(schema='thread_object_evidence_v8',views=views,failures=failures,numerical_coverage_checks_passed=not failures,
        warnings=['ambiguous_short_needle_adjacent_features'] if any(v['ambiguous_needle_adjacent_skeleton_pixels'] for v in views.values()) else [],
        criteria=dict(short_component_pixels=SHORT_COMPONENT_PIXELS,needle_mask_tolerance_px=NEEDLE_BOUNDARY_TOLERANCE_PX,
            observed_needle_arc_tolerance_px=RepairLimits().mask_check_px,thread_projection_tolerance_px=RepairLimits().mask_check_px,
            min_thread_coverage=RepairLimits().min_visible_coverage),original_mask_is_ground_truth=False,
        ambiguous_features_are_confirmed_needle=False,independent_accuracy_measured=False,training_eligible=False,source_fingerprints=before)
    if out is not None:
        out=Path(out);out.mkdir(parents=True,exist_ok=False);report['overlays']={}
        for side,image in overlays.items():
            path=out/(side+'_object_evidence.png');cv2.imwrite(str(path),image);report['overlays'][side]=str(path.resolve())
        write_json(out/'report.json',report)
    if fingerprints(sources)!=before:raise ValueError('Object evidence changed during verification')
    return report
