"""Join one supported endpoint pair, preserving every other fragment.

This supports multiple gaps without weakening existing stereo/shape/depth
constraints. Independent gaps are not competing matches; alternatives sharing
an endpoint remain subject to the ambiguity gate. No scene/physics approval.
"""
from dataclasses import asdict
from pathlib import Path
import cv2
import numpy as np
from skimage.morphology import skeletonize
from reconstruction_tools import fingerprints
from thread_visibility_repairs_v5 import (RepairLimits,load_inputs,load_parts,join_proposals,
    project,save_parts,overlays,endpoint_table,in_view)
from complex_case_diagnostics import write_json


def separate_coverage(parts,masks,K,baseline,limit):
    result={}
    for side,mask in masks.items():
        canvas=np.zeros(mask.shape,'uint8')
        for part in parts:
            xy=project(part['points_m'],K,baseline if side=='right' else 0.)
            cv2.polylines(canvas,[np.rint(xy).astype('int32')],False,1,1)
        distance=cv2.distanceTransform(1-canvas,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        skeleton=skeletonize(mask)
        result[side]=float((distance[skeleton]<=limit).mean()) if skeleton.any() else 0.
    return result


def connect_fragment_pair(case,pipeline,fragment_path,proposal_id,out,limits=RepairLimits(),
                          inference_mode=None,selection_evidence=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    before=fingerprints({**case['inputs'],'fragments':fragment_path});parts=load_parts(fragment_path)
    if len(parts)<2:raise ValueError('At least two explicit fragments required')
    rows=join_proposals(case,pipeline,parts,limits)
    _,images,masks,_,_,K,baseline=load_inputs(case,pipeline)
    for row in rows:
        curve=row['_curve']
        outside=(~in_view(project(curve,K),masks['left'].shape)) & (~in_view(project(curve,K,baseline),masks['right'].shape))
        if outside[5:-5].any():
            row['failures']=sorted(set(row['failures']+['unobserved_out_of_frame_connection']))
            row['numerical_checks_passed']=False
    selected=next((r for r in rows if r['proposal_id']==proposal_id),None)
    if selected is None:raise ValueError('Unknown/stale endpoint-pair proposal')
    failures=list(selected['failures']);ends=set(selected['endpoints'])
    uncertainty=[]
    if inference_mode is not None:
        if inference_mode not in ('boundary','topology'):raise ValueError('Unknown bounded inference mode')
        evidence=selection_evidence or {}
        if (not {'image.left','image.right'} <= set(evidence.get('evidence_ids',[]))
                or not isinstance(evidence.get('reason'),str) or not evidence['reason'].strip()):
            raise ValueError('Inferred connection requires a logged stereo hypothesis assessment')
        if inference_mode=='boundary':
            alternatives={tuple(sorted(r['endpoints'])) for r in rows
                          if set(r['failures'])<= {'unobserved_out_of_frame_connection'}}
            if (len(parts)!=2 or len(alternatives)!=1 or tuple(sorted(ends)) not in alternatives
                    or set(failures)!={'unobserved_out_of_frame_connection'}):
                raise ValueError('Boundary inference requires two fragments and a unique otherwise admissible pair')
            # Only this epistemic gate becomes a warning. Distance, depth,
            # shape, intersections and visible-background checks still apply.
            failures.remove('unobserved_out_of_frame_connection')
            uncertainty.append('unobserved_out_of_frame_connection')
    competitors=[r for r in rows if r['numerical_checks_passed'] and set(r['endpoints'])!=ends and ends.intersection(r['endpoints'])]
    if competitors and min(r['score'] for r in competitors)-selected['score']<limits.ambiguity_score_margin:
        if inference_mode=='topology' and not selected['failures']:
            uncertainty.append('ambiguous_endpoint_pair')
        else:failures.append('ambiguous_endpoint_pair')
    curve=selected['_curve']
    outside_both=(~in_view(project(curve,K),masks['left'].shape)) & (~in_view(project(curve,K,baseline),masks['right'].shape))
    # Neither image constrains an excursion beyond both fields of view.
    # A smooth one-thread hypothesis cannot identify its reconnection there.
    if outside_both[5:-5].any() and inference_mode!='boundary':failures.append('unobserved_out_of_frame_connection')
    fa,ea=map(int,(selected['endpoints'][0].split('.')[0][1:],selected['endpoints'][0][-1]))
    fb,eb=map(int,(selected['endpoints'][1].split('.')[0][1:],selected['endpoints'][1][-1]))
    output=parts;added=0
    if not failures:
        a={k:(v[::-1] if ea==0 else v) for k,v in parts[fa].items()}
        b={k:(v[::-1] if eb==1 else v) for k,v in parts[fb].items()}
        curve=selected['_curve'][1:-1];added=len(curve);joined={}
        for key in ['points_m','depth_inferred','geometry_inferred','observed_left','observed_right']:
            middle=curve if key=='points_m' else np.full(added,key in ['depth_inferred','geometry_inferred'],bool)
            joined[key]=np.concatenate((a[key],middle,b[key]))
        joined['reference_xy']=project(joined['points_m'],K,baseline)
        output=[joined if i==min(fa,fb) else part for i,part in enumerate(parts) if i!=max(fa,fb)]
        cov=separate_coverage(output,masks,K,baseline,limits.mask_check_px)
        if min(cov.values())<limits.min_visible_coverage:failures.append('visible_fragments_not_fully_represented')
    else:cov=separate_coverage(parts,masks,K,baseline,limits.mask_check_px)
    # Rejected proposals still preserve inspectable prior fragments.
    if failures:output=parts;added=0
    path=out/'fragments.npz';save_parts(path,output);overlays(output,images,K,baseline,out)
    report=dict(schema='partial_thread_pair_assembly_v7',status='proposal_only',reference_view='mixed',
        selected={k:v for k,v in selected.items() if not k.startswith('_')},
        competing_matches=[{k:v for k,v in r.items() if not k.startswith('_')} for r in competitors],
        failures=sorted(set(failures)),numerical_checks_passed=not failures,
        fragment_count=len(output),remaining_fragment_count=len(output),inferred_shape_samples=added,
        unselected_fragments_preserved=[i for i in range(len(parts)) if i not in [fa,fb]],
        source_fragment_count=len(parts),joined_source_fragments=[fa,fb] if not failures else [],
        inference_mode=inference_mode,uncertainty_warnings=uncertainty,
        connected_hypothesis_only=bool(inference_mode and not failures),
        selection_evidence=selection_evidence if inference_mode else None,
        original_observations_preserved=True,hidden_curve_identified=False,
        visible_mask_coverage=cov,endpoints=endpoint_table(output,K,baseline,masks['left'].shape,limits),
        outside_both_views_fraction=float(outside_both.mean()),
        limits=asdict(limits),provenance=before,single_thread_prior=case['scene_priors'],
        independent_accuracy_measured=False,hidden_geometry_is_ground_truth=False,
        training_eligible=False,shared_reconstruction_accepted=False,physical_collision_validated=False)
    if fingerprints({**case['inputs'],'fragments':fragment_path})!=before:raise ValueError('Protected input changed')
    write_json(out/'report.json',report);return report
