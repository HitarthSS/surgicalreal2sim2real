"""Bounded image-coordinate occluder proposals shared by both controllers.

Controller-proposed polygons are uncertain perception, never ground truth.
They cannot cover observed needle/thread support, alter source pixels, or relax
any physical connection check. Full-frame overlays expose the hypothesis.
"""
from request_protocol_v11 import construct_polygon_request
from pathlib import Path
import cv2
import numpy as np
from reconstruction_tools import fingerprints
from complex_case_diagnostics import write_json
from run_shared_refiner_v6 import conservative_occluder


def polygon_masks(parameters,shape):
    if set(parameters)!={'polygons'} or set(parameters['polygons'])!={'left','right'}:
        raise ValueError('Require polygons for exactly both stereo views')
    h,w=shape;masks={}
    for side,polygons in parameters['polygons'].items():
        if not isinstance(polygons,list) or len(polygons)>2:raise ValueError('At most two visible instruments per view')
        mask=np.zeros(shape,'uint8')
        for polygon in polygons:
            points=np.asarray(polygon)
            if (points.dtype.kind not in 'fi' or points.ndim!=2 or points.shape[1]!=2 or not 3<=len(points)<=16
                or not np.isfinite(points).all() or np.any(points[:,0]<0) or np.any(points[:,0]>w-1)
                or np.any(points[:,1]<0) or np.any(points[:,1]>h-1)):
                raise ValueError('Finite 3–16 vertex polygons within original image required')
            points=np.rint(points).astype('int32');part=np.zeros(shape,'uint8');cv2.fillPoly(part,[points],1)
            boundary=bool(part[0].any() or part[-1].any() or part[:,0].any() or part[:,-1].any())
            area=int(part.sum())
            if not boundary or not 256<=area<=h*w*.30:raise ValueError('Instrument mask must enter boundary and respect area sanity bound')
            mask|=part
        if int(mask.sum())>h*w*.60:raise ValueError('Combined two-tool area exceeds bound')
        masks[side]=mask>0
    return masks


def propose_occluder_polygons(case,parameters,out,candidate=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);inputs=case['inputs'];before=fingerprints(inputs)
    shapes={side:cv2.imread(inputs[side]).shape[:2] for side in ('left','right')}
    if shapes['left']!=shapes['right']:raise ValueError('Shared tool requires matching stereo dimensions')
    shape=shapes['left']
    parameters,request_audit=construct_polygon_request(parameters,{s:[v[1],v[0]] for s,v in shapes.items()})
    masks=polygon_masks(parameters,shape);report=dict(
        schema='occluder_polygon_hypothesis_v11',parameters=parameters,request_construction=request_audit,views={},feedback_images={},masks={},
        semantic_correctness_validated=False,source_masks_modified=False,independent_accuracy_measured=False,
        training_eligible=False,visual_review_required=True,source='controller-proposed bounded tool polygons')
    for side,mask in masks.items():
        needle=cv2.imread(inputs['needle_'+side+'_mask'],0);thread=cv2.imread(inputs['thread_'+side+'_mask'],0)
        previous=inputs.get(side+'_occluder')
        prior=cv2.imread(previous,0) if previous else np.zeros(shape,'uint8')
        if prior is None or prior.shape!=shape:raise ValueError('Prior occlusion evidence shape mismatch')
        mask,audit=conservative_occluder(mask|(prior>0),needle,thread)
        if int(mask.sum())>mask.size*.60:raise ValueError('Accumulated occlusion exceeds full-frame area bound')
        audit['prior_occlusion_pixels']=int((prior>0).sum());audit['prior_evidence_preserved']=True
        path=out/(side+'_occluder.png')
        cv2.imwrite(str(path),mask.astype('uint8')*255);report['masks'][side]=str(path.resolve());report['views'][side]=audit
        image=cv2.imread(candidate['images'][side] if candidate else inputs[side])
        image[mask]=(image[mask]*.60+np.array([190,60,220])*.40).astype('uint8')
        cv2.putText(image,'Magenta: approximate tool polygon, not measured occlusion',(6,18),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1,cv2.LINE_AA)
        overlay=out/(side+'_occluder_feedback.png');cv2.imwrite(str(overlay),image)
        report['feedback_images'][side]=str(overlay.resolve())
    if fingerprints(inputs)!=before:raise ValueError('Source inputs changed')
    write_json(out/'report.json',report);return report
