"""Read-only stereo diagnostics for bounded surgical scene refinement.

No geometry, calibration, masks, physical thresholds or hidden endpoints change.
All distances are estimates from the existing inputs, not independent truth.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from reconstruction_tools import fingerprints, sha256
from reconstruction_verification import load_base


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def project(points, K, baseline=0.):
    p = np.array(points, dtype=float, copy=True)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or np.any(p[:, 2] <= 0):
        raise ValueError('Expected finite points in front of the rectified camera')
    p[:, 0] -= baseline
    q = p @ np.asarray(K).T
    return q[:, :2] / q[:, 2:]


def line_coverage(mask, a, b):
    """Fraction of an image chord covered by a mask; not hidden-path evidence."""
    xy = np.linspace(a, b, max(2, int(np.linalg.norm(np.asarray(a)-b))+1))
    ij = np.rint(xy).astype(int)
    valid = ((ij[:, 0] >= 0) & (ij[:, 0] < mask.shape[1]) &
             (ij[:, 1] >= 0) & (ij[:, 1] < mask.shape[0]))
    return {'fraction_in_image': float(valid.mean()),
            'occluder_fraction': float(mask[ij[valid, 1], ij[valid, 0]].mean()) if valid.any() else 0.}


def inspect_thread_masks(inputs, pipeline_tools, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    base = load_base(pipeline_tools)
    relevant = {k: inputs[k] for k in ('left', 'right', 'thread_left_mask', 'thread_right_mask')}
    before = fingerprints(relevant)
    report = {'schema': 'thread_mask_evidence_v1', 'views': {},
              'independent_accuracy_measured': False, 'geometry_modified': False,
              'input_sha256': before}
    for side in ('left', 'right'):
        image = base.load_rgb(inputs[side]); mask = base.load_mask(inputs['thread_'+side+'_mask'], image.shape[:2])
        skel, graph, _, count, discarded = base.skeleton_graph(mask, 0., 30)
        endpoints = np.array([n for n in graph if graph.degree[n] == 1], dtype=int).reshape(-1, 2)
        junctions = np.array([n for n in graph if graph.degree[n] > 2], dtype=int).reshape(-1, 2)
        row = {'mask_pixels': int(mask.sum()), 'retained_components': count,
               'discarded_short_components': discarded, 'terminal_count': len(endpoints),
               'junction_pixel_count': len(junctions),
               'endpoints_xy': endpoints[:, ::-1].tolist(),
               'border_endpoint_ids': [i for i, (y, x) in enumerate(endpoints)
                                       if min(x, y, image.shape[1]-1-x, image.shape[0]-1-y) <= 3],
               'join_only_topology_possible': len(endpoints) == 2*count,
               'junctions_are_mask_skeleton_features_not_physical_branches': True}
        overlay = image.copy()
        overlay[mask] = (.55*overlay[mask] + .45*np.array([45, 225, 135])).astype('uint8')
        for y, x in junctions:
            cv2.circle(overlay, (int(x), int(y)), 3, (245, 75, 95), -1)
        for i, (y, x) in enumerate(endpoints):
            cv2.circle(overlay, (int(x), int(y)), 4, (255, 210, 55), -1)
            cv2.putText(overlay, str(i), (max(0,int(x)-13), max(14,int(y)-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1, cv2.LINE_AA)
        target = out / (side+'_mask_evidence.png')
        if not cv2.imwrite(str(target), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
            raise RuntimeError('Could not write diagnostic image')
        row['overlay'] = str(target.resolve()); report['views'][side] = row
    if fingerprints(relevant) != before:
        raise ValueError('Inputs changed during mask inspection')
    write_json(out/'report.json', report)
    return report


def inspect_attachment(bundle_path, inputs, baseline_m, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    relevant = {'bundle': str(bundle_path), **{k: inputs[k] for k in ('left','right')}}
    relevant.update({k: inputs[k] for k in ('left_occluder','right_occluder') if k in inputs})
    before = fingerprints(relevant)
    with np.load(bundle_path, allow_pickle=False) as d:
        t, n, K = d['points_camera_m'].copy(), d['needle_points_camera_m'].copy(), d['K'].copy()
        tn, nn = d['points_newton_m'].copy(), d['needle_points_newton_m'].copy()
    pairs = [(float(np.linalg.norm(t[ti]-n[ni])), ti, ni)
             for ti in (0,len(t)-1) for ni in (0,len(n)-1)]
    gap, ti, ni = min(pairs)
    delta = t[ti]-n[ni]
    invariance_error = max(abs(np.linalg.norm(t[a]-n[b])-np.linalg.norm(tn[a]-nn[b]))
                           for _,a,b in pairs)
    report = {'schema': 'attachment_diagnosis_v1', 'closest_pair': {'thread_index':ti, 'needle_index':ni},
              'endpoint_gap_mm':gap*1000, 'delta_camera_mm':(delta*1000).tolist(),
              'transverse_gap_mm':float(np.linalg.norm(delta[:2])*1000),
              'absolute_depth_gap_mm':float(abs(delta[2])*1000),
              'coordinate_transform_distance_error_mm':float(invariance_error*1000),
              'views':{}, 'protected_connector_limit_mm':4., 'input_sha256':before,
              'geometry_modified':False, 'independent_accuracy_measured':False,
              'hidden_geometry_recovered':False}
    for side in ('left','right'):
        image = cv2.imread(inputs[side])
        if image is None: raise ValueError('Could not read original stereo image')
        xy_t = project(t,K,baseline_m if side=='right' else 0.)
        xy_n = project(n,K,baseline_m if side=='right' else 0.)
        a,b = xy_t[ti],xy_n[ni]
        row = {'thread_endpoint_xy':a.tolist(), 'needle_endpoint_xy':b.tolist(),
               'projected_gap_px':float(np.linalg.norm(a-b)), 'occluder_supplied':side+'_occluder' in inputs}
        if row['occluder_supplied']:
            mask = cv2.imread(inputs[side+'_occluder'], cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != image.shape[:2]: raise ValueError('Invalid occluder mask')
            row.update(line_coverage(mask>0,a,b))
        for xy,color in ((xy_t,(90,210,60)),(xy_n,(255,210,30))):
            cv2.polylines(image,[np.rint(xy).astype('int32')],False,color,2,cv2.LINE_AA)
        cv2.line(image,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)),(190,75,240),2,cv2.LINE_AA)
        for xy,label,color in [(a,'Thread endpoint',(90,210,60)),(b,'Visible needle endpoint',(255,210,30))]:
            x,y=np.rint(xy).astype(int); cv2.circle(image,(x,y),5,color,-1)
            cv2.putText(image,label,(min(max(3,x+8),image.shape[1]-205),max(18,y-8)),
                        cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
        cv2.rectangle(image,(0,0),(image.shape[1],49),(20,25,30),-1)
        cv2.putText(image,f'{side.upper()}: estimated gap {gap*1000:.2f} mm / {row["projected_gap_px"]:.1f} px',
                    (8,19),cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(image,'Pink line is a diagnostic chord, NOT proposed geometry',
                    (8,40),cv2.FONT_HERSHEY_SIMPLEX,.40,(215,215,215),1,cv2.LINE_AA)
        target=out/(side+'_attachment_evidence.png'); cv2.imwrite(str(target),image)
        row['overlay']=str(target.resolve()); report['views'][side]=row
    left=report['views']['left']
    report['hypothesis'] = ('Intervening geometry may be hidden by the tool. The visible needle endpoint '
        'is not evidence of the physical thread attachment endpoint; do not extend the needle or enlarge '
        'the connector limit.') if left.get('occluder_fraction',0.) >= .55 else (
        'Mismatch needs image/endpoint/depth inspection; no validated attachment repair is established.')
    report['hypothesis_is_not_ground_truth']=True
    if fingerprints(relevant)!=before: raise ValueError('Attachment inputs changed during inspection')
    write_json(out/'report.json',report)
    return report


def mask_revision_audit(inputs, candidate_masks, pipeline_tools):
    """Audit every edit against frozen original masks; never convert this to accuracy."""
    base=load_base(pipeline_tools); report={}
    for side in ('left','right'):
        original=base.load_mask(inputs['thread_'+side+'_mask'],base.load_rgb(inputs[side]).shape[:2])
        candidate=base.load_mask(candidate_masks[side],original.shape)
        skeleton,*_=base.skeleton_graph(original,0.,30)
        needle=base.load_mask(inputs['needle_'+side+'_mask'],original.shape) if 'needle_'+side+'_mask' in inputs else np.zeros_like(original)
        distance=cv2.distanceTransform((~candidate).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        report[side]={'original_pixels':int(original.sum()),'candidate_pixels':int(candidate.sum()),
                      'added_pixels':int((candidate & ~original).sum()),
                      'removed_pixels':int((original & ~candidate).sum()),
                      'original_skeleton_coverage_at_2px':float((distance[skeleton]<=2.).mean()) if skeleton.any() else 0.,
                      'needle_overlap_pixels':int((candidate & needle).sum()),
                      'unchanged_original_sha256':sha256(inputs['thread_'+side+'_mask']),
                      'candidate_sha256':sha256(candidate_masks[side])}
    return {'schema':'mask_revision_audit_v1','views':report,'independent_accuracy_measured':False,
            'original_mask_is_not_ground_truth':True,
            'mask_revision_requires_inspection':True}


def propose_stereo_thread_mask(inputs, baseline_m, parameters, pipeline_tools, out):
    """Propose an edit using the frozen opposite-view mask and estimated depth.

    This is a segmentation hypothesis, NOT recovered hidden thread. It neither
    extends needle geometry nor joins components. Right mask remains unchanged.
    Invalid depth/left-image truncation never licenses removal of old pixels.
    """
    allowed={'mode','tolerance_px','recovery_band_px'}
    if set(parameters)!=allowed or parameters['mode'] not in ('trim','recover'):
        raise ValueError('Invalid stereo mask parameters')
    for key,lo,hi in [('tolerance_px',1,4),('recovery_band_px',2,8)]:
        value=parameters[key]
        if isinstance(value,bool) or not isinstance(value,int) or not lo<=value<=hi:
            raise ValueError('Out-of-bound integer '+key)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    relevant={k:inputs[k] for k in ('left','right','depth_npy','K','thread_left_mask','thread_right_mask')}
    before=fingerprints(relevant);base=load_base(pipeline_tools)
    left=base.load_rgb(inputs['left']);lm=base.load_mask(inputs['thread_left_mask'],left.shape[:2])
    rm=base.load_mask(inputs['thread_right_mask'],left.shape[:2]);K=base.load_intrinsics(inputs['K'])
    depth=np.squeeze(np.load(inputs['depth_npy'],allow_pickle=False))
    if depth.shape!=lm.shape:raise ValueError('Depth shape mismatch')
    y,x=np.indices(lm.shape,dtype='float32')
    valid=np.isfinite(depth)&(depth>=.03)&(depth<=.5)
    disparity=np.zeros_like(depth);disparity[valid]=K[0,0]*baseline_m/depth[valid]
    xr=x-disparity;valid &= (xr>=0)&(xr<=lm.shape[1]-1)
    distance=cv2.distanceTransform((~rm).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    sampled=cv2.remap(distance,xr.astype('float32'),y,cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT,borderValue=1e6)
    supported=valid&(sampled<=parameters['tolerance_px'])
    if parameters['mode']=='trim':
        candidate=lm & (supported | ~valid)
    else:
        band=cv2.distanceTransform((~lm).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        candidate=lm | (supported & (band<=parameters['recovery_band_px']))
    for side,mask in [('left',candidate),('right',rm)]:
        base_mask=out/(side+'_thread_mask.png')
        if not cv2.imwrite(str(base_mask),mask.astype('uint8')*255):raise RuntimeError('Mask write failed')
    if fingerprints(relevant)!=before:raise ValueError('Inputs changed during mask proposal')
    paths={s:str((out/(s+'_thread_mask.png')).resolve()) for s in ('left','right')}
    report={'schema':'stereo_mask_proposal_v1','parameters':parameters,'input_sha256':before,
            'proposed_masks':paths,'status':'proposal_only','estimated_depth_used':True,
            'new_hidden_geometry':False,'audit':mask_revision_audit(inputs,paths,pipeline_tools),
            'independent_accuracy_measured':False,'training_eligible':False}
    write_json(out/'report.json',report)
    return report
