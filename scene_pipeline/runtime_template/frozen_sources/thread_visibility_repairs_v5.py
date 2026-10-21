"""Separate cross-view and single-thread repair proposals; never ground truth.

Existing masks, QP, occlusion connector, physical limits and strict Newton
adapter are untouched. Every controller uses these same bounded functions.
"""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import cv2
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import median_filter
from complex_case_diagnostics import project, write_json
from reconstruction_tools import fingerprints
from reconstruction_verification import load_base


@dataclass(frozen=True)
class RepairLimits:
    stereo_match_px: float = 2.
    mask_check_px: float = 8.
    min_anchor_fraction: float = .70
    max_unanchored_span_px: float = 160.
    min_depth_m: float = .03
    max_depth_m: float = .5
    max_depth_relative_difference: float = .25
    max_endpoint_gap_m: float = .04
    max_hidden_length_m: float = .06
    min_explained_fraction: float = .8
    min_visible_coverage: float = .85
    ambiguity_score_margin: float = .10
    tangent_length_px: float = 15.
    tangent_scales: tuple = (.35, .65, 1.)

    def __post_init__(self):
        if self.stereo_match_px not in (2.,4.): raise ValueError('Use frozen strict/relaxed correspondence cards')
        # All safety and acceptance limits are fixed, never model-controlled.
        expected=dict(mask_check_px=8.,min_anchor_fraction=.70,max_unanchored_span_px=160.,
            min_depth_m=.03,max_depth_m=.5,max_depth_relative_difference=.25,
            max_endpoint_gap_m=.04,max_hidden_length_m=.06,min_explained_fraction=.8,
            min_visible_coverage=.85,ambiguity_score_margin=.10,tangent_length_px=15.,
            tangent_scales=(.35,.65,1.))
        if any(getattr(self,k)!=v for k,v in expected.items()):raise ValueError('Protected geometry/verification limits cannot be tuned')


def load_inputs(case, pipeline):
    base = load_base(pipeline); inputs = case['inputs']
    images = {s:base.load_rgb(inputs[s]) for s in ('left','right')}
    if images['left'].shape != images['right'].shape: raise ValueError('Stereo dimensions differ')
    shape = images['left'].shape[:2]; K = base.load_intrinsics(inputs['K'])
    if not np.isfinite(K).all() or not np.allclose(K[2],[0,0,1]): raise ValueError('Invalid K')
    baseline = case['baseline_m']
    if isinstance(baseline,bool) or not np.isfinite(baseline) or baseline <= 0: raise ValueError('Invalid baseline')
    depth = np.squeeze(np.load(inputs['depth_npy'],allow_pickle=False))
    if depth.shape != shape: raise ValueError('Depth dimensions differ')
    masks = {s:base.load_mask(inputs['thread_'+s+'_mask'],shape) for s in images}
    occ = {s:base.load_mask(inputs[s+'_occluder'],shape) if s+'_occluder' in inputs else np.zeros(shape,bool) for s in images}
    return base, images, masks, occ, depth, K, baseline


def distances(mask, xy):
    dist = cv2.distanceTransform((~mask).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    return cv2.remap(dist,xy[:,0].astype('float32')[None],xy[:,1].astype('float32')[None],
                     cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=1e6).ravel()


def in_view(xy, shape):
    h,w = shape
    return (xy[:,0]>=0)&(xy[:,0]<=w-1)&(xy[:,1]>=0)&(xy[:,1]<=h-1)


def at(mask,xy):
    inside=in_view(xy,mask.shape); ij=np.rint(xy[inside]).astype(int)
    result=np.zeros(len(xy),bool);result[inside]=mask[ij[:,1],ij[:,0]]
    return result


def coverage(points, masks, K, baseline, limit):
    result={}
    from skimage.morphology import skeletonize
    for s in ('left','right'):
        xy=project(points,K,baseline if s=='right' else 0.)
        canvas=np.zeros(masks[s].shape,'uint8')
        cv2.polylines(canvas,[np.rint(xy).astype('int32')],False,1,1)
        d=cv2.distanceTransform(1-canvas,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        skel=skeletonize(masks[s]);result[s]=float((d[skel]<=limit).mean()) if skel.any() else 0.
    return result


def max_missing_span(valid, arclength):
    longest=0.;start=None
    for i,v in enumerate(valid):
        if not v and start is None:start=max(0,i-1)
        if v and start is not None:longest=max(longest,float(arclength[i]-arclength[start]));start=None
    if start is not None:longest=max(longest,float(arclength[-1]-arclength[start]))
    return longest


def endpoint_id(fragment, end): return f'f{fragment}.e{end}'


def endpoint_table(parts,K,baseline,shape,limits=RepairLimits()):
    rows=[]
    for i,part in enumerate(parts):
        xy=part['reference_xy']; points=part['points_m']
        s=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
        for end,index in ((0,0),(1,len(points)-1)):
            inner=int(np.searchsorted(s,limits.tangent_length_px)) if end==0 else int(np.searchsorted(s,s[-1]-limits.tangent_length_px))
            inner=min(max(inner,0),len(points)-1)
            delta=points[index]-points[inner];norm=np.linalg.norm(delta)
            if norm<1e-9:raise ValueError('Endpoint tangent not estimable')
            p=points[index:index+1]; right=project(p,K,baseline)[0];left=project(p,K)[0]
            rows.append(dict(endpoint_id=endpoint_id(i,end),fragment=i,end=end,points_m=p[0].tolist(),
                outward_tangent=(delta/norm).tolist(),left_xy=left.tolist(),right_xy=right.tolist(),
                near_boundary=bool(min(left[0],left[1],shape[1]-1-left[0],shape[0]-1-left[1],
                                       right[0],right[1],shape[1]-1-right[0],shape[0]-1-right[1])<=6),
                depth_inferred=bool(part['depth_inferred'][index])))
    return rows


def save_parts(path,parts):
    values={}
    for i,part in enumerate(parts):
        for key,value in part.items():values[f'f{i}_{key}']=value
    np.savez_compressed(path,**values)


def load_parts(path):
    with np.load(path,allow_pickle=False) as a:
        ids=sorted({int(k.split('_')[0][1:]) for k in a.files})
        return [{k.split('_',1)[1]:a[k].copy() for k in a.files if k.startswith(f'f{i}_')} for i in ids]


def cross_view_recovery(case,pipeline,reference,out,limits=RepairLimits()):
    if reference not in ('left','right'): raise ValueError('Unknown reference view')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    before=fingerprints(case['inputs']);base,images,masks,occ,depth,K,baseline=load_inputs(case,pipeline)
    _,graph,_,count,discarded=base.skeleton_graph(masks[reference],0.,30)
    ordered=[]
    for nodes in sorted(nx.connected_components(graph),key=lambda x:min(x)):
        sub=graph.subgraph(nodes);ends=sorted(n for n in sub if sub.degree[n]==1)
        if len(ends)!=2 or any(sub.degree[n]>2 for n in sub):
            raise ValueError('Reference mask branches/loops; segmentation repair needed before path recovery')
        ordered.append(np.asarray(nx.shortest_path(sub,*ends,weight='weight'),float))
    return lift_ordered_reference_paths(case,pipeline,reference,out,ordered,count,discarded,limits)


def lift_ordered_reference_paths(case,pipeline,reference,out,ordered,count,discarded,limits=RepairLimits()):
    """Shared lifting/verification for ordinary paths and explicit topology hypotheses.

    Callers must validate derived paths against the immutable source skeleton;
    ordinary cross_view_recovery retains its original ordering and behavior.
    """
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    before=fingerprints(case['inputs']);base,images,masks,occ,depth,K,baseline=load_inputs(case,pipeline)
    if reference not in ('left','right') or not ordered or count!=len(ordered):
        raise ValueError('Invalid ordered reference paths')
    for path in ordered:
        if path.ndim!=2 or path.shape[1]!=2 or len(path)<2 or not np.isfinite(path).all():
            raise ValueError('Finite ordered row/column paths required')
    _,left_graph,_,_,_=base.skeleton_graph(masks['left'],0.,30)
    lrc=np.array(sorted(left_graph),float)
    z,_=base.sample_depth(depth,masks['left'],lrc,3,limits.min_depth_m,limits.max_depth_m)
    good=np.isfinite(z);lrc,z=lrc[good],z[good]
    if not len(z):raise ValueError('No visible left-thread depth anchors')
    xyz=base.backproject(lrc,z,K); anchor_xy=project(xyz,K,baseline if reference=='right' else 0.)
    reference_xy=np.vstack([rc[:,::-1] for rc in ordered]);tree=cKDTree(reference_xy)
    distance,index=tree.query(anchor_xy)
    component_ids=np.concatenate([np.full(len(rc),i) for i,rc in enumerate(ordered)])
    buckets={};ambiguous=0
    for anchor,d,ix,zz in zip(anchor_xy,distance,index,z):
        if d<=limits.stereo_match_px:
            neighbors=tree.query_ball_point(anchor,limits.stereo_match_px)
            if len(set(component_ids[neighbors]))>1:
                ambiguous+=1;continue
            buckets.setdefault(int(ix),[]).append((float(zz),float(d)))
    parts=[];metrics=[];offset=0;failures=[]
    for rc in ordered:
        xy=rc[:,::-1];s=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
        raw=np.full(len(rc),np.nan);residual=np.full(len(rc),np.nan)
        for i in range(len(rc)):
            pairs=buckets.get(offset+i,[]);values=[p[0] for p in pairs]
            if values and (max(values)-min(values))/np.median(values)<=limits.max_depth_relative_difference:
                raw[i]=np.median(values)
                residual[i]=np.median([p[1] for p in pairs])
        valid=np.isfinite(raw);fraction=float(valid.mean());span=max_missing_span(valid,s)
        if valid.sum()<2:raise ValueError('Reference fragment has fewer than two consistent stereo/depth anchors')
        # Missing depth uses neighboring estimated anchors, never tool/background depth.
        clean=np.interp(s,s[valid],raw[valid]);clean=median_filter(clean,size=9,mode='nearest')
        left_rc=rc.copy()
        if reference=='right':left_rc[:,1]+=K[0,0]*baseline/clean
        points=base.backproject(left_rc,clean,K)
        part=dict(points_m=points,reference_xy=xy,depth_inferred=~valid,
                  observed_left=(distances(masks['left'],project(points,K))<=limits.stereo_match_px),
                  observed_right=(distances(masks['right'],project(points,K,baseline))<=limits.stereo_match_px),
                  correspondence_residual_px=np.where(valid,residual,-1.),
                  geometry_inferred=np.zeros(len(points),bool))
        parts.append(part);offset+=len(rc)
        metrics.append(dict(samples=len(rc),depth_anchor_fraction=fraction,max_unanchored_span_px=span,
                            depth_inferred_samples=int((~valid).sum()),
                            loose_correspondence_samples=int((residual>2.).sum())))
        if fraction<limits.min_anchor_fraction:failures.append('insufficient_visible_depth_anchors')
        if span>limits.max_unanchored_span_px:failures.append('unbounded_right_only_depth_span')
        if np.max(clean)/np.min(clean)-1>limits.max_depth_relative_difference:failures.append('excessive_fragment_depth_range')
    # Each fragment separately; do not rasterize a fictitious join between them.
    cov={s:0. for s in masks}
    from skimage.morphology import skeletonize
    for s in masks:
        canvas=np.zeros(masks[s].shape,'uint8')
        for part in parts:
            xy=project(part['points_m'],K,baseline if s=='right' else 0.)
            cv2.polylines(canvas,[np.rint(xy).astype('int32')],False,1,1)
        d=cv2.distanceTransform(1-canvas,cv2.DIST_L2,cv2.DIST_MASK_PRECISE);skel=skeletonize(masks[s])
        cov[s]=float((d[skel]<=limits.mask_check_px).mean())
    if min(cov.values())<limits.min_visible_coverage:failures.append('visible_fragments_not_fully_represented')
    save_parts(out/'fragments.npz',parts)
    report=dict(schema='cross_view_recovery_v5',reference_view=reference,status='proposal_only',
                numerical_checks_passed=not failures,failures=sorted(set(failures)),fragment_count=count,
                discarded_short_components=discarded,fragments=metrics,visible_mask_coverage=cov,
                ambiguous_cross_fragment_anchors_excluded=ambiguous,
                endpoints=endpoint_table(parts,K,baseline,masks['left'].shape,limits),limits=asdict(limits),
                provenance=before,independent_accuracy_measured=False,training_eligible=False,
                labels={'observed_left/right':'2D mask support, not measured 3D',
                        'depth_inferred':'neighboring estimated depth interpolation or endpoint extrapolation',
                        'geometry_inferred':'curve shape hidden from both images'},
                ordered_reference_path_observed=True,source_masks_modified=False)
    overlays(parts,images,K,baseline,out)
    if fingerprints(case['inputs'])!=before:raise ValueError('Inputs changed')
    write_json(out/'report.json',report);return report


def hermite3d(a,b,ta,tb,scale,n=101):
    t=np.linspace(0,1,n)[:,None];gap=np.linalg.norm(b-a)
    return (2*t**3-3*t**2+1)*a+(t**3-2*t**2+t)*ta*gap*scale+(-2*t**3+3*t**2)*b+(t**3-t**2)*(-tb)*gap*scale


def join_proposals(case,pipeline,parts,limits=RepairLimits()):
    if case.get('scene_priors',{}).get('single_thread') is not True:
        raise ValueError('Single-thread prior must be explicitly declared; never inferred from model confidence')
    _,_,masks,occ,_,K,baseline=load_inputs(case,pipeline)
    endpoints=endpoint_table(parts,K,baseline,masks['left'].shape,limits);rows=[]
    for i,a in enumerate(endpoints):
        for b in endpoints[i+1:]:
            if a['fragment']==b['fragment']:continue
            for scale in limits.tangent_scales:
                pa,pb=np.asarray(a['points_m']),np.asarray(b['points_m'])
                curve=hermite3d(pa,pb,np.asarray(a['outward_tangent']),np.asarray(b['outward_tangent']),scale)
                lengths=np.linalg.norm(np.diff(curve,axis=0),axis=1);gap=float(np.linalg.norm(pb-pa));length=float(lengths.sum())
                failures=[];views={}
                if gap>limits.max_endpoint_gap_m:failures.append('metric_endpoint_gap_cap')
                if length>limits.max_hidden_length_m:failures.append('hidden_length_cap')
                if min(pa[2],pb[2])<limits.min_depth_m or max(pa[2],pb[2])>limits.max_depth_m:failures.append('depth_range')
                if abs(pa[2]-pb[2])/min(pa[2],pb[2])>limits.max_depth_relative_difference:failures.append('endpoint_depth_mismatch')
                for s in masks:
                    xy=project(curve,K,baseline if s=='right' else 0.)
                    padded=cv2.dilate(occ[s].astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))>0
                    visible=in_view(xy,masks[s].shape);supported=distances(masks[s],xy)<=limits.mask_check_px
                    # Hidden may be outside the camera or behind the segmented tool.
                    explained=(~visible)|at(padded,xy)|supported
                    fraction=float(explained[5:-5].mean())
                    views[s]=dict(explained_fraction=fraction,outside_fraction=float((~visible).mean()),
                                  occluder_fraction=float(at(padded,xy).mean()),mask_support_fraction=float(supported.mean()))
                    if fraction<limits.min_explained_fraction:failures.append(s+'_visible_background_contradiction')
                # Reject near-intersections with unrelated visible thread interiors.
                interiors=[p['points_m'][30:-30] for p in parts if len(p['points_m'])>60]
                combined=np.vstack(interiors) if interiors else np.empty((0,3))
                if len(combined):
                    near=cKDTree(combined).query(curve[10:-10])[0]
                    if near.min()<.0005:failures.append('visible_thread_intersection')
                directions=np.diff(curve,axis=0);unit=directions/np.maximum(lengths[:,None],1e-12)
                turn=np.arccos(np.clip(np.sum(unit[:-1]*unit[1:],axis=1),-1,1))
                if lengths.min()<1e-9 or turn.max()>.2:failures.append('cusp_or_abrupt_direction_change')
                score=float(np.mean([1-v['explained_fraction'] for v in views.values()])+.2*(length/max(gap,1e-9)-1))
                rows.append(dict(proposal_id=a['endpoint_id']+'__'+b['endpoint_id']+'__'+str(scale),
                    endpoints=[a['endpoint_id'],b['endpoint_id']],tangent_scale=scale,endpoint_gap_m=gap,
                    hidden_length_m=length,max_discrete_turn_rad=float(turn.max()),views=views,
                    failures=sorted(set(failures)),numerical_checks_passed=not failures,score=score,
                    _curve=curve))
    return rows


def smooth_connection(case,pipeline,fragment_path,proposal_id,out,limits=RepairLimits()):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    before=fingerprints({**case['inputs'],'fragments':fragment_path});parts=load_parts(fragment_path)
    rows=join_proposals(case,pipeline,parts,limits)
    selected=next((r for r in rows if r['proposal_id']==proposal_id),None)
    if selected is None:raise ValueError('Unknown or stale endpoint/parameter proposal ID')
    # Similar shapes of the SAME endpoint pair are expected uncertainty, not alternatives in topology.
    competitors=[r for r in rows if r['numerical_checks_passed'] and r['endpoints']!=selected['endpoints']]
    failures=list(selected['failures'])
    if competitors and min(r['score'] for r in competitors)-selected['score']<limits.ambiguity_score_margin:
        failures.append('ambiguous_endpoint_pair')
    if len(parts)!=2:failures.append('more_than_two_fragments_need_explicit_assembly')
    _,images,masks,_,_,K,baseline=load_inputs(case,pipeline)
    public=[{k:v for k,v in r.items() if not k.startswith('_')} for r in rows]
    report=dict(schema='single_thread_smooth_connection_v5',status='proposal_only',proposal_id=proposal_id,
        selected={k:v for k,v in selected.items() if not k.startswith('_')},alternatives=public,
        failures=sorted(set(failures)),numerical_checks_passed=False,single_thread_prior=case['scene_priors'],
        independent_accuracy_measured=False,hidden_geometry_is_ground_truth=False,training_eligible=False,
        physical_collision_validated=False,limits=asdict(limits),provenance=before)
    if not failures:
        a,b=selected['endpoints'];fa,ea=map(int,(a[1:a.index('.')],a[-1]));fb,eb=map(int,(b[1:b.index('.')],b[-1]))
        first={k:v[::-1] if ea==0 else v for k,v in parts[fa].items()}
        second={k:v[::-1] if eb==1 else v for k,v in parts[fb].items()}
        curve=selected['_curve'][1:-1];n=len(curve)
        joined={}
        for k in ('points_m','depth_inferred','geometry_inferred','observed_left','observed_right'):
            middle=curve if k=='points_m' else np.full(n,k in ('depth_inferred','geometry_inferred'),bool)
            joined[k]=np.concatenate((first[k],middle,second[k]))
        joined['reference_xy']=project(joined['points_m'],K,baseline)
        cov=coverage(joined['points_m'],masks,K,baseline,limits.mask_check_px)
        if min(cov.values())<limits.min_visible_coverage:failures.append('visible_fragments_not_fully_represented')
        report.update(visible_mask_coverage=cov,failures=failures,numerical_checks_passed=not failures,
                      remaining_fragment_count=1,observed_fragment_ids_preserved=[fa,fb],
                      inferred_shape_samples=n,depth_inferred_samples=int(joined['depth_inferred'].sum()),
                      shared_reconstruction_accepted=False,review_required=True)
        save_parts(out/'joined_candidate.npz',[joined]);overlays([joined],images,K,baseline,out)
    else:overlays(parts,images,K,baseline,out)
    if fingerprints({**case['inputs'],'fragments':fragment_path})!=before:raise ValueError('Inputs changed')
    write_json(out/'report.json',report);return report


def overlays(parts,images,K,baseline,out):
    for side,image in images.items():
        bgr=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        for part in parts:
            xy=project(part['points_m'],K,baseline if side=='right' else 0.)
            shape=part['geometry_inferred'];dep=part['depth_inferred']
            for i in range(len(xy)-1):
                color=(0,165,255) if shape[i] or shape[i+1] else ((210,65,210) if dep[i] or dep[i+1] else (50,220,80))
                cv2.line(bgr,tuple(np.rint(xy[i]).astype(int)),tuple(np.rint(xy[i+1]).astype(int)),color,2,cv2.LINE_AA)
        cv2.rectangle(bgr,(0,0),(image.shape[1],48),(20,20,20),-1)
        for j,text in enumerate(('PROPOSAL: green=estimated depth support; pink=inferred depth',
                                 'orange=inferred hidden shape; NOT ground truth / DO NOT TRAIN')):
            cv2.putText(bgr,text,(8,18+j*21),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1)
        cv2.imwrite(str(Path(out)/(side+'_proposal.png')),bgr)
