#!/usr/bin/env python3
"""Standalone experimental reconstruction; legacy scripts/settings are untouched."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import cv2
import networkx as nx
import numpy as np
import reconstruct_thread_from_files as base
from occlusion_aware_thread_joining import JoinSettings, propose_joins, ordered_path, mask_at


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('left','right','left-mask','right-mask','depth-npy','K','left-occluder','out'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--right-occluder')
    p.add_argument('--baseline-m',type=float,required=True)
    p.add_argument('--samples',type=int,default=200)
    p.add_argument('--segment-min-skeleton-px',type=int,default=30)
    p.add_argument('--max-gap-px',type=float,default=240)
    p.add_argument('--max-gap-m',type=float,default=.04)
    p.add_argument('--min-occluder-fraction',type=float,default=.55)
    p.add_argument('--min-direction-cosine',type=float,default=.25)
    p.add_argument('--ambiguity-margin',type=float,default=.1)
    p.add_argument('--min-visible-coverage',type=float,default=.85)
    p.add_argument('--min-valid-visible-depth',type=float,default=.70)
    p.add_argument('--min-visible-right-fraction',type=float,default=.50)
    a=p.parse_args()
    out=Path(a.out)
    # Avoid accidental replacement of an existing successful experiment.
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise ValueError('Use a new empty output directory; existing artifacts are never overwritten')
    report={'status':'failed','algorithm':'experimental_occlusion_aware_stereo_thread_v1','fallbacks':False,
            'inferred_geometry_is_ground_truth':False,'inputs':vars(a)}
    try:
        if a.samples<2 or a.segment_min_skeleton_px<2: raise ValueError('Invalid sampling/component count')
        for v in (a.min_visible_coverage,a.min_valid_visible_depth,a.min_visible_right_fraction):
            if not 0<=v<=1: raise ValueError('Invalid validation fraction')
        left,right=base.load_rgb(a.left),base.load_rgb(a.right)
        if left.shape!=right.shape: raise ValueError('Stereo image dimensions differ')
        shape=left.shape[:2]
        lm,rm=base.load_mask(a.left_mask,shape),base.load_mask(a.right_mask,shape)
        lo=base.load_mask(a.left_occluder,shape)
        ro=None if a.right_occluder is None else base.load_mask(a.right_occluder,shape)
        depth=np.squeeze(np.load(a.depth_npy)).astype(float)
        if depth.shape!=shape: raise ValueError('Depth dimensions differ from image')
        K=base.load_intrinsics(a.K)
        settings=JoinSettings(max_gap_px=a.max_gap_px,max_gap_m=a.max_gap_m,
            min_occluder_fraction=a.min_occluder_fraction,min_direction_cosine=a.min_direction_cosine,
            ambiguity_margin=a.ambiguity_margin)
        settings.validate(); report['settings']=asdict(settings)
        skel,graph,_,components,discarded=base.skeleton_graph(lm,0,a.segment_min_skeleton_px)
        # Save every observed fragment even if assembly is rejected.
        fragments={f'fragment_{i:03d}_rc':np.array(sorted(nodes),int) for i,nodes in enumerate(nx.connected_components(graph))}
        np.savez(out/'observed_fragments.npz',**fragments)
        joins,candidates=propose_joins(graph,lm,rm,depth,K,a.baseline_m,lo,ro,settings)
        report.update(component_count_before=components,component_count_after=nx.number_connected_components(graph),
            discarded_short_components=discarded,joins=joins,candidates=candidates,
            right_occluder_supplied=ro is not None)
        image=cv2.cvtColor(left,cv2.COLOR_RGB2BGR)
        tint=image.copy(); tint[lo]=(150,150,150); image=cv2.addWeighted(image,.75,tint,.25,0)
        image[lm]=(0,200,0)
        for x,y,e in graph.edges(data=True):
            if e.get('synthetic'):
                cv2.polylines(image,[np.rint(e['curve_rc'][:,::-1]).astype('int32')],False,(0,165,255),2)
        cv2.imwrite(str(out/'join_diagnostic.png'),image)
        path,hidden=ordered_path(graph)
        # Uniform arclength sample provenance alongside the curve.
        distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
        targets=np.linspace(0,distance[-1],a.samples)
        sample=base.resample_path(path,a.samples)
        inferred=np.interp(targets,distance,hidden.astype(float))>1e-6
        # Classify by inferred edge provenance, not whether the gap happens to overlap an unrelated mask.
        observed=~inferred
        raw,support=base.sample_depth(depth,lm,sample,3,settings.min_depth_m,settings.max_depth_m)
        raw[inferred]=np.nan; support[inferred]=0
        cleaned,accepted,depth_metrics=base.clean_depth(raw,9,21)
        points=base.backproject(sample,cleaned,K)
        rd,rxy=base.right_mask_distances(sample,cleaned,rm,K[0,0],a.baseline_m)
        valid_fraction=float(np.mean(np.isfinite(raw[observed]))) if observed.any() else 0.
        right_fraction=float(np.mean(rd[observed]<=settings.right_tolerance_px)) if observed.any() else 0.
        raster=np.zeros(shape,'uint8')
        cv2.polylines(raster,[np.rint(sample[:,::-1]).astype('int32')],False,1,1)
        distances=cv2.distanceTransform(1-raster,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        coverage=float(np.mean(distances[skel]<=settings.right_tolerance_px))
        report['metrics']={'visible_skeleton_coverage':coverage,'valid_visible_depth_fraction':valid_fraction,
            'visible_right_mask_fraction':right_fraction,'inferred_sample_count':int(inferred.sum()),
            'sample_count':len(sample),**depth_metrics}
        failures=[]
        if coverage<a.min_visible_coverage: failures.append('visible_fragments_not_fully_represented')
        if valid_fraction<a.min_valid_visible_depth: failures.append('insufficient_visible_depth')
        if right_fraction<a.min_visible_right_fraction: failures.append('visible_stereo_projection_failed')
        base.save_overlays(out,left,right,sample,rxy,settings.right_tolerance_px,rd)
        if failures: raise RuntimeError('; '.join(failures))
        reliability=support*np.exp(-rd/settings.right_tolerance_px)*accepted
        reliability[inferred]=0.
        np.savez(out/'thread_samples.npz',u=np.linspace(0,1,len(points)),points=points*1000,
            points_m=points,pixel_xy=sample[:,::-1],depth_m=cleaned,reliability=reliability,
            inferred=inferred,observed=observed,algorithm=np.array(report['algorithm']))
        report['status']='passed'
        print(json.dumps(report['metrics'],indent=2))
    except Exception as e:
        report['error']=str(e)
        raise
    finally:
        (out/'quality_report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
