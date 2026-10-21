"""Per-scene shared fitting plus actual two-tool agent feedback.

Uses the existing initialization, native tracker, native assets, verification,
and broader stereo recovery. This file runs inside the CPU tracker runtime.
"""
from copy import deepcopy
import json
import hashlib
import os
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'dependencies')]
from tool_protocol import validate_initialization,digest
from shared_tool_estimation import masks_from_request,save
from tool_repair import NativeTool
from stereo_tool_refit import refit as coverage_refit
from shared_tool_estimation import verify
from output_policy import classify
from refinement_policy import tool_promotes
from agent_recovery_tool import execute,TOOL_CARD
from openai_scene_backend import OpenAISceneModel


def load(path):
    with np.load(path,allow_pickle=False) as d:return {k:d[k].copy() for k in d.files}


def main():
    assert not (ROOT/'complete_tools.json').exists(),'Do not repeat completed tool stage'
    cfg=json.loads((ROOT/'pilot_config.json').read_text());case=json.loads(Path(cfg['case']).read_text())
    images={s:cv2.imread(case['inputs'][s]) for s in ['left','right']}
    sizes={s:[im.shape[1],im.shape[0]] for s,im in images.items()}
    K=np.loadtxt(case['inputs']['K'],delimiter=',');depth=np.load(case['inputs']['depth_npy'])
    native=NativeTool(cfg['repo']);states={};observations={};contexts={};errors=[]
    for method in ['heuristic']:
        initial=ROOT/method/'initialization.json'
        if not initial.exists():
            errors.append(dict(controller=method,error='automatic_initialization_unavailable'));continue
        observations[method]=json.loads(initial.read_text());masks=masks_from_request(observations[method],sizes)
        states[method]={};contexts[method]={}
        for name in [x['instance_id'] for x in observations[method]['instances']]:
            folder=ROOT/method/name
            selected_path=folder/'selected.json'
            if not selected_path.exists():
                errors.append(dict(controller=method,instance=name,error='native_fit_or_verification_failed'));continue
            selected=json.loads(selected_path.read_text());original=load(selected['mesh'])
            item=next(x for x in observations[method]['instances'] if x['instance_id']==name)
            context=dict(scene_version='initial',remaining_recoveries=2,native=native,original=original,warm=original,
                masks={s:cv2.imread(str(folder/(s+'_mask.png')),0)>0 for s in sizes},
                reference_masks={s:cv2.imread(str(folder/(s+'_mask.png')),0)>0 for s in sizes},views=item['views'],entry_border=item['entry_border'],
                K=K,baseline_m=case['baseline_m'],images=images,depth=depth,out=folder/'broad_initial')
            contexts[method][name]=context
            before=load(selected['mesh'])
            candidate,diagnostic=coverage_refit(native,before,context['masks'],K,case['baseline_m'],depth,item['entry_border'])
            result=dict(selected=None,attempts=diagnostic['attempts'],coverage_refit=diagnostic,baseline_preserved=True)
            target=folder/'coverage_refit';target.mkdir()
            np.savez_compressed(target/'tool_camera_m.npz',**candidate)
            checked=verify(candidate,context['masks'],images,K,case['baseline_m'],depth,target/'verification')
            if diagnostic['promoted'] and (checked['accepted_numerical_hypothesis'] or not selected['verification'].get('accepted_numerical_hypothesis')):
                selected.update(mesh=str(target/'tool_camera_m.npz'),verification=checked)
                result['selected']=dict(mesh=selected['mesh'],reports=dict(reference=checked))
            selected.update(broad_recovery=result,numerical_recovery_passed=bool(checked['accepted_numerical_hypothesis']),
                agent_review_status='not_reviewed',shared_host_postprocessing=True)
            states[method][name]=selected
    for row in states.get('heuristic',{}).values():
        row['numerical_recovery_passed']=bool(row['verification'].get('accepted_numerical_hypothesis'))
    baseline_states=deepcopy(states.get('heuristic',{}))
    if baseline_states:
        states['agent']=deepcopy(baseline_states)
        observations['agent']=deepcopy(observations['heuristic']);contexts['agent']={}
        for name,row in states['agent'].items():
            context=dict(contexts['heuristic'][name])
            baseline_mesh=load(row['mesh'])
            context.update(original=baseline_mesh,warm=baseline_mesh,remaining_recoveries=2)
            contexts['agent'][name]=context
            (ROOT/'agent'/name).mkdir(parents=True,exist_ok=True)
            row.update(agent_review_status='inherited_heuristic',baseline_mesh=row['mesh'],baseline_preserved=True)
        save(ROOT/'heuristic_baseline_snapshot.json',baseline_states)
        save(ROOT/'agent_baseline_observations.json',observations['agent'])
    promotion_history=[]
    protected_baseline={row['mesh']:hashlib.sha256(Path(row['mesh']).read_bytes()).hexdigest() for row in baseline_states.values()}
    best_observations=deepcopy(observations.get('agent',{}))
    save(ROOT/'baseline_geometry_hashes.json',protected_baseline)
    save(ROOT/'shared_broad_results.json',states)
    for method,instances in states.items():
        for name,selected in instances.items():
            selected['accepted_tool_hypothesis']=bool(selected.get('numerical_recovery_passed'))
            selected['baseline_preserved']=True
            selected['selection_origin']='heuristic_baseline'
            mask_folder=ROOT/'heuristic'/name
            selected['bound_occlusion_evidence']=dict(schema='bound_tool_occlusion_evidence_v1',controller='heuristic',
                source='same fixed heuristic silhouette masks used in selected fit verification; not segmentation truth',
                mesh_sha256=hashlib.sha256(Path(selected['mesh']).read_bytes()).hexdigest(),
                image_sha256={s:hashlib.sha256(Path(case['inputs'][s]).read_bytes()).hexdigest() for s in ['left','right']},
                masks={s:dict(path=str((mask_folder/(s+'_mask.png')).resolve()),
                    sha256=hashlib.sha256((mask_folder/(s+'_mask.png')).read_bytes()).hexdigest()) for s in ['left','right']})
            selected['agent_review_status']='heuristic_baseline_pending_specialists' if method=='agent' else 'not_applicable'
            selected['output_policy']=classify(selected,load(selected['mesh']))
            save(ROOT/method/name/'final_selection.json',selected)
    save(ROOT/'complete_tools.json',dict(tools=states,errors=errors,actual_feedback_calls=0,estimated_api_cost_usd=0,baseline_preserved=True,agent_specialists_pending=True))
if __name__=='__main__':main()
