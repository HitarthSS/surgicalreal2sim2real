"""Use the completed tool stage's own masks, then retry bounded connections.

No model calls, synthetic occluders, mask dilation beyond the existing connector,
or relaxed geometry gates. Run after tools, before exports, in a fresh revision.
"""
from pathlib import Path
from copy import deepcopy
import argparse,json,sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'scene_sources')]
from reconstruction_tools import sha256,fingerprints
from complex_case_diagnostics import write_json
from run_shared_refiner_v6 import conservative_occluder
from repair_aware_scene_tools import RepairAwareSceneTools
from thread_visibility_repairs_v5 import load_inputs,load_parts,join_proposals,RepairLimits,project,in_view
from refinement_policy import scene_promotes
from thread_gap_inventory_v7 import build_gap_inventory
from needle_attachment_inventory_v7 import build_attachment_inventory
from run_scene_orchestrator_v7 import SceneOrchestrator,EndpointLedger,Budget,NeedleAttachmentSpecialist
from repair_loop_policy import reconstruction_completion


def read(p):return json.loads(Path(p).read_text())


def verified_tool_masks(case,folder,method,tool_results):
    """Bind actual SAM3 masks to their image/request hashes, independently of pose acceptance.

    Never borrow agent masks for the heuristic. A poor pose does not erase a source-bound 2D observation. Unrelated,
    malformed, or semantically incompatible mask proposals remain excluded.
    """
    selected=tool_results.get('tools',{}).get(method,{})
    search=Path(folder)/('tool_agent' if method=='agent' else 'tools')
    masks={s:[] for s in ('left','right')};receipts=[];rejected=[]
    configs=list(search.rglob('fit_config.json')) if search.exists() else []
    for name,row in selected.items():
        if not row.get('mesh'):continue
        mesh=Path(row['mesh']).resolve()
        bound=row.get('bound_occlusion_evidence')
        if bound:
            try:
                if bound['schema']!='bound_tool_occlusion_evidence_v1' or bound['controller'] not in ('heuristic','agent'):
                    raise ValueError('Unknown bound mask receipt')
                if method=='heuristic' and bound['controller']!='heuristic':raise ValueError('Agent evidence cannot enter heuristic baseline')
                if bound['mesh_sha256']!=sha256(mesh):raise ValueError('Bound mask receipt belongs to a different fitted mesh')
                arrays={}
                for side,item in bound['masks'].items():
                    if bound['image_sha256'][side]!=sha256(case['inputs'][side]) or item['sha256']!=sha256(item['path']):
                        raise ValueError('Bound mask input/hash mismatch')
                    mask=cv2.imread(item['path'],0);image=cv2.imread(case['inputs'][side],0)
                    if mask is None or mask.shape!=image.shape or not (mask>0).any():raise ValueError('Invalid bound mask')
                    arrays[side]=mask>0
                if set(arrays)!={'left','right'}:raise ValueError('Bound masks require both stereo views')
                from geometry_repairs import tool_mask_diagnostics
                thread={side:cv2.imread(case['inputs']['thread_'+side+'_mask'],0)>0 for side in ('left','right')}
                sanity=tool_mask_diagnostics(arrays,thread)
                if sanity['requires_resegmentation']:raise ValueError('Observed tool mask requires identity repair')
                for side,mask in arrays.items():masks[side].append(mask)
                receipts.append(dict(instance=name,bound_receipt=bound,source='selected own-method actual SAM3 masks',
                    pose_accepted=bool(row.get('accepted_tool_hypothesis')),pose_acceptance_not_implied=True,mask_diagnostics=sanity))
                continue
            except (ValueError,KeyError,FileNotFoundError) as error:
                rejected.append(dict(instance=name,reason=str(error)))
        for cfg_path in configs:
            cfg=read(cfg_path)
            if cfg.get('instance_id')!=name or not mesh.is_relative_to(Path(cfg['out']).resolve()):continue
            try:
                paths={s:Path(cfg['masks'][s]).resolve() for s in ('left','right')}
                report_paths={p.parent/'report.json' for p in paths.values()}
                if len(report_paths)!=1:raise ValueError('Stereo mask reports differ')
                rp=next(iter(report_paths));report=read(rp)
                if report.get('actual_model_producer') is not True or report.get('full_frame') is not True or report.get('image_crop') is not None:
                    raise ValueError('Full-frame actual SAM3 receipt required')
                arrays={}
                for side,path in paths.items():
                    if report['source_fingerprints'][side]['sha256']!=sha256(case['inputs'][side]):
                        raise ValueError('Tool mask belongs to different stereo input')
                    if sha256(path)!=report['views'][side][name]['mask_sha256']:
                        raise ValueError('Tool mask hash changed')
                    mask=cv2.imread(str(path),0);image=cv2.imread(case['inputs'][side],0)
                    if mask is None or mask.shape!=image.shape or not (mask>0).any():raise ValueError('Invalid tool mask')
                    arrays[side]=mask>0
                for side,mask in arrays.items():masks[side].append(mask)
                receipts.append(dict(instance=name,mesh=str(mesh),mesh_sha256=sha256(mesh),
                    config=str(cfg_path),config_sha256=sha256(cfg_path),report=str(rp),report_sha256=sha256(rp),
                    masks={s:dict(path=str(p),sha256=sha256(p)) for s,p in paths.items()},
                    source='selected accepted tool fit; actual SAM3 masks; not independent occlusion truth'))
            except (ValueError,KeyError,FileNotFoundError) as error:
                rejected.append(dict(instance=name,config=str(cfg_path),reason=str(error)))
    # Shared initial SAM3 observations are also evidence when no fitted mesh exists.
    # They remain hypotheses and never promote a pose or enlarge an occluder.
    marker=Path(folder)/'ready.json'
    if not marker.exists():marker=Path(folder).parent/'ready'/(Path(folder).name+'.json')
    cache_value=read(marker).get('cache') if marker.exists() else None
    if cache_value:
        from run_shared_refiner_v6 import SharedRepairToolsV6
        from geometry_repairs import tool_mask_diagnostics
        try:
            casepath=Path(folder)/'case.json'
            pipeline=Path(folder).parent/'pipeline-tools'
            cache_tools=SharedRepairToolsV6(casepath,pipeline,cache=Path(cache_value))
            for card in ('occluder.sam3.grasper','occluder.sam3.instrument'):
                paths,report=cache_tools.read_cache(card)
                arrays={side:cv2.imread(path,0)>0 for side,path in paths.items()}
                thread={side:cv2.imread(case['inputs']['thread_'+side+'_mask'],0)>0 for side in ('left','right')}
                sanity=tool_mask_diagnostics(arrays,thread)
                if sanity['requires_resegmentation']:continue
                if any(mask.sum()>mask.size*.60 for mask in arrays.values()):continue
                for side,mask in arrays.items():masks[side].append(mask)
                receipts.append(dict(source='original shared actual SAM3 masks; independent of fitted pose acceptance',card=card,
                    source_report_sha256=sha256(Path(cache_value)/case['case_id']/card/'report.json'),
                    masks={side:dict(path=path,sha256=sha256(path)) for side,path in paths.items()},
                    mask_diagnostics=sanity,pose_acceptance_not_implied=True,occlusion_is_hypothesis=True))
                break
        except (ValueError,KeyError,FileNotFoundError,RuntimeError) as error:
            rejected.append(dict(source='shared SAM3 observations',reason=str(error)))
    return masks,receipts,rejected


def install_occlusion(tools,masks,out):
    """Union the verified new instances with retained old evidence; exclude metal/thread labels."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    _,images,_,old,_,_,_=load_inputs(tools.case,tools.pipeline)
    changed=False;paths={};audits={}
    for side in ('left','right'):
        raw=old[side].copy()
        for mask in masks[side]:raw|=mask
        needle=cv2.imread(tools.original['inputs']['needle_'+side+'_mask'],0)
        thread=cv2.imread(tools.case['inputs']['thread_'+side+'_mask'],0)
        mask,audit=conservative_occluder(raw,needle,thread)
        if int(mask.sum())>mask.size*.60:raise ValueError('Accumulated occlusion exceeds full-frame area bound')
        changed|=not np.array_equal(mask,old[side]);audits[side]=audit
        path=out/(side+'_occluder.png');cv2.imwrite(str(path),mask.astype('uint8')*255)
        paths[side]=str(path.resolve())
        image=cv2.cvtColor(images[side],cv2.COLOR_RGB2BGR)
        image[mask]=(image[mask]*.65+np.array([190,65,220])*.35).astype('uint8')
        cv2.imwrite(str(out/(side+'_feedback.png')),image)
    if changed:
        tools.case['inputs'].update({s+'_occluder':p for s,p in paths.items()})
        tools.version+=1;tools.working_path=out/'derived_case.json'
        write_json(tools.working_path,tools.case)
        tools.register_derived({str(p):str(p) for p in out.iterdir() if p.is_file()})
    return dict(changed=changed,views=audits,original_pixels_modified=False,
                prior_occlusion_preserved=True,semantic_truth_claimed=False)


def unique_connection(tools,candidate,reserved,attempted):
    if not candidate or not candidate['report'].get('numerical_checks_passed'):return None
    cards,_=tools.menu(candidate,attempted)
    parts=load_parts(candidate['fragment_path']);limits=RepairLimits()
    rows=join_proposals(tools.case,tools.pipeline,parts)
    good=[]
    for aid,card in cards.items():
        if card['operation']!='smooth_connection':continue
        row=next(r for r in rows if r['proposal_id']==card['parameters']['proposal_id'])
        if row['failures']:continue
        _,_,masks,_,_,K,baseline=load_inputs(tools.case,tools.pipeline)
        outside=(~in_view(project(row['_curve'],K),masks['left'].shape)) & (~in_view(project(row['_curve'],K,baseline),masks['right'].shape))
        if outside[5:-5].any():continue
        from geometry_repairs import inferred_connection_guard
        if not inferred_connection_guard(row['_curve'],[p['points_m'] for p in parts])['allowed']:continue
        ends=set(row['endpoints'])
        competitors=[r for r in rows if not r['failures'] and set(r['endpoints'])!=ends and ends.intersection(r['endpoints'])]
        if competitors and min(r['score'] for r in competitors)-row['score']<limits.ambiguity_score_margin:continue
        endpoints=[]
        for eid in row['endpoints']:
            fragment,index=eid.split('.');endpoints.append(parts[int(fragment[1:])]['points_m'][0 if index=='e0' else -1])
        if any(np.linalg.norm(p-q)<1e-8 for p in endpoints for q in reserved):continue
        good.append((row['score'],aid,card))
    return min(good,key=lambda x:(x[0],x[1]))[1:] if good else None


def finalize(case_path,pipeline,needle,summary,folder,method,out,max_tool_calls=4):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    tools=RepairAwareSceneTools(case_path,pipeline,needle_path=needle)
    before=deepcopy(summary);candidate=deepcopy(summary.get('candidate'))
    if candidate:tools.geometry_version=sha256(candidate['fragment_path'])
    protected=dict(tools.paths)
    if candidate:protected.update(candidate['protected_paths'])
    original_hashes=fingerprints(protected)
    events=[];calls=0;attempted=set();old_attachment=deepcopy(summary.get('attachment'))
    reserved=[np.asarray(old_attachment['source_thread_endpoint']['points_camera_m'])] if old_attachment and old_attachment.get('source_thread_endpoint') else []
    selected=Path(folder)/'tools/complete_tools.json'
    mask_lists,receipts,rejected=verified_tool_masks(tools.case,folder,method,read(selected) if selected.exists() else {})
    if any(mask_lists.values()) and calls<max_tool_calls and candidate and candidate['report']['fragment_count']>1:
        calls+=1;operation=install_occlusion(tools,mask_lists,out/'tool_occlusion')
        events.append(dict(operation='reuse_selected_tool_masks',result=operation,receipts=receipts,rejected_sources=rejected))
    # Reserve the last slot for rebuilding a previously established attachment.
    reserve=int(bool(old_attachment and old_attachment.get('status') in ('geometric_attachment_proposal','attachment_identity_hypothesis')))
    while calls<max_tool_calls-reserve and candidate and candidate['report']['fragment_count']>1:
        choice=unique_connection(tools,candidate,reserved,attempted)
        if choice is None:break
        aid,card=choice;attempted.add(tools.action_key(card));calls+=1
        try:
            proposal,audit=tools.execute_repair(aid,card,candidate,out/f'connection_{calls:03d}',
                selection_evidence=dict(controller='shared_post_tool_geometry',evidence_ids=['image.left','image.right'],
                    reason='Unique admissible endpoint pair after provenance-checked tool-mask evidence; all original gates and crossing guards retained.'))
            promote,reason=scene_promotes(candidate,proposal)
            events.append(dict(operation=card['operation'],card=card,promoted=promote,reason=reason,
                failures=proposal['report']['failures'],audit=audit))
            if promote:candidate=proposal;tools.geometry_version=sha256(candidate['fragment_path'])
        except Exception as error:
            events.append(dict(operation=card['operation'],error=f'{type(error).__name__}: {error}',baseline_retained=True))
    blocked=[]
    if candidate and candidate['report']['fragment_count']>1:
        parts=load_parts(candidate['fragment_path'])
        blocked=[{k:v for k,v in row.items() if not k.startswith('_')} for row in join_proposals(tools.case,tools.pipeline,parts)]
        write_json(out/'remaining_pair_diagnostics.json',blocked)
    changed=(candidate or {}).get('fragment_path')!=(before.get('candidate') or {}).get('fragment_path')
    attachment=old_attachment;attachment_rebuilt=False
    if changed and reserve:
        inv=build_attachment_inventory(tools.case,pipeline,candidate,needle,out/'attachment_inventory')
        runner=object.__new__(SceneOrchestrator);runner.tools=tools;runner.needle=Path(needle);runner.out=out
        runner.attachment=old_attachment;runner.reserved=reserved;runner.controller=method
        runner.history=[];runner.snapshot=0;runner.feedback=None;runner.models=None
        runner.attachment_specialist=NeedleAttachmentSpecialist()
        runner.budget=Budget(max_model_calls=0,max_tool_calls=max_tool_calls,tool_calls=calls)
        reuse=runner.reusable_attachment(old_attachment,inv)
        if reuse:
            runner.attach(candidate,inv,reuse,construct=True);calls=runner.budget.tool_calls
            attachment=runner.attachment;attachment_rebuilt=True
        else:
            # Keep the original complete bundle; a stale endpoint is never exported.
            events.append(dict(operation='preserve_attachment',reason='Original swage endpoint cannot be uniquely rebound; geometry fallback'))
            candidate=deepcopy(before['candidate']);attachment=old_attachment;changed=False
    summary=deepcopy(before);summary['candidate']=candidate;summary['attachment']=attachment
    if changed:
        inventory=build_gap_inventory(tools.case,pipeline,candidate,out/'final_inventory')
        ledger=EndpointLedger();ledger.refresh(inventory)
        write_json(out/'endpoint_ledger.json',ledger.entries)
        summary['completion']=dict(all_current_endpoints_assessed=False,unassessed_count=len(ledger.pending()),
            source='fresh inventory; no fabricated model assessments')
        summary['endpoint_ledger_path']=str(out/'endpoint_ledger.json')
    summary['reconstruction_completion']=reconstruction_completion(candidate,summary.get('completion',{}).get('all_current_endpoints_assessed',False))
    if summary['reconstruction_completion']['partial_reconstruction']:summary['status']='partial_reconstruction'
    elif changed:summary['status']='connected_geometry_after_tool_evidence'
    summary['selected_case_path']=str(tools.working_path)
    summary['post_tool_connection_refinement']=dict(model_calls=0,tool_calls=calls,max_tool_calls=max_tool_calls,
        geometry_changed=changed,attachment_rebuilt=attachment_rebuilt,history=events,
        remaining_pair_diagnostics=str(out/'remaining_pair_diagnostics.json') if blocked else None,
        source_summary_preserved=True,independent_accuracy_measured=False,physics_validated=False)
    if fingerprints(protected)!=original_hashes:raise ValueError('Original input or reconstruction changed')
    write_json(out/'source_summary.json',before);write_json(out/'summary.json',summary)
    write_json(out/'protected_inputs.json',original_hashes)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--pipeline',type=Path,required=True)
    args=p.parse_args();folder=args.scene
    for method in ('heuristic','agent'):
        source=folder/method/'summary.json';summary=read(source)
        context=folder.parent/'export_context'/method/folder.name
        case=Path(summary.get('selected_case_path',context/'case.json'))
        needle_value=(summary.get('needle_provenance') or {}).get('path') or context/'common_needle.npz'
        needle=Path(needle_value) if Path(needle_value).is_file() else None
        out=folder/'connection_refinement'/method
        if (out/'summary.json').exists():result=read(out/'summary.json')
        else:result=finalize(case,args.pipeline,needle,summary,folder,method,out,
                max_tool_calls=max(0,10-summary.get('budget',{}).get('tool_calls',0)))
        # Fresh-run pipeline integration only: originals are retained before
        # updating its canonical selection. Diagnostic callers use finalize().
        backup=source.with_name('pre_connection_summary.json')
        if not backup.exists():write_json(backup,summary)
        prior=result.get('budget',{});extra=result['post_tool_connection_refinement']['tool_calls']
        result['budget']=dict(prior,max_tool_calls=10,tool_calls=prior.get('tool_calls',0)+extra,
                            pre_tool_scene_cap=6,post_tool_calls=extra)
        assert result['budget']['tool_calls']<=10
        write_json(source,result)
        write_json(context/'case.json',read(result['selected_case_path']))


if __name__=='__main__':main()
