"""One fixed needle template plus rigid pose; attachment is a separate task."""
from pathlib import Path
from copy import deepcopy
import argparse,json,os,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'scene_sources')]
from fixed_needle_fitter import (build_catalog,selection_messages,validate_selection,load_npz,render,inputs,save,sha,CATALOG)
from rigid_needle_contract import audit as rigid_audit
from openai_scene_backend import OpenAISceneModel
from needle_thread_connection_tool import prepare_connected_geometry,decode_attachment_response
from newton_scene_adapter import CAMERA_TO_NEWTON,transform
from thread_visibility_repairs_v5 import load_parts

def construct_attachment(case,summary,row,out,decision):
    parts=load_parts(summary['candidate']['fragment_path']);pair=row['report']['connection']
    part=parts[pair['fragment']];needle=load_npz(row['needle_path'])
    if not rigid_audit(needle,CATALOG,require_pose=True)['usable']:raise ValueError('Attachment requires a verified rigid needle')
    inferred=part['geometry_inferred']|part['depth_inferred'];p=part['points_m'];n=needle['points_m']
    K=np.loadtxt(case['inputs']['K'],delimiter=',')
    geometry=dict(points_camera_m=p,points_newton_m=transform(p,CAMERA_TO_NEWTON),
        needle_points_camera_m=n,needle_points_newton_m=transform(n,CAMERA_TO_NEWTON),
        inferred=inferred,needle_inferred=needle['inferred'],observed=~inferred,
        reliability=(~inferred).astype(float),T_newton_from_camera=CAMERA_TO_NEWTON,K=K)
    selected=dict(pair,inferred_connection_authorized=True,specialist_asserted_physical_swage=True,
        numerical_checks_passed=True,visual_evidence_ids=['image.left','image.right'],
        uncertainty_warnings=['nominal_needle_size_hypothesis','occluded_needle_endpoint_inferred'])
    connected,metadata=prepare_connected_geometry(geometry,selected,case['baseline_m'])
    path=Path(out)/'connected_fragment.npz';np.savez_compressed(path,**connected)
    t=p[pair['thread_endpoint_index']]
    from geometry_repairs import project
    endpoint=dict(points_camera_m=t.tolist(),left_xy=project(t[None],K)[0].tolist(),
        right_xy=project(t[None],K,case['baseline_m'])[0].tolist())
    return dict(status='geometric_attachment_proposal',selection=selected,metadata=metadata,
        artifact=str(path.resolve()),geometry_constructed=True,source_geometry_sha256=sha(summary['candidate']['fragment_path']),
        source_thread_endpoint=endpoint,semantic_anchor=endpoint,decision=decision,semantic_decision=decision,
        source_image_hashes={s:sha(case['inputs'][s]) for s in ['left','right']},source_needle_hash=sha(row['needle_path']),
        hypothesis_not_ground_truth=True,physics_validated=False,training_eligible=False)

def geometry_choice(catalog):
    rows=catalog['candidates']
    good=[r for r in rows if r['report']['geometry_eligible']]
    if good:return max(good,key=lambda r:r['report']['score']),True
    # Preserve a plausible *rigid* estimate as provisional, never a deformed fallback.
    approximate=[r for r in rows if min(v['observed_coverage'] for v in r['report']['views'].values())>=.5
                 and r['report']['background_fraction_both']<=.5]
    return (max(approximate,key=lambda r:r['report']['score']),False) if approximate else (None,False)

def install_needle(summary,row,source_path,verified):
    out=deepcopy(summary);previous=(out.get('needle_provenance') or {}).get('path',source_path)
    if not previous or sha(previous)!=sha(row['needle_path']):out['attachment']=None
    out['needle_provenance']=dict(path=row['needle_path'],sha256=sha(row['needle_path']),modified=True,
        derived_from=source_path,original_preserved=True,nominal_size_hypothesis=True,
        representation='rigid_catalog_arc_v2',rigid_contract=row['report']['rigid_contract'],
        stereo_fit_verified=bool(verified),provisional=not verified)
    return out

def quarantine_legacy(summary,source_path):
    out=deepcopy(summary);out['attachment']=None
    out['needle_provenance']=dict(path=None,original_preserved=True,legacy_reference_path=source_path,
        representation='no_usable_rigid_needle',usable=False,reason='Legacy/free-radius result is archived, not certified as the fixed-size selection')
    return out

def run(case,needle_path,summary,out,method='agent',backend_factory=None,catalog=None,allow_model=True,require_visual_promotion=False):
    needle_path=str(needle_path) if needle_path else None
    out=Path(out);out.mkdir(parents=True,exist_ok=False);before=deepcopy(summary)
    save(out/'source_summary.json',before);started=time.perf_counter()
    original=load_npz(needle_path);original_audit=rigid_audit(original,CATALOG,require_pose=True)
    selected=deepcopy(before) if original_audit['usable'] else quarantine_legacy(before,needle_path)
    visual_fallback=deepcopy(selected)
    result=dict(controller=method,promoted=False,model_calls=0,tool_calls=1,source_summary_preserved=True,
        source_needle_audit=original_audit,independent_accuracy_measured=False,physics_validated=False)
    source_fragment=(summary.get('candidate') or {}).get('fragment_path')
    fragment_hash=sha(source_fragment) if source_fragment else None
    original_needle_hash=sha(needle_path) if needle_path and Path(needle_path).is_file() else None
    backend=None;parts=[];row=None
    try:
        parts=load_parts(source_fragment) if source_fragment else []
        catalog=catalog or build_catalog(case,needle_path,parts,out/'catalog')
        from rigid_endpoint_candidates import augment
        if 'endpoint_pose_refinement' not in catalog:
            catalog=augment(case,parts,catalog,out/'endpoint_pose_catalog')
        save(out/'catalog_reference.json',dict(path=catalog['catalog_image'],state_version=catalog['state_version']))
        row,verified=geometry_choice(catalog)
        if row is not None:
            selected=install_needle(selected,row,needle_path,verified)
            result.update(status='rigid_needle_selected_attachment_separate' if verified else 'provisional_rigid_needle_selected',
                promoted=True,selected_candidate_id=row['candidate_id'],radius_mm=row['radius_mm'],
                span_degrees=row['span_degrees'],geometry_verified=verified,attachment_eligible=row['report']['attachment_eligible'],
                decision_source='shared_deterministic_rigid_template_fit')
        else:result['status']='no_supported_rigid_needle_legacy_reference_only'
        if method=='agent' and allow_model and row is not None and verified:
            messages=selection_messages(case,catalog);save(out/'messages.json',messages)
            backend=backend_factory(out/'model_requests') if backend_factory else OpenAISceneModel(out/'model_requests',max_calls=1)
            result['model_calls']=1
            raw=backend(messages);(out/'raw_response.txt').write_text(raw+'\n')
            chosen=validate_selection(raw,catalog)
            if chosen is not None:
                decision=json.loads(decode_attachment_response(raw)[0]);selected=install_needle(selected,chosen,needle_path,True)
                if decision['physical_swage_identified']:
                    selected['attachment']=construct_attachment(case,selected,chosen,out,decision)
                result.update(status='VLM_rigid_needle_selected',selected_candidate_id=chosen['candidate_id'],
                    radius_mm=chosen['radius_mm'],span_degrees=chosen['span_degrees'],decision_source='actual_VLM',
                    attachment_constructed=bool(selected.get('attachment')),decision=decision)
                row=chosen
            else:result['status']='VLM_kept_verified_rigid_baseline'
        elif method=='agent' and not allow_model:result['VLM_status']='not_run_no_new_API_authorization'
    except Exception as error:
        # A failed model never resurrects an unphysical legacy free-radius fit.
        result.update(status='rigid_estimate_preserved_after_failure' if row else 'rigid_fit_failed_legacy_reference_only',
            error=f'{type(error).__name__}: {error}',framework_failure=True)
    if require_visual_promotion and result.get('decision_source')!='actual_VLM':
        selected=visual_fallback
        result.update(promoted=False,status='unaccepted_mask_candidate_not_promoted',actual_VLM_selection_required=True)
    path=selected.get('needle_provenance',{}).get('path')
    final_audit=rigid_audit(load_npz(path),CATALOG,require_pose=True)
    if path and not final_audit['usable']:raise ValueError('Final needle violates rigid-template export contract')
    if path:
        images,_,_,_,K,_,b=inputs(case);a=load_npz(path)
        render(images,a['points_m'],parts,K,b,a['inferred'],None,
            ('FIXED RIGID NEEDLE' if result.get('geometry_verified') else 'PROVISIONAL RIGID NEEDLE')+' (attachment separate)',out/'selected.jpg')
    if source_fragment and sha(source_fragment)!=fragment_hash:raise ValueError('Protected thread reconstruction changed')
    if original_needle_hash and sha(needle_path)!=original_needle_hash:raise ValueError('Protected baseline needle changed')
    result.update(selected_path=path,rigid_contract=final_audit,runtime_seconds=time.perf_counter()-started,
        estimated_API_cost_usd=getattr(backend,'estimated_api_cost_usd',0.),usage=getattr(backend,'last_usage',None),
        cost_is_estimate_not_invoice=True,legacy_baseline_is_reference_only=not original_audit['usable'])
    selected['needle_size_refinement']=result
    save(out/'summary.json',selected);save(out/'result.json',result)
    return selected,result

def enforce_export_contract(folder):
    """A failed optional stage cannot leak its old flexible-size seed to export."""
    import shutil
    folder=Path(folder);audits={}
    for method in ['heuristic','agent']:
        source=folder/method/'summary.json'
        if not source.exists():continue
        summary=json.loads(source.read_text());context=folder.parent/'export_context'/method/folder.name
        context.mkdir(parents=True,exist_ok=True);canonical=context/'common_needle.npz'
        needle=summary.get('needle_provenance',{}).get('path',str(canonical))
        audit=rigid_audit(load_npz(needle),CATALOG,require_pose=True);audits[method]=audit
        if not audit['usable']:
            backup=source.with_name('pre_rigid_export_summary.json')
            if not backup.exists():save(backup,summary)
            summary=quarantine_legacy(summary,needle);summary['rigid_export_contract']=audit
            save(source,summary)
            if canonical.exists():
                archive=context/'legacy_common_needle.npz'
                if not archive.exists():canonical.rename(archive)
                else:
                    # Preserve a second distinct legacy copy rather than overwrite it.
                    second=context/('legacy_common_needle_'+sha(canonical)[:12]+'.npz')
                    if not second.exists():canonical.rename(second)
                    else:canonical.unlink()
        else:
            if canonical.exists() and sha(canonical)!=sha(needle):
                archive=context/('legacy_common_needle_'+sha(canonical)[:12]+'.npz')
                if not archive.exists():shutil.copy2(canonical,archive)
            if Path(needle).resolve()!=canonical.resolve():shutil.copy2(needle,canonical)
    return audits

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--audit-export',action='store_true');args=parser.parse_args()
    if args.audit_export:
        save(args.scene/'rigid_export_audit.json',enforce_export_contract(args.scene));return
    key=sys.stdin.readline().strip();assert key;os.environ['OPENAI_API_KEY']=key;del key
    from contract import api_attempts
    folder=args.scene
    for method in ['heuristic','agent']:
        path=folder/method/'summary.json';summary=json.loads(path.read_text())
        context=folder.parent/'export_context'/method/folder.name
        casepath=Path(summary.get('selected_case_path',context/'case.json'));case=json.loads(casepath.read_text())
        needle=summary.get('needle_provenance',{}).get('path',str(context/'common_needle.npz'))
        out=folder/'needle_size_refinement'/method
        if (out/'summary.json').exists():selected=json.loads((out/'summary.json').read_text())
        else:
            from needle_tool_handoff import prepare
            case,pending_visual,casepath=prepare(case,needle,folder,method,folder/('needle_tool_handoff_'+method))
            selected,result=run(case,needle,summary,out,method,allow_model=len(api_attempts(folder))<23,require_visual_promotion=pending_visual)
            if not pending_visual or result.get('decision_source')=='actual_VLM':selected['selected_case_path']=casepath
        backup=path.with_name('pre_needle_size_summary.json')
        if not backup.exists():save(backup,summary)
        save(path,selected)
        import shutil
        canonical=context/'common_needle.npz';archive=context/'legacy_common_needle.npz'
        if canonical.exists() and not archive.exists():shutil.copy2(canonical,archive)
        selected_needle=selected.get('needle_provenance',{}).get('path')
        if selected_needle:
            if not rigid_audit(load_npz(selected_needle),CATALOG,require_pose=True)['usable']:raise ValueError('Refuse nonrigid export')
            if Path(selected_needle).resolve()!=canonical.resolve():shutil.copy2(selected_needle,canonical)
        elif canonical.exists():canonical.unlink() # Archived above; never export invalid baseline as a successful needle.
    save(folder/'rigid_export_audit.json',enforce_export_contract(folder))

if __name__=='__main__':main()
