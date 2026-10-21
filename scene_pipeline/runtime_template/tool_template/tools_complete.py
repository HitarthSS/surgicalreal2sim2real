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
                masks=masks[name],reference_masks=masks[name],views=item['views'],entry_border=item['entry_border'],
                K=K,baseline_m=case['baseline_m'],images=images,depth=depth,out=folder/'broad_initial')
            contexts[method][name]=context
            request=dict(scene_version='initial',action_id='reinitialize_stereo_tool',
                parameters=dict(instance_id=name,strategy='warm_start'),evidence_ids=['image.left','image.right'],
                reason='Shared broader pose/articulation registration after native automatic initialization.')
            result=dict(selected=None,attempts=[],baseline_reused=True) if cfg.get('baseline_is_final') else execute(request,context)
            if result['selected']:
                proposal=result['selected'];selected.update(mesh=proposal['mesh'],verification=proposal['reports']['reference'])
                context['warm']=load(proposal['mesh'])
            selected.update(broad_recovery=result,numerical_recovery_passed=bool(result['selected']),
                agent_review_status='not_reviewed',shared_host_postprocessing=True)
            if method=='heuristic' and not result['selected'] and not cfg.get('baseline_is_final'):
                context.update(scene_version='heuristic_retry',out=folder/'broad_stereo_retry')
                request.update(scene_version='heuristic_retry',parameters=dict(instance_id=name,strategy='multi_start'))
                second=execute(request,context)
                if second['selected']:
                    proposal=second['selected'];selected.update(mesh=proposal['mesh'],verification=proposal['reports']['reference'],numerical_recovery_passed=True)
                selected['stereo_retry']=second
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
    credential=sys.stdin.readline().strip()
    if not credential:raise ValueError('Missing memory-only feedback credential')
    os.environ['OPENAI_API_KEY']=credential;del credential
    history=[];backend=None
    if states.get('agent'):
        backend=OpenAISceneModel(ROOT/'tool_feedback_requests',max_calls=4)
        try:
            for turn in range(4):
                evidence=[('image.'+s,case['inputs'][s]) for s in images]
                for name,row in states['agent'].items():
                    evidence += [(name+'.'+s,p) for s,p in row['verification']['feedback_images'].items()]
                compact={name:dict(verification=row['verification']['views'],
                    numerical_pass=row.get('numerical_recovery_passed',False),
                    remaining_recoveries=contexts['agent'][name]['remaining_recoveries'],
                    observations=next(x for x in observations['agent']['instances'] if x['instance_id']==name))
                    for name,row in states['agent'].items()}
                state=dict(case_id=case['case_id'],tools=compact,remaining_model_calls=4-turn,
                    remaining_repair_calls={n:c['remaining_recoveries'] for n,c in contexts['agent'].items()},
                    previous_actions=[x['decision']['parameters']['tools'] for x in history[-2:]],evidence_ids=[e for e,_ in evidence],broad_tool_card=TOOL_CARD,promotion_criteria='Verified fixed-reference stereo improvement; no per-view IoU, depth-error or depth-support regression. Preserve baseline on failure or budget exhaustion.',
                    available_operations=['keep','keep_approximate','defer'] if turn==3 else ['keep','keep_approximate','defer','reinitialize_stereo_tool','revise_tool_observations'])
                version='t'+digest(state)[:12];state['scene_version']=version
                prompt='''Inspect BOTH full original stereo views and the actual fitted tool overlays. Review EACH available tool independently. Your starting geometry is the preserved FINAL heuristic reconstruction. Keep it unless a verified repair strictly improves fixed-reference stereo consistency; failed/no-op proposals never replace it.
Visual IDs tool_A=left entry and tool_B=right entry; never invent PSM identity. A numerical pass is not accuracy or visual acceptance.
The common fitter now supports broad pose reinitialization (90deg/component,50mm translation,45deg joint deltas within original physical limits), preserving dimensions. The 15deg restriction is gone.
Always preserve the best usable reconstructed geometry. Choose keep for visually supported geometry in BOTH views with numerical_pass=true. Choose keep_approximate for an imperfect or uncertain fit; poor overlap, hidden surfaces, or small alignment errors are not reasons to suppress reconstruction. Try a supported repair when budget permits, then retain the best estimate. No reconstruction is reserved for missing or structurally unusable geometry. Verification remains separate from availability.
For reinitialize_stereo_tool parameters={strategy:"multi_start"|"stereo_geometry"|"warm_start"}; select a useful DIFFERENT strategy if the current fit is wrong.
For revise_tool_observations parameters={views:{left:{polygon:[[x,y],...],jaw_tips_xy:[[x,y],[x,y]]},right:{polygon:[[x,y],...],jaw_tips_xy:[[x,y],[x,y]]}},reason:"..."}; original pixels x0..639/y0..479,3-16 polygon vertices, two visibly supported tips or [] for unavailable tip evidence, trace only metal and meet its observed entering border. This reruns the SAME broad shared fitter.
For keep/keep_approximate parameters={}. Legacy defer is interpreted as keep_approximate when usable geometry exists. You cannot change calibration, CAD dimensions or acceptance checks. At most two total recovery operations per tool, including shared initial broad fitting. No repair operations on final turn.
Return ONLY exact JSON fields scene_version,action_id="review_tools",parameters={tools:[{instance_id,operation,parameters,reason},...]},evidence_ids,reason. Include each available instance once, cite image.left,image.right and relevant tool overlays. Each reason<=400chars. Preserve uncertainty and inspect shaft as well as jaw alignment.
'''
                content=[dict(type='text',text=prompt+'\nSTATE:'+json.dumps(state,separators=(',',':')))]
                if len(content[0]['text'])>24000:raise ValueError('Tool feedback prompt overflow; stop, do not truncate evidence')
                for eid,path in evidence:content += [dict(type='text',text='EVIDENCE_ID='+eid),dict(type='image',url=path)]
                raw=backend([dict(role='user',content=content)]);decision=json.loads(raw)
                save(ROOT/f'tool_feedback_decision_{turn+1:02d}.json',decision)
                if set(decision)!={'scene_version','action_id','parameters','evidence_ids','reason'} or decision['scene_version']!=version or decision['action_id']!='review_tools':raise ValueError('Invalid tool review envelope')
                if not {'image.left','image.right'}<=set(decision['evidence_ids'])<=set(state['evidence_ids']):raise ValueError('Unregistered stereo evidence')
                if set(decision['parameters'])!={'tools'}:raise ValueError('Invalid review parameters')
                rows=decision['parameters']['tools'];names=[x['instance_id'] for x in rows]
                if len(names)!=len(set(names)) or set(names)!=set(states['agent']):raise ValueError('Review each available tool once')
                step=dict(decision=decision,results=[]);repaired=False
                for row in rows:
                    if set(row)!={'instance_id','operation','parameters','reason'}:raise ValueError('Invalid tool action fields')
                    name=row['instance_id'];op=row['operation'];selected=states['agent'][name];context=contexts['agent'][name]
                    if op not in state['available_operations']:raise ValueError('Unavailable tool operation')
                    if op in ['keep','keep_approximate','defer']:
                        if row['parameters']!={}:raise ValueError('Keep/defer takes no parameters')
                        if op=='keep' and not selected.get('numerical_recovery_passed'):raise ValueError('Cannot keep failed geometry')
                        selected['agent_review_status']='keep_approximate' if op=='defer' else op;continue
                    if context['remaining_recoveries']<1:raise ValueError('Tool recovery budget exhausted')
                    previous_observations=deepcopy(observations['agent'])
                    previous_views=deepcopy(context['views']);previous_masks={s:m.copy() for s,m in context['masks'].items()}
                    if op=='revise_tool_observations':
                        params=row['parameters']
                        if set(params)!={'views','reason'}:raise ValueError('Invalid observation revision')
                        revised=deepcopy(observations['agent']);item=next(x for x in revised['instances'] if x['instance_id']==name)
                        item.update(views=params['views'],reason=params['reason'])
                        revised=validate_initialization(revised,sizes);newmasks=masks_from_request(revised,sizes)
                        for side in images:
                            support=cv2.dilate(newmasks[name][side].astype('uint8'),np.ones((9,9),'uint8'))
                            for x,y in item['views'][side]['jaw_tips_xy']:
                                if not support[int(round(y)),int(round(x))]:raise ValueError('Revised jaw tip unsupported by visible tool mask')
                        context.update(masks=newmasks[name],views=item['views'])
                        observations['agent']=revised;save(ROOT/f'observation_revision_{turn+1:02d}_{name}.json',revised)
                        strategy='multi_start'
                    else:
                        if set(row['parameters'])!={'strategy'}:raise ValueError('Invalid recovery parameters')
                        strategy=row['parameters']['strategy']
                    context.update(scene_version=version,out=ROOT/'agent'/name/f'agent_recovery_{turn+1:02d}')
                    request=dict(scene_version=version,action_id='reinitialize_stereo_tool',parameters=dict(instance_id=name,strategy=strategy),
                        evidence_ids=['image.left','image.right','fit.left','fit.right'],reason=row['reason'])
                    result=execute(request,context);step['results'].append(dict(instance=name,result=result));repaired=True
                    if result['selected']:
                        proposal=result['selected']
                        promote,why=tool_promotes(selected['verification'],proposal['reports']['reference'])
                        promotion_history.append(dict(instance=name,proposal=proposal['mesh'],previous=selected['mesh'],promoted=promote,reason=why))
                        if promote:
                            selected.update(mesh=proposal['mesh'],verification=proposal['reports']['reference'],numerical_recovery_passed=True)
                            context['warm']=load(proposal['mesh'])
                            best_observations=deepcopy(observations['agent'])
                        else:
                            # Rejected mask changes may not become the next working reference.
                            context.update(masks=previous_masks,views=previous_views);observations['agent']=previous_observations

                    if not result['selected']:
                        context.update(masks=previous_masks,views=previous_views);observations['agent']=previous_observations
                        promotion_history.append(dict(instance=name,promoted=False,reason='no_verified_candidate',previous=selected['mesh']))
                    selected['agent_review_status']='pending_after_repair'
                    save(ROOT/'promotion_history.json',promotion_history)
                history.append(step);save(ROOT/'tool_feedback_history.json',history)
                if not repaired:break
        except Exception as error:
            if best_observations:
                observations['agent']=deepcopy(best_observations)
                restored=masks_from_request(best_observations,sizes)
                for name,context in contexts.get('agent',{}).items():
                    context.update(masks=restored[name],views=next(x['views'] for x in best_observations['instances'] if x['instance_id']==name))
            errors.append(dict(stage='tool_agent_feedback',error=f'{type(error).__name__}: {error}'))
            for row in states.get('agent',{}).values():
                if row['agent_review_status']!='keep':row['agent_review_status']='framework_failed_best_effort'
    for method,instances in states.items():
        for name,selected in instances.items():
            selected['accepted_tool_hypothesis']=bool(selected.get('numerical_recovery_passed'))
            selected['baseline_preserved']=True
            selected['selection_origin']='heuristic_baseline' if method=='agent' and selected['mesh']==baseline_states[name]['mesh'] else method+'_verified_selection'
            selected['output_policy']=classify(selected,load(selected['mesh']))
            for side,mask in contexts[method][name]['masks'].items():
                cv2.imwrite(str(ROOT/method/name/(side+'_final_proposed_mask.png')),mask.astype('uint8')*255)
            save(ROOT/method/name/'final_selection.json',selected)
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha for path,sha in protected_baseline.items()),'Protected heuristic baseline changed'
    result=dict(case_id=case['case_id'],baseline_geometry_hashes=protected_baseline,tools=states,errors=errors,actual_feedback_calls=backend.index if backend else 0,
        estimated_feedback_cost_usd=backend.estimated_api_cost_usd if backend else 0.,independent_accuracy_measured=False,physics_validated=False)
    save(ROOT/'complete_tools.json',result)


if __name__=='__main__':main()
