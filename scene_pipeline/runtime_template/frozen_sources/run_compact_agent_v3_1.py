"""Bounded v3.1 action-card controller; explicit evidence and live scene preflight feedback.

Invalid model requests consume a decision and are logged, never rewritten or
executed. The next decision may use concise rejection feedback within the SAME
two-decision budget. The original controller remains unchanged as a fallback.
"""
import json
from pathlib import Path
import time

from compact_agent_protocol_v3_1 import (PROTOCOL,image_evidence,make_state,messages_for,
                                       shared_request,resolve_action,signature,validate_decision)
from reconstruction_tools import fingerprints
from run_bounded_agent_pilot import assert_unchanged
from needle_thread_connection_tool import decode_attachment_response


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def run_compact_loop(session,infer,model_manifest,initial_result=None,inference_kind='actual_model',
                     scene_feedback_provider=None):
    if inference_kind not in {'actual_model','scripted_contract_test'}:
        raise ValueError('Unknown inference kind')
    if initial_result is not None:
        recorded=session.out/initial_result['call_id']/'tool_result.json'
        if not recorded.is_file() or json.loads(recorded.read_text())!=initial_result:
            raise ValueError('Initial result is not recorded in this session')
    source_paths={name:Path(__file__).with_name(file) for name,file in {
        'protocol_v3_1':'compact_agent_protocol_v3_1.py','protocol_v3':'compact_agent_protocol_v3.py','protocol_v2':'compact_agent_protocol_v2.py','protocol_fallback':'compact_agent_protocol.py',
        'runner_v3_1':'run_compact_agent_v3_1.py','scene_feedback':'scene_feedback_v2.py',
        'transport':'needle_thread_connection_tool.py','transport_eot':'run_model_tool_checkpoint.py'}.items()}
    if model_manifest.get('gemma_runner'):source_paths['gemma_runner']=Path(model_manifest['gemma_runner'])
    expected=fingerprints(source_paths)
    provenance={'inputs':session.input_fingerprints,'sources':session.source_fingerprints,
                'controller_sources':expected,'case':session.case}
    target=model_manifest['target'];result=initial_result;rounds=[];attempted=[]
    if result and result.get('tool'):
        attempted.append({'operation':result['tool'],'parameters':result['effective_parameters'],
                          'status':result['status'],'kind':'common_baseline'})
    start=time.perf_counter();evidence_reads=0;requested=None;controller_feedback=None
    summary={'protocol':PROTOCOL,'model':model_manifest,'rounds':rounds,'max_model_calls':2,
             'max_total_tool_calls':session.max_calls,'max_evidence_reads':1,'model_calls_attempted':0,
             'model_calls_completed':0,'actual_model_used':False,'inference_kind':inference_kind,
             'api_cost_usd':0.,'compute_cost_usd':None,'status':'started','invalid_decisions':0,
             'common_baseline_included_in_tool_budget':initial_result is not None}

    def build_state(revision):
        feedback=scene_feedback_provider(result) if scene_feedback_provider else None
        images=image_evidence(session,target,result)
        state,catalog=make_state(session.observe(),provenance,target,
            {'model_decisions':2-summary['model_calls_attempted'],
             'tool_calls':session.max_calls-session.index,'evidence_reads':1-evidence_reads},
            result,revision,attempted,images,feedback,controller_feedback)
        return state,catalog,images,feedback

    try:
        state,catalog,images,feedback=build_state(0)
        if state['finish_allowed']:
            summary['status']='routine_target_pass_no_model'
        else:
            for index in range(2):
                assert_unchanged(session,source_paths,expected)
                state,catalog,images,feedback=build_state(index)
                folder=session.out/f'compact_round_{index+1:03d}';folder.mkdir()
                hashes=fingerprints({i:path for i,_,path in images})
                messages=messages_for(state,images,requested)
                for name,value in [('state.json',state),('evidence_catalog.json',catalog),
                                   ('input.json',{'messages':messages,'image_hashes':hashes,
                                                  'provenance':provenance,'model':model_manifest})]:write(folder/name,value)
                before=time.perf_counter();summary['model_calls_attempted']+=1
                raw=infer(messages)
                summary['model_calls_completed']+=1
                summary['actual_model_used']=inference_kind=='actual_model'
                (folder/'raw_response.txt').write_text(raw+'\n')
                assert_unchanged(session,source_paths,expected)
                if scene_feedback_provider and hasattr(scene_feedback_provider,'assert_unchanged'):
                    scene_feedback_provider.assert_unchanged()
                if fingerprints({i:path for i,_,path in images})!=hashes:
                    raise ValueError('Prompt image changed during inference')
                row={'round':index+1,'runtime_seconds':time.perf_counter()-before,
                     'usage':getattr(infer,'last_usage',None),'scene_version':state['scene_version']}
                rounds.append(row)
                try:
                    decoded,transport=decode_attachment_response(raw);row['transport']=transport
                    decision=validate_decision(decoded,state,catalog)
                except (ValueError,TypeError) as error:
                    row.update(status='invalid_model_decision',error=str(error),tool_executed=False)
                    summary['invalid_decisions']+=1
                    controller_feedback={'previous_decision_rejected':str(error),
                        'no_tool_executed':True,'budget_consumed':'one model decision; no extra retry',
                        'instruction':'Select one currently listed action_id, with 1-3 exact evidence IDs. No operation/parameters fields.'}
                    summary['status']='invalid_decision_at_budget'
                    session._event('compact_v3_decision_rejected',**row)
                    write(folder/'result.json',row)
                    continue
                controller_feedback=None;row['decision']=decision;row['resolved_action']=resolve_action(decision,state);row['choice_constrained_to_defer']=state['choice_is_constrained_to_defer'];row['status']='valid_model_decision'
                session._event('compact_v3_model_decision',**row)
                op=state['actions'][decision['action_id']]['operation']
                if op in {'finish','defer'}:
                    summary['status']='target_finished' if op=='finish' else 'deferred'
                elif op=='inspect_evidence':
                    evidence_reads+=1;requested={i:catalog[i] for i in decision['evidence_ids']}
                    row['evidence_returned']=list(requested);summary['status']='evidence_requested'
                else:
                    result=session.call(shared_request(decision,state))
                    attempted.append({'operation':op,'parameters':result.get('effective_parameters'),
                                      'status':result['status'],'kind':'model_selected'})
                    row['tool_result']=result;row['tool_executed']=True;requested=None
                    updated,_,_,feedback=build_state(index+1)
                    row['scene_feedback_after_tool']={k:v for k,v in (feedback or {}).items() if k!='_provenance'}
                    summary['status']='target_passed_with_scene_preflight' if updated['finish_allowed'] else 'failed_at_budget'
                write(folder/'result.json',row)
                if op in {'finish','defer'} or summary['status']=='target_passed_with_scene_preflight':break
    except Exception as error:
        summary.update(status='controller_or_model_failed',error=f'{type(error).__name__}: {error}')
        session._event('compact_v3_failed',error=summary['error'])
    # If feedback computation itself fails, retain a conservative failed state.
    try:state,catalog,images,feedback=build_state(len(rounds))
    except Exception as error:
        feedback={'blocking_for_finish':True,'preflight_passed':False,
                  'assembly':{'status':'feedback_failed','error':str(error)},'attachment':{'status':'not_assessed'}}
        images=image_evidence(session,target,result)
        state,catalog=make_state(session.observe(),provenance,target,
            {'model_decisions':2-summary['model_calls_attempted'],'tool_calls':session.max_calls-session.index,
             'evidence_reads':1-evidence_reads},result,len(rounds),attempted,images,feedback,controller_feedback)
    summary.update(validity=state['validity'],finish_allowed=state['finish_allowed'],last_tool_result=result,
                   tool_calls=session.index,evidence_reads=evidence_reads,
                   scene_feedback=feedback,runtime_seconds=time.perf_counter()-start)
    write(session.out/'final_state.json',state);write(session.out/'compact_summary.json',summary)
    return summary
