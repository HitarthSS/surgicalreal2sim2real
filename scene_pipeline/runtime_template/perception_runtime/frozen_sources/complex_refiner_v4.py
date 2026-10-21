"""Isolated agent/heuristic loop for diagnostic-driven, reversible scene proposals.

Inputs/limits/original results remain frozen. Same tool menu and verification for
both controllers. Model output is data, never code or paths. This development
stage returns inspectable candidates, never claims real accuracy or training.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import time

from compact_agent_protocol import digest, encoded
from reconstruction_tools import fingerprints, strict_json
from needle_thread_connection_tool import decode_attachment_response
from complex_case_diagnostics import write_json


FIELDS={'scene_version','target','action_id','evidence_ids','reason'}
PROTOCOL='surgical_complex_refinement_v4_2'


def repair_signature(op,parameters,revision):
    # Mask proposals always start from the frozen original. Repeating them
    # after a revision would produce identical edits and waste a call.
    return encoded({'operation':op,'parameters':parameters,
                    'parent_revision':revision if op=='reconstruct_thread' else 0})


def compact_tool_feedback(result):
    value={k:v for k,v in result.items() if k not in ('inspection','backend_report','shared_tool_result')}
    if result.get('shared_tool_result'):
        t=result['shared_tool_result']
        value['shared_reconstruction']={k:t.get(k) for k in
            ('status','accepted','tool','effective_parameters','backend_exit_code','backend_status','error','verification')}
    return value


@dataclass(frozen=True)
class Budget:
    model_calls:int=4
    tool_calls:int=4
    evidence_reads:int=1

    def __post_init__(self):
        for key,maximum in [('model_calls',6),('tool_calls',6),('evidence_reads',2)]:
            value=getattr(self,key)
            if isinstance(value,bool) or not isinstance(value,int) or not 0<=value<=maximum:
                raise ValueError('Invalid bounded budget: '+key)


def action_menu(report, attempted, remaining, candidate, available):
    """Same capability-derived menu for both controllers; no case-ID rules."""
    actions={'defer':{'operation':'defer','parameters':{},'effect':'Retain unresolved geometry and say why.'}}
    if report.get('last_tool_result',{}).get('infrastructure_failure'):
        return actions # Algorithm tuning cannot fix a broken execution runtime.
    if remaining['tool_calls']<=0:return actions
    def add(aid,op,params,effect):
        signature=repair_signature(op,params,candidate.get('mask_content_id',candidate.get('revision',0)))
        if op in available and signature not in attempted:
            actions[aid]={'operation':op,'parameters':params,'effect':effect}
    if report.get('attachment'):
        add('inspect.attachment','inspect_attachment',{},
            'Show both-view endpoints and decompose the 3D gap; this does not create an attachment.')
        # No geometry-fitting knob can expose a hidden endpoint. Preserve the
        # user's exclusion of fixed-size needles, switching and hidden extension.
        return actions
    if remaining['evidence_reads']>0 and not candidate.get('mask_revision'):
        add('inspect.fragments','inspect_fragments',{},
            'Read endpoint IDs, mask junctions and image-boundary truncation in both views.')
    pending = candidate.get('mask_revision') and not candidate.get('last_reconstruction')
    can_propose = not pending and remaining['tool_calls']>=2 and remaining['model_calls']>=2
    for aid,mode,tol,band,effect in [
        ('stereo.trim','trim',2,4,'Propose removing left-mask pixels unsupported by the right mask; inspect losses.'),
        ('stereo.trim.strict','trim',1,4,'Test a stricter 1px stereo mask agreement hypothesis; inspect any lost thread support.'),
        ('stereo.recover','recover',2,6,'Propose nearby visible-mask additions supported by right-view mask and estimated depth.'),
        ('stereo.recover.wide','recover',2,8,'Test an 8px visible recovery band when the 6px band is inadequate; no hidden connector.')]:
        if can_propose:add(aid,'stereo_mask',{'mode':mode,'tolerance_px':tol,'recovery_band_px':band},effect)
    for aid,prompt,threshold in [('segment.precision','suture thread',.35),
                                 ('segment.recall','thin suture thread',.20)]:
        if can_propose:add(aid,'resegment_thread',{'prompt':prompt,'confidence_threshold':threshold},
                          'Rerun existing fine-tuned SAM3 in both views; freeze needle masks; propose, audit and verify.')
    if candidate.get('mask_revision'):
        add('reconstruct.qp','reconstruct_thread',{'method':'thread_qp','parameters':{}},
            'Run unchanged original QP and shared verifier on the proposed masks; retain original-mask audit.')
        left=report.get('candidate_mask',{}).get('views',{}).get('left',{})
        if left.get('join_only_topology_possible') and left.get('retained_components',0)>1:
            add('reconstruct.occlusion','reconstruct_thread',{'method':'thread_occlusion','parameters':{}},
                'Run existing occlusion/stereo joining and shared verifier; no relaxed physical thresholds.')
    return actions


def validate_decision(raw,state):
    d=strict_json(raw) if isinstance(raw,str) else strict_json(encoded(raw))
    if not isinstance(d,dict) or set(d)!=FIELDS:raise ValueError('Exactly five decision fields required')
    if d['scene_version']!=state['scene_version'] or d['target']!=state['target']:
        raise ValueError('Stale scene version or wrong target')
    if not isinstance(d['action_id'],str) or d['action_id'] not in state['actions']:
        raise ValueError('Action is unavailable or already attempted')
    ids=d['evidence_ids']
    if (not isinstance(ids,list) or not 1<=len(ids)<=3 or any(not isinstance(i,str) for i in ids)
        or len(set(ids))!=len(ids) or any(i not in state['evidence_ids'] for i in ids)):
        raise ValueError('Use 1-3 exact, unique current evidence IDs')
    if not isinstance(d['reason'],str) or not 1<=len(d['reason'].strip())<=500:
        raise ValueError('A brief reason is required')
    return d


def heuristic_decision(state):
    """Transparent matched control. Its order is frozen before pilot execution."""
    report=state['evidence'];mask=report.get('candidate_mask',report.get('mask',{})).get('views',{})
    left,right=mask.get('left',{}),mask.get('right',{})
    priority=['inspect.attachment'] if report.get('attachment') else ([] if left and right else ['inspect.fragments'])
    priority+=(['reconstruct.occlusion','reconstruct.qp'] if left.get('retained_components',0)>1
               else ['reconstruct.qp','reconstruct.occlusion'])
    if left.get('terminal_count',0)>right.get('terminal_count',0) or left.get('junction_pixel_count',0)>0:
        priority+=['segment.precision','segment.recall','stereo.trim','stereo.trim.strict','stereo.recover']
    else:
        priority+=['segment.recall','segment.precision','stereo.recover','stereo.recover.wide','stereo.trim']
    priority+=['defer']
    aid=next(a for a in priority if a in state['actions'])
    return {'scene_version':state['scene_version'],'target':state['target'],'action_id':aid,
            'evidence_ids':['diagnosis'],
            'reason':'Frozen common-menu heuristic: prefer SAM3 precision for surplus left endpoints or mask junctions, otherwise recall; verify split candidates with occlusion when legal, otherwise QP; try a distinct alternative only after verification.'}


def messages_for(state,images):
    def concise(value):
        if isinstance(value,dict):
            return {k:concise(v) for k,v in value.items() if k not in
                    {'input_sha256','provenance','overlay','proposed_masks','protected_candidate_paths',
                     'candidate_fingerprints','input_fingerprints','source_fingerprints','artifacts','artifact_sha256'}}
        if isinstance(value,list):return [concise(v) for v in value]
        return value
    instructions=('Choose one exact action_id. Return ONLY JSON with scene_version,target,action_id,evidence_ids,reason. '
        'Use numerical evidence AND both images. Mask branches are not physical branches. Boundary endpoints may '
        'mean thread exits the field of view; do not invent an in-frame join. A visible needle endpoint need not '
        'be its hidden attachment endpoint. Do not change needle size/estimator, extend it, relax the 4mm connector '
        'limit, or pretend resegmentation is accuracy. Mask tools propose revisions; reconstruction verifies them. '
        'Choose a repair hypothesis based on its failure mechanism, then inspect tool results. If evidence cannot '
        'support a repair, defer. Do not force success, repeat an unchanged action, or fix missing manual tool poses '
        'by changing thread parameters. No more than 500 reason characters.\nSTATE:'+encoded(concise(state)))
    if len(instructions)>16000:raise ValueError('Compact prompt exceeds declared character budget')
    content=[{'type':'text','text':instructions}]
    for eid,path in images:
        content += [{'type':'text','text':'EVIDENCE_ID='+eid},{'type':'image','url':str(path)}]
    return [{'role':'user','content':content}]


class ComplexRefiner:
    def __init__(self,case_id,report,images,protected_paths,out,tools,budget=Budget(),controller='agent',
                 inference_kind='actual_model'):
        if controller not in ('agent','heuristic'):raise ValueError('Unknown controller')
        if inference_kind not in ('actual_model','scripted_contract_test'):raise ValueError('Unknown inference kind')
        self.case_id=case_id;self.evidence=json.loads(encoded(report));self.images=list(images)
        self.paths={**protected_paths,**{eid:str(p) for eid,p in images},'refiner_source':__file__}
        self.provenance=fingerprints(self.paths);self.out=Path(out);self.out.mkdir(parents=True,exist_ok=False)
        self.live_evidence_paths={};self.live_evidence_fingerprints={}
        self.tools=tools;self.budget=budget;self.controller=controller;self.inference_kind=inference_kind
        self.model_calls=0;self.model_completed=0;self.decisions=0;self.tool_calls=0;self.reads=0;self.attempted=set();self.candidate={'revision':0}
        self.history=[];self.invalid=0;self.failed_tools=0;self.feedback=None

    def assert_unchanged(self):
        if fingerprints(self.paths)!=self.provenance:raise ValueError('Protected sources or inputs changed')
        if fingerprints(self.live_evidence_paths)!=self.live_evidence_fingerprints:
            raise ValueError('Diagnostic image evidence changed outside a logged tool')
        if self.candidate.get('protected_candidate_paths'):
            if fingerprints(self.candidate['protected_candidate_paths'])!=self.candidate['candidate_fingerprints']:
                raise ValueError('Candidate changed outside the logged tool execution')

    def state(self):
        remaining={'model_calls':self.budget.model_calls-self.decisions,'tool_calls':self.budget.tool_calls-self.tool_calls,
                   'evidence_reads':self.budget.evidence_reads-self.reads}
        state={'protocol':PROTOCOL,'case_id':self.case_id,'target':'scene_refinement','remaining':remaining,
               'evidence':self.evidence,'candidate':{k:v for k,v in self.candidate.items() if k not in ('protected_candidate_paths','candidate_fingerprints')},
               'actions':action_menu(self.evidence,self.attempted,remaining,self.candidate,self.tools),
               'evidence_ids':['diagnosis','tool_result',*[eid for eid,_ in self.images]],
               'previous_feedback':self.feedback,'independent_accuracy_measured':False,'training_eligible':False}
        state['scene_version']=digest({'state':state,'provenance':self.provenance,
                                       'image_evidence':self.live_evidence_fingerprints,'attempted':sorted(self.attempted)})
        return state

    def run(self,infer=None):
        start=time.perf_counter();status='budget_exhausted';error=None
        try:
            # Decision budget counts inspections, rejected outputs and deferral;
            # the heuristic receives the same maximum number of decisions.
            for iteration in range(self.budget.model_calls):
                self.assert_unchanged();state=self.state();write_json(self.out/f'state_{iteration:03d}.json',state)
                if list(state['actions'])==['defer']:
                    status='automatic_defer_no_supported_action';break
                before=time.perf_counter();row={'iteration':iteration,'scene_version':state['scene_version']}
                self.history.append(row)
                self.decisions+=1
                if self.controller=='agent':
                    if infer is None:raise ValueError('Actual agent requires an inference backend')
                    self.model_calls+=1;messages=messages_for(state,self.images)
                    write_json(self.out/f'input_{iteration:03d}.json',{'messages':messages,'provenance':self.provenance})
                    raw=infer(messages);(self.out/f'raw_{iteration:03d}.txt').write_text(raw+'\n')
                    self.model_completed+=1
                    row['usage']=getattr(infer,'last_usage',None)
                    self.assert_unchanged()
                    try:
                        decoded,transport=decode_attachment_response(raw);row['transport']=transport
                        decision=validate_decision(decoded,state)
                    except (ValueError,TypeError) as exc:
                        self.invalid+=1;self.feedback={'invalid_decision':str(exc),'no_tool_executed':True}
                        row.update(status='invalid_decision',error=str(exc));continue
                else:decision=validate_decision(heuristic_decision(state),state)
                row['decision']=decision;card=state['actions'][decision['action_id']];row['resolved_action']=card
                if card['operation']=='defer':row['status']='deferred';status='deferred';break
                self.tool_calls+=1
                if card['operation']=='inspect_fragments':self.reads+=1
                signature=repair_signature(card['operation'],card['parameters'],
                                           self.candidate.get('mask_content_id',self.candidate.get('revision',0)))
                self.attempted.add(signature)
                folder=self.out/f'tool_{self.tool_calls:03d}'
                try:
                    result=self.tools[card['operation']](dict(card['parameters']),self.candidate,folder)
                    self.assert_unchanged()
                    row['tool_result']=result;row['status']='tool_executed'
                    if result.get('candidate'):
                        # Only shared executable tools may revise state; model text cannot.
                        self.candidate=result['candidate'];self.assert_unchanged()
                    inspection=result.get('inspection',{})
                    visual=inspection.get('candidate_mask',inspection.get('mask',inspection.get('attachment',{})))
                    views=visual.get('views',{})
                    if all(views.get(side,{}).get('overlay') for side in ('left','right')):
                        overlays={side:views[side]['overlay'] for side in ('left','right')}
                        self.images=[('overlay.'+side,overlays[side]) for side in ('left','right')]
                        self.live_evidence_paths={'overlay.'+s:p for s,p in overlays.items()}
                        self.live_evidence_fingerprints=fingerprints(self.live_evidence_paths)
                        if result.get('candidate'):
                            self.candidate['protected_candidate_paths'].update(self.live_evidence_paths)
                            self.candidate['candidate_fingerprints']=fingerprints(self.candidate['protected_candidate_paths'])
                    self.evidence['last_tool_result']=compact_tool_feedback(result)
                    if result.get('inspection'):self.evidence.update(result['inspection'])
                    self.feedback=None
                    if result.get('candidate_consistency_passed'):
                        status='candidate_checks_passed_needs_review';break
                except Exception as exc:
                    self.failed_tools+=1;row.update(status='tool_failed',error=f'{type(exc).__name__}: {exc}')
                    self.feedback={'failed_operation':card['operation'],'error':row['error'],'no_fallback':True}
                row['runtime_seconds']=time.perf_counter()-before
                write_json(self.out/f'event_{iteration:03d}.json',row)
        except Exception as exc:
            status='controller_failed';error=f'{type(exc).__name__}: {exc}'
        summary={'protocol':PROTOCOL,'case_id':self.case_id,'controller':self.controller,'status':status,
                 'inference_kind':self.inference_kind,'actual_model_used':self.controller=='agent' and self.model_completed>0 and self.inference_kind=='actual_model',
                 'model_calls_attempted':self.model_calls,'model_calls_completed':self.model_completed,
                 'decision_count':self.decisions,'tool_calls':self.tool_calls,'evidence_reads':self.reads,
                 'invalid_decisions':self.invalid,'failed_tools':self.failed_tools,'history':self.history,
                 'candidate':self.candidate,'provenance':self.provenance,'budget':self.budget.__dict__,
                 'runtime_seconds':time.perf_counter()-start,'error':error,
                 'independent_accuracy_measured':False,'physics_validated':False,'training_eligible':False,
                 'api_cost_usd':0.,'compute_cost_usd':None}
        write_json(self.out/'summary.json',summary);write_json(self.out/'final_state.json',self.state())
        return summary
