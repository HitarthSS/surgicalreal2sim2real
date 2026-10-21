"""Matched bounded controllers over explicit visibility/hidden-connection tools.

Actual agents require an actual-model queue. No scripted decisions are labeled
as model decisions. Numerical proposal checks do not approve production assembly.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
from compact_agent_protocol import digest,encoded
from complex_case_diagnostics import inspect_thread_masks,write_json
from complex_refiner_v4 import validate_decision
from reconstruction_tools import fingerprints
from needle_thread_connection_tool import decode_attachment_response
from run_multiscene_pilot import QueueModel
from thread_visibility_repairs_v5 import (RepairLimits,cross_view_recovery,load_parts,
    join_proposals,smooth_connection,load_inputs)


class VisibilityRepairTools:
    def __init__(self,case_path,pipeline):
        self.case_path=Path(case_path);self.case=json.loads(self.case_path.read_text());self.pipeline=Path(pipeline)
        self.paths={**self.case['inputs'],'case':self.case_path,'runner':__file__,
                    'repairs':Path(__file__).with_name('thread_visibility_repairs_v5.py')}
        for name in ('compact_agent_protocol.py','complex_case_diagnostics.py','complex_refiner_v4.py',
                     'reconstruction_tools.py','reconstruction_verification.py','run_multiscene_pilot.py',
                     'needle_thread_connection_tool.py'):
            self.paths['dependency/'+name]=Path(__file__).with_name(name)
        for name in ('reconstruct_thread_from_files.py',):self.paths[name]=self.pipeline/name
        self.expected=fingerprints(self.paths)
        base,_,masks,_,_,_,_=load_inputs(self.case,self.pipeline)
        self.reference_preconditions={}
        for side,mask in masks.items():
            try:_,graph,_,count,discarded=base.skeleton_graph(mask,0.,30)
            except RuntimeError as error:
                if 'usable skeleton' not in str(error):raise
                self.reference_preconditions[side]=dict(eligible=False,components=0,endpoints=0,junction_pixels=0,reason='No usable thread skeleton; segmentation repair required')
                continue
            ends=sum(graph.degree[n]==1 for n in graph);branches=sum(graph.degree[n]>2 for n in graph)
            eligible=ends==2*count and branches==0
            self.reference_preconditions[side]=dict(eligible=eligible,components=count,endpoints=ends,
                junction_pixels=branches,reason='ordered fragment paths available' if eligible else
                'reference mask branches/loops need segmentation repair; tolerance cannot fix topology')
        self.assert_unchanged()

    def assert_unchanged(self,candidate=None):
        if fingerprints(self.paths)!=self.expected:raise ValueError('Protected source/input changed')
        if candidate and fingerprints(candidate['protected_paths'])!=candidate['fingerprints']:
            raise ValueError('Proposal changed outside a logged tool')

    def execute(self,card,candidate,out):
        self.assert_unchanged(candidate);op=card['operation'];p=card['parameters']
        if op=='cross_view_recovery':
            if set(p)!={'reference','stereo_match_px'}:raise ValueError('Unexpected cross-view parameters')
            report=cross_view_recovery(self.case,self.pipeline,p['reference'],out,
                                      RepairLimits(stereo_match_px=p['stereo_match_px']))
            fragment_path=Path(out)/'fragments.npz'
            # One verified continuous observed path needs no hidden join.
            stage=('connected' if report['numerical_checks_passed'] and report['fragment_count']==1
                   else 'fragments')
        elif op=='smooth_connection':
            if set(p)!={'proposal_id'} or not candidate or not candidate['report']['numerical_checks_passed']:
                raise ValueError('Need a numerically checked cross-view proposal')
            report=smooth_connection(self.case,self.pipeline,candidate['fragment_path'],p['proposal_id'],out)
            fragment_path=Path(out)/'joined_candidate.npz';stage='connected' if report['numerical_checks_passed'] else 'fragments'
        else:raise ValueError('Unavailable operation')
        if not fragment_path.exists():fragment_path=Path(candidate['fragment_path'])
        protected={'fragments':str(fragment_path.resolve()),'report':str((Path(out)/'report.json').resolve())}
        images={s:str((Path(out)/(s+'_proposal.png')).resolve()) for s in ('left','right')}
        protected.update({'overlay.'+s:p for s,p in images.items()})
        proposal=dict(stage=stage,fragment_path=str(fragment_path.resolve()),report=report,
                      protected_paths=protected,fingerprints=fingerprints(protected),images=images)
        self.assert_unchanged();return proposal

    def menu(self,candidate,attempted):
        actions={'defer':{'operation':'defer','parameters':{},'effect':'Preserve unresolved/ambiguous geometry.'}}
        preview=[]
        if candidate and candidate['stage']=='connected':return actions,preview
        if candidate and candidate['report'].get('schema')=='single_thread_smooth_connection_v5':
            # Re-lifting the identical original masks cannot resolve ambiguous
            # endpoint topology or a failed final connection verification.
            return actions,preview
        if candidate and candidate['report']['numerical_checks_passed'] and self.case.get('scene_priors',{}).get('single_thread') is True:
            rows=join_proposals(self.case,self.pipeline,load_parts(candidate['fragment_path']))
            preview=[{k:v for k,v in row.items() if not k.startswith('_')} for row in rows]
            for row in preview:
                if row['numerical_checks_passed']:
                    aid='connect.'+row['proposal_id']
                    if aid not in attempted:
                        actions[aid]={'operation':'smooth_connection','parameters':{'proposal_id':row['proposal_id']},
                                      'effect':'Propose explicit endpoint connection; hidden shape/depth remain inferred.'}
        if not candidate or not candidate['report']['numerical_checks_passed']:
            for view in ('left','right'):
                if not self.reference_preconditions[view]['eligible']:continue
                for tolerance in (2.,4.):
                    aid=f'cross.{view}.{int(tolerance)}px'
                    if aid not in attempted:
                        actions[aid]={'operation':'cross_view_recovery','parameters':{'reference':view,'stereo_match_px':tolerance},
                            'effect':'Preserve full ordered mask paths in reference view; infer only unsupported depth from visible anchors.'}
        return actions,preview


def heuristic_parameters(config=None):
    values={'initial_tolerance_px':2,'connection_preference':'score'}
    config=config or {}
    if set(config)-set(values):raise ValueError('Unknown heuristic parameter')
    values.update(config)
    if type(values['initial_tolerance_px']) is not int or values['initial_tolerance_px'] not in (2,4):
        raise ValueError('Heuristic must use the same bounded 2px/4px tools')
    if values['connection_preference'] not in ('score','scale_0.65','scale_1.0'):
        raise ValueError('Unknown bounded connection preference')
    return values


def heuristic_action(state,config=None):
    config=heuristic_parameters(config)
    if state['connection_candidates']:
        valid=[r for r in state['connection_candidates'] if 'connect.'+r['proposal_id'] in state['actions']]
        if valid:
            preference=config['connection_preference']
            def rank(row):
                return ((abs(row['tangent_scale']-float(preference[6:])) if preference!='score' else 0),row['score'])
            return 'connect.'+min(valid,key=rank)['proposal_id']
    views=state['evidence']['mask']['views']
    ranked=sorted(views,key=lambda s:(views[s]['junction_pixel_count'],
        abs(views[s]['terminal_count']-2*views[s]['retained_components']),len(views[s]['border_endpoint_ids']),s))
    for s in ranked:
        first=config['initial_tolerance_px']
        for tol in (first,6-first):
            aid=f'cross.{s}.{tol}px'
            if aid in state['actions']:return aid
    return 'defer'


def messages_for(state,images):
    # Compact evidence, exact candidate IDs, no paths or full provenance dump.
    s=json.loads(encoded(state))
    if s.get('candidate'):
        report=s['candidate']
        for k in ('provenance','alternatives','labels'):report.pop(k,None)
    for v in s['evidence']['mask']['views'].values():v.pop('overlay',None)
    s['evidence']['mask'].pop('input_sha256',None)
    all_connections=s['connection_candidates']
    s['rejected_connection_diagnostics']=dict(Counter(f for r in all_connections for f in r['failures']))
    s['connection_candidates']=[{k:r[k] for k in ('proposal_id','endpoints','tangent_scale','endpoint_gap_m','hidden_length_m','views','score')}
                               for r in all_connections if not r['failures']]
    if s.get('candidate') and s['candidate'].get('endpoints'):
        s['candidate']['endpoints']=[{k:r[k] for k in ('endpoint_id','left_xy','right_xy','near_boundary','depth_inferred')}
                                    for r in s['candidate']['endpoints']]
    def compact_numbers(value):
        if isinstance(value,dict):return {k:compact_numbers(v) for k,v in value.items()}
        if isinstance(value,list):return [compact_numbers(v) for v in value]
        if isinstance(value,float):return round(value,6)
        return value
    s=compact_numbers(s) # Display precision only; executable cards remain unchanged.
    text=('Select one exact action_id using BOTH stereo images and numerical feedback. Return ONLY JSON '
          'with scene_version,target,action_id,evidence_ids,reason. Use 1-3 exact current evidence IDs; reason <=500 characters. '
          'Cross-view visibility is distinct from tool occlusion: do not force an in-frame shortcut at a boundary. '
          'Right-only 2D support cannot directly triangulate depth. Single-thread is a user-declared development assumption, '
          'not measured truth. Hidden curves are hypotheses: prefer supported endpoint pairs and smooth continuation; '
          'preserve alternatives. All geometry/depth caps and verification limits are protected. Do not change needle, '
          'invent tool poses or attachments, claim independent accuracy, or repeat an attempted action. '
          'Choose defer when no supported repair exists.\nSTATE:'+encoded(s))
    if len(text)>18000:raise ValueError('Compact prompt exceeds bound')
    content=[{'type':'text','text':text}]
    for eid,path in images:content += [{'type':'text','text':'EVIDENCE_ID='+eid},{'type':'image','url':str(path)}]
    return [{'role':'user','content':content}]


def run(case_path,pipeline,out,controller,queue=None,infer_backend=None,model_manifest=None,heuristic_config=None):
    if controller not in ('heuristic','agent'):raise ValueError('Unknown controller')
    if controller=='agent' and queue is None and infer_backend is None:raise ValueError('Actual-model backend required')
    if queue is not None and infer_backend is not None:raise ValueError('Choose exactly one model backend')
    if controller!='agent' and infer_backend is not None:raise ValueError('Heuristic does not invoke a model')
    config=heuristic_parameters(heuristic_config)
    if controller=='agent' and heuristic_config is not None:raise ValueError('Agent chooses its own legal cards')
    out=Path(out);out.mkdir(parents=True,exist_ok=False);tools=VisibilityRepairTools(case_path,pipeline)
    evidence={'mask':inspect_thread_masks(tools.case['inputs'],pipeline,out/'initial_inspection')}
    candidate=None;attempted=set();history=[];status='budget_exhausted';infer=infer_backend or (QueueModel(Path(queue)) if queue else None)
    if queue and any(infer.root.glob('request_*.json')):raise ValueError('Use a fresh model queue; no stale response reuse')
    completed=0
    start=time.perf_counter();images=[('image.'+s,tools.case['inputs'][s]) for s in ('left','right')]
    for iteration in range(4):
        tools.assert_unchanged(candidate);actions,preview=tools.menu(candidate,attempted)
        state=dict(protocol='visibility_refiner_v5_1',target='scene_refinement',case_id=tools.case['case_id'],
            evidence=evidence,candidate=candidate['report'] if candidate else None,
            reference_preconditions=tools.reference_preconditions,
            scene_priors=tools.case.get('scene_priors',{}),connection_candidates=preview,actions=actions,
            last_tool_feedback=({k:history[-1][k] for k in ('status','error','decision') if k in history[-1]} if history else None),
            remaining={'decisions':4-iteration},evidence_ids=['diagnosis',*[eid for eid,_ in images]])
        state['scene_version']=digest({'state':state,'inputs':tools.expected,'attempted':sorted(attempted)})
        write_json(out/f'state_{iteration:03d}.json',state)
        if len(actions)==1:status='deferred_no_supported_action';break
        t0=time.perf_counter()
        if infer:
            messages=messages_for(state,images);write_json(out/f'input_{iteration:03d}.json',{'messages':messages})
            try:
                raw=infer(messages);completed+=1;(out/f'raw_{iteration:03d}.txt').write_text(raw+'\n')
            except Exception as exc:
                event=dict(iteration=iteration,status='inference_failed',error=f'{type(exc).__name__}: {exc}')
                history.append(event);write_json(out/f'event_{iteration:03d}.json',event);status='inference_failed';break
            try:decoded,transport=decode_attachment_response(raw);decision=validate_decision(decoded,state)
            except (ValueError,TypeError) as exc:
                event=dict(iteration=iteration,status='invalid_decision',error=str(exc),usage=infer.last_usage)
                history.append(event);write_json(out/f'event_{iteration:03d}.json',event);continue
        else:
            decision=validate_decision(dict(scene_version=state['scene_version'],target=state['target'],
                action_id=heuristic_action(state,config),evidence_ids=['diagnosis'],
                reason='Common-menu heuristic: rank branch-free views and apply logged fixed correspondence/curve preference; never change physical gates.'),state)
        aid=decision['action_id'];card=actions[aid]
        event=dict(iteration=iteration,decision=decision,resolved_action=card,usage=infer.last_usage if infer else None)
        attempted.add(aid)
        if aid=='defer':event['status']='deferred';status='deferred';history.append(event);write_json(out/f'event_{iteration:03d}.json',event);break
        try:
            candidate=tools.execute(card,candidate,out/f'tool_{iteration+1:03d}')
            event.update(status='tool_executed',report=candidate['report'])
            tools.assert_unchanged(candidate);images=[('overlay.'+s,candidate['images'][s]) for s in ('left','right')]
            if candidate['stage']=='connected':status='connected_proposal_needs_review'
        except Exception as exc:event.update(status='tool_failed',error=f'{type(exc).__name__}: {exc}')
        event['runtime_seconds']=time.perf_counter()-t0;history.append(event);write_json(out/f'event_{iteration:03d}.json',event)
        if status=='connected_proposal_needs_review':break
    tools.assert_unchanged(candidate)
    summary=dict(protocol='visibility_refiner_v5_1',controller=controller,status=status,history=history,
        actual_model_used=bool(completed),model_calls_completed=completed,
        model_calls_attempted=infer.index if infer else 0,decision_count=len(history),
        candidate=candidate,provenance=tools.expected,runtime_seconds=time.perf_counter()-start,
        independent_accuracy_measured=False,training_eligible=False,shared_reconstruction_accepted=False,
        policy_trained=False,physics_validated=False,api_cost_usd=None if infer_backend else 0,compute_cost_usd=None,
        model_manifest=model_manifest,
        estimated_api_cost_usd=getattr(infer,'estimated_api_cost_usd',None))
    summary['heuristic_config']=config if controller=='heuristic' else None
    write_json(out/'summary.json',summary);return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('case','pipeline-tools','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--controller',choices=('heuristic','agent'),required=True);p.add_argument('--queue',type=Path)
    a=p.parse_args();s=run(a.case,a.pipeline_tools,a.out,a.controller,a.queue)
    print(json.dumps({k:s[k] for k in ('status','controller','decision_count','actual_model_used')}))


if __name__=='__main__':main()
