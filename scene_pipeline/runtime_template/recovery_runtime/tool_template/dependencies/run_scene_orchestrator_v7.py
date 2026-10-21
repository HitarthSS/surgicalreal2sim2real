"""Coded, bounded orchestration of stereo-gap and attachment specialists.

The main planner schedules specialists. Specialists inspect full stereo plus
versioned endpoint diagnostics and choose shared tools. Only the host executes
tools, verifies proposals and commits geometry. Every endpoint retains a ledger
entry; deferring one item cannot silently defer the scene. No policy/physics or
independent accuracy approval is implied by a completed assessment.
"""
import argparse
from copy import deepcopy
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import time
import numpy as np

from request_protocol_v11 import (pack_observation, compact_json, polygon_contract, construct_polygon_request, MAX_PROMPT_CHARACTERS)
from compact_agent_protocol import digest
from complex_case_diagnostics import write_json
from reconstruction_tools import fingerprints,sha256
from openai_scene_backend import OpenAISceneModel
from shared_scene_tools_v7 import SharedSceneToolsV7, heuristic_polygons
from thread_gap_inventory_v7 import build_gap_inventory
from needle_attachment_inventory_v7 import build_attachment_inventory, select_attachment_candidate
from needle_thread_connection_tool import decode_attachment_response, prepare_connected_geometry
from thread_visibility_repairs_v5 import load_parts
from newton_scene_adapter import CAMERA_TO_NEWTON, transform


@dataclass
class Budget:
    max_model_calls: int = 12
    max_tool_calls: int = 10
    model_calls: int = 0
    tool_calls: int = 0

    def consume(self, kind):
        name=kind+'_calls'; maximum='max_'+name
        if getattr(self,name)>=getattr(self,maximum):raise ValueError(kind+' budget exhausted')
        setattr(self,name,getattr(self,name)+1) # Invalid responses/failed tools count.


class RoleModels:
    """Separate specialist prompts/backends, one global accounting boundary."""
    def __init__(self,out,key_file=None,backend_factory=None):
        self.out=Path(out);self.key=key_file;self.backends={};self.factory=backend_factory
        self.calls=[]

    def infer(self,role,messages,budget,charged=False):
        if not charged:budget.consume('model')
        number=sum(c['role']==role for c in self.calls)
        shard=number//6;key=(role,shard)
        row=dict(role=role,index=number+1,status='attempted')
        self.calls.append(row)
        try:
            if key not in self.backends:
                self.backends[key]=(self.factory(role,shard) if self.factory else
                    OpenAISceneModel(self.out/(role+f'_{shard}'),self.key,max_calls=6))
            backend=self.backends[key]
            raw=backend(messages);row.update(status='completed',usage=getattr(backend,'last_usage',None))
            return raw
        except Exception as error:
            row.update(status='failed',error_type=type(error).__name__);raise

    def accounting(self):
        return dict(calls=self.calls,attempted=len(self.calls),completed=sum(c['status']=='completed' for c in self.calls),
            actual_model_used=self.factory is None and any(c['status']=='completed' for c in self.calls),
            estimated_api_cost_usd=sum(getattr(b,'estimated_api_cost_usd',0.) for b in self.backends.values()),
            cost_is_estimate_not_invoice=True)


class EndpointLedger:
    """Persistent per-endpoint coverage, including superseded observations."""
    def __init__(self):self.entries={};self.revision=0

    def refresh(self,inventory):
        current=set(g['gap_id'] for g in inventory['gaps'])
        for key,row in self.entries.items():
            if key not in current and row['status']!='superseded_by_reconstruction':
                row.update(status='superseded_by_reconstruction',superseded_revision=self.revision,
                    correctness_claimed=False)
        for gap in inventory['gaps']:
            key=gap['gap_id'];old=self.entries.get(key)
            if old and old['inventory_version']==inventory['inventory_version']:
                old['gap']=gap
                continue
            # Mask changes can invalidate a prior decision: the evidence version
            # is always logged and reassessed after any committed tool result.
            represented=gap['status']=='represented_by_checked_candidate'
            row=dict(gap_id=key,gap=gap,status='represented' if represented else 'pending',
                inventory_version=inventory['inventory_version'],revision=self.revision,
                assessment_history=old.get('assessment_history',[]) if old else [])
            self.entries[key]=row
        self.revision+=1

    def assess(self,alias_map,assessments,decision,evidence_version):
        seen=set()
        for value in assessments:
            if set(value)!={'focus_id','classification','reason'}:raise ValueError('Unexpected assessment fields')
            alias=value['focus_id']
            if alias not in alias_map or alias in seen:raise ValueError('Unknown/duplicate endpoint focus ID')
            seen.add(alias)
            if value['classification'] not in ('defer','terminal_hypothesis','attachment_terminal'):
                raise ValueError('Only local deferral or explicit terminal hypotheses can close assessments')
            if not isinstance(value['reason'],str) or not 1<=len(value['reason'])<=300:raise ValueError('Assessment reason bound')
        for value in assessments:
            row=self.entries[alias_map[value['focus_id']]]
            event=dict(**value,scene_version=evidence_version,evidence_ids=decision['evidence_ids'],
                hypothesis_not_ground_truth=True)
            row['assessment_history'].append(event)
            row.update(status='deferred' if value['classification']=='defer' else 'terminal_hypothesis',assessment=event)

    def pending(self):return [key for key,row in self.entries.items() if row['status']=='pending']

    def finish(self):
        if self.pending():raise ValueError('Cannot finish: unassessed endpoints remain')
        return dict(all_current_endpoints_assessed=True,geometry_correctness_claimed=False)


def rounded(value):
    if isinstance(value,dict):return {k:rounded(v) for k,v in value.items()}
    if isinstance(value,list):return [rounded(v) for v in value]
    if isinstance(value,float):return round(value,6)
    return value


def validate_decision(raw,state,role):
    decoded,_=decode_attachment_response(raw);d=json.loads(decoded)
    if set(d)!={'scene_version','action_id','parameters','evidence_ids','reason'}:raise ValueError('Exact decision fields required')
    if d['scene_version']!=state['scene_version']:raise ValueError('Stale scene version')
    if not isinstance(d['parameters'],dict):raise ValueError('Parameter object required')
    if not isinstance(d['reason'],str) or not 1<=len(d['reason'])<=500:raise ValueError('Reason bound')
    if not isinstance(d['evidence_ids'],list) or not {'image.left','image.right'}<=set(d['evidence_ids']):
        raise ValueError('Both stereo evidence IDs required')
    if any(e not in state['evidence_ids'] for e in d['evidence_ids']):raise ValueError('Unknown evidence ID')
    if role=='orchestrator':
        if d['action_id']!='plan' or set(d['parameters'])!={'priority'}:raise ValueError('Main agent must plan specialist task coverage')
        ids=d['parameters']['priority']
        if not isinstance(ids,list) or len(ids)!=len(set(ids)) or set(ids)!=set(state['task_ids']):
            raise ValueError('Plan must cover every current task exactly once')
    elif role=='attachment':
        if d['action_id']=='defer':
            if d['parameters']:raise ValueError('Defer cannot change parameters')
        elif d['action_id']=='attach':
            if set(d['parameters'])!={'candidate_id','physical_swage_identified'} or d['parameters']['physical_swage_identified'] is not True:
                raise ValueError('Explicit visual swage identity required')
            if d['parameters']['candidate_id'] not in state['eligible_candidate_ids']:raise ValueError('Unknown/blocked attachment candidate')
        else:raise ValueError('Unknown attachment action')
    elif role=='gaps':
        # A named operation is an unambiguous envelope alias only when it
        # matches the selected, version-bound card. Preserve/log the raw model
        # response; never change the selected tool or its fixed parameters.
        selected=state['tool_map'].get(d['parameters'].get('tool_id'))
        if selected and d['action_id']==selected['card']['operation']:
            d['action_id']='repair'
        if d['action_id']=='assess':
            if set(d['parameters'])!={'assessments'} or not isinstance(d['parameters']['assessments'],list) or not d['parameters']['assessments']:
                raise ValueError('Nonempty per-endpoint assessments required')
        elif d['action_id']=='repair':
            p=d['parameters']
            if not {'tool_id','tool_parameters'}<=set(p):raise ValueError('Tool selection fields required')
            if p['tool_id'] not in state['tool_map'] or not isinstance(p['tool_parameters'],dict):raise ValueError('Unknown tool')
            card=state['tool_map'][p['tool_id']]['card']
            pair_tool=card['operation'] in ('smooth_connection','inferred_connection')
            expected={'tool_id','tool_parameters'} if pair_tool else {'tool_id','tool_parameters','focus_ids'}
            if set(p)!=expected:raise ValueError('Pair tools derive endpoints from the selected card; other tools require focus_ids')
            if not pair_tool and (not isinstance(p['focus_ids'],list) or not p['focus_ids'] or any(k not in state['focus_map'] for k in p['focus_ids'])):raise ValueError('Current endpoint focus required')
            if card['operation']!='occluder_polygons' and p['tool_parameters']:
                raise ValueError('Fixed card parameters cannot be overridden')
            if not pair_tool and len(p['focus_ids'])!=len(set(p['focus_ids'])):raise ValueError('Duplicate focus IDs')
            if card['operation']=='inferred_connection' and not {'image.left','image.right'}<=set(d['evidence_ids']):
                raise ValueError('Inferred pair selection requires both stereo views')
        else:raise ValueError('Unknown gap action')
    return d


ROLE_PROMPTS={
 'orchestrator': 'You are the main scene orchestrator. Plan the assessment order of ALL task_ids exactly once. Schedule needle_attachment early to reserve the physical attachment terminal. Every image endpoint is an assessment task, not necessarily a break. One border exit must never suppress a separate interior gap. Do not reconstruct or assert results yourself.',
 'attachment': 'You are the needle/thread attachment specialist. Inspect BOTH complete stereo images and labeled endpoints. Identify the plausible swaged thread end, not the free sharp tip or merely the nearest fitted arc endpoint. Inferred endpoint labels express uncertainty, not automatic rejection. Keep a sensible eligible attachment when stereo appearance, endpoint proximity, thread direction and needle shape support that relation; explain any occlusion or inferred support in your reason. Treat direction diagnostics cautiously for tiny gaps. No perfect accuracy or visibility is required. Never extend the needle, ignore conflicting visual evidence, or resolve equally plausible endpoint pairs by guessing. Otherwise defer ONLY this attachment; do not veto thread repairs. Every accepted attachment remains an inferred hypothesis, not measured truth.',
 'gaps': 'You are the stereo thread-gap specialist. Assess every pending focus independently. Look in BOTH images: a missing left segment may be visibly continuous in the right. When masks/order/depth are inadequate select a shared segmentation/topology/cross-view tool first. Tool-covered gaps can use bounded smooth endpoint-pair proposals, with direction checks. A declared single-thread prior is available; it does not reveal hidden curves. Use numerical rejection feedback to choose the repair; never override bounds. Never join a reserved needle-attachment terminal. Return one pair repair bound by its tool card (omit redundant focus IDs), one other repair with affected focus IDs, OR explicit assessments for pending focus IDs. A boundary exit is local; continue interior repairs. Only classify an endpoint as a physical terminal hypothesis if visually justified. All rejected pairs are recorded; feedback is per pair/scale. Prefer a supported pair repair before more perception calls. An offered inferred_connection card permits an explicitly uncertain boundary or topology hypothesis under the declared single-thread prior; inspect both views and endpoint directions, select a sensible pair, and explain its uncertainty. Its hidden curve is never measured truth. Defer unlisted, unbounded or visually contradicted connections. Do not repeatedly request unchanged default tools. Tool masks are uncertain hypotheses: inspect feedback against original images; do not treat a polygon satisfying a curve as independent confirmation.'}


def original_image_sizes(images):
    from PIL import Image
    sizes={}
    for eid,path in images:
        if eid in ("image.left","image.right"):
            with Image.open(path) as image:sizes[eid.split(".")[1]]=list(image.size)
    if set(sizes)!={"left","right"}:raise ValueError("Both original image dimensions required")
    return sizes


def messages_for(role,state,images, image_sizes=None):
    public={k:v for k,v in state.items() if k not in ('focus_map','tool_map')}
    public=rounded(public)
    if role=='gaps':
        public['polygon_request_contract']=polygon_contract(image_sizes or original_image_sizes(images))
    # Shared schemas preserve every diagnostic field; small requests stay readable.
    prompt=(ROLE_PROMPTS[role]+' Return ONLY JSON with exactly scene_version,action_id,parameters,evidence_ids,reason. '
        'Use the exact short scene_version and image.left,image.right evidence IDs; reason <=500 chars. '+
        ('For plan: parameters={"priority":[all task_ids in order]}.' if role=='orchestrator' else
         'For attach: parameters={"candidate_id":eligible ID,"physical_swage_identified":true}; defer: parameters={}.' if role=='attachment' else
         'Gap action_id is repair or assess. For pair repairs, action_id="repair", parameters={"tool_id":"tN","tool_parameters":{}}; endpoints are fixed by that card, omit focus_ids. For other repairs: action_id="repair", parameters={"tool_id":"tN","tool_parameters":{} (fixed cards),"focus_ids":["gN"]}. '
         'Only occluder_polygons takes tool_parameters={"polygons":{"left":[polygons],"right":[polygons]}}; obey polygon_request_contract original pixel ranges. '
         'For assess: parameters={"assessments":[{"focus_id":"gN","classification":"defer|terminal_hypothesis|attachment_terminal","reason":"specific evidence, <=300 chars"}]}.')+
        '\nSTATE:'+json.dumps(public,separators=(',',':'),allow_nan=False))
    prefix=prompt.split('\nSTATE:',1)[0]+'\nSTATE:'
    if len(prompt)>MAX_PROMPT_CHARACTERS:
        prompt=prefix+compact_json(pack_observation(public))
    if len(prompt)>MAX_PROMPT_CHARACTERS:
        raise ValueError('Prompt bound: lossless request still exceeds 28000 chars; no evidence was truncated')
    content=[dict(type='text',text=prompt)]
    for eid,path in images:content.extend([dict(type='text',text='EVIDENCE_ID='+eid),dict(type='image',url=str(path))])
    return [dict(role='user',content=content)]


def compact_repeated_text(value):
    """Lossless dictionary encoding; no candidates or numeric evidence dropped."""
    from collections import Counter
    counts=Counter()
    def collect(item):
        if isinstance(item,str):counts[item]+=1
        elif isinstance(item,dict):
            for v in item.values():collect(v)
        elif isinstance(item,list):
            for v in item:collect(v)
    collect(value)
    repeated=[s for s,n in counts.items() if n>1 and len(s)>=64]
    mapping={}
    for s in repeated:
        alias='@text'+str(len(mapping))
        while alias in counts:alias+='x'
        mapping[s]=alias
    def encode(item):
        if isinstance(item,str):return mapping.get(item,item)
        if isinstance(item,dict):return {k:encode(v) for k,v in item.items()}
        if isinstance(item,list):return [encode(v) for v in item]
        return item
    return dict(observation=encode(value),text_dictionary={alias:s for s,alias in mapping.items()},
        encoding='String values @textN in observation refer verbatim to text_dictionary. All original fields, candidates and numbers are retained.')


def bind_state(state,evidence):
    full=digest(state);state['scene_version']='s'+full[:12];state['full_state_sha256']=full
    state['evidence_ids']=[eid for eid,_ in evidence]
    return state


class NeedleAttachmentSpecialist:
    role='attachment'
    def state(self,inventory,budget,images):
        fields=['candidate_id','thread_endpoint_id','needle_endpoint_id','distance_m','blockers','views',
                'numerical_checks_passed','eligible_for_specialist_selection','uncertainty_warnings',
                'inferred_endpoint_present','direction_diagnostics']
        pairs=[{k:c[k] for k in fields if k in c} for c in inventory['candidates']]
        return bind_state(dict(task='needle_attachment',endpoints=inventory['endpoints'],candidates=pairs,
            eligible_candidate_ids=[c['candidate_id'] for c in inventory['candidates'] if c['eligible_for_specialist_selection']],
            budget=asdict(budget),limits='unique stereo-assessed endpoint pair within existing 4mm bound; inferred endpoints allowed and labeled; no hidden needle extension'),images)


class StereoGapSpecialist:
    role='gaps'
    def state(self,inventory,ledger,tools,candidate,attempted,budget,reserved_points,images,feedback=None):
        cards,_=tools.menu(candidate,attempted);cards={k:v for k,v in cards.items() if v['operation'] not in ('keep','defer')}
        lookup={e['endpoint_id']:e for e in inventory['endpoints_3d']};observed={e['endpoint_id']:e for e in inventory['endpoints_2d']}
        pair_lookup={p['pair_id']:p for p in inventory['pairing_candidates']}
        pending_ids=ledger.pending()
        # Inventory coverage remains exhaustive; model observations are bounded
        # batches. Unseen items stay pending and cannot pass the finish gate.
        focus_map={f'g{i}':key for i,key in enumerate(pending_ids[:8])};focus=[]
        for alias,key in focus_map.items():
            gap=ledger.entries[key]['gap'];endpoints=[]
            for eid in gap['endpoint_ids']:
                e=lookup.get(eid,observed.get(eid,{}))
                endpoints.append({k:e[k] for k in ('endpoint_id','tool_endpoint_id','left_xy','right_xy','xy','view',
                    'category_candidates','depth_inferred','stereo_evidence') if k in e})
            focus.append(dict(focus_id=alias,kind=gap['kind'],endpoints=endpoints,source_views=gap['source_views'],
                pair_ids=[pid for pid in gap['candidate_pair_ids'] if pid in pair_lookup],reasons=gap['unresolved_reasons']))
        pairs=[]
        vocabulary=sorted({failure for pair in inventory['pairing_candidates'] for row in pair['scale_proposals'] for failure in row['failures']})
        failure_codes={failure:'f'+str(i) for i,failure in enumerate(vocabulary)}
        for pair in inventory['pairing_candidates']:
            pairs.append(dict(pair_id=pair['pair_id'],tool_endpoint_ids=pair['tool_endpoint_ids'],
                classification=pair['classification'],stereo_cues={s:dict(available=v['available'],
                    continuous_visible_mask_path=v['continuous_visible_mask_path'],reason=v.get('reason'))
                    for s,v in pair['stereo_cues'].items()},ambiguous=pair['endpoint_pair_ambiguous'],
                scales=[[r['tangent_scale'],[failure_codes[f] for f in r['failures']],r['score'],
                    [[v['explained_fraction'],v['occluder_fraction'],v['outside_fraction'],v['mask_support_fraction']]
                        for v in (r['views']['left'],r['views']['right'])],r.get('unobserved_out_of_frame_fraction')]
                    for r in pair['scale_proposals']]))
        allowed=[];tool_map={}
        for aid,card in cards.items():
            if budget.tool_calls>=budget.max_tool_calls:continue
            if card['operation'] in ('smooth_connection','inferred_connection'):
                row=next((r for p in inventory['pairing_candidates'] for r in p['scale_proposals']
                          if r['proposal_id']==card['parameters']['proposal_id']),None)
                boundary=card['parameters'].get('inference_mode')=='boundary'
                if not row or (not row['numerical_checks_passed'] and not (boundary and set(row['failures'])=={'unobserved_out_of_frame_connection'})):continue
                endpoint_ids=row['endpoints']
                coords=[e['points_m'] for e in inventory['endpoints_3d'] if e['tool_endpoint_id'] in endpoint_ids]
                if any(np.linalg.norm(np.asarray(p)-q)<1e-8 for p in coords for q in reserved_points):continue
            alias='t'+str(len(tool_map));tool_map[alias]=dict(action_id=aid,card=card)
            allowed.append(dict(tool_id=alias,operation=card['operation'],parameters=card['parameters'],
                parameter_schema=card.get('parameter_schema'),effect=card['effect']))
            if card['operation'] in ('smooth_connection','inferred_connection'):
                allowed[-1].update(endpoint_ids=card['parameters']['proposal_id'].split('__')[:2],
                    endpoint_binding='host derives canonical endpoints from this versioned card; omit focus_ids')
        state=dict(focus_map=focus_map,tool_map=tool_map,pending=focus,tools=allowed,pairings=pairs,
            pending_total=len(pending_ids),unassessed_not_in_this_batch_count=max(0,len(pending_ids)-len(focus)),
            reference_preconditions=tools.reference_preconditions,scene_priors=tools.case.get('scene_priors'),
            current_geometry=dict(stage=candidate['stage'],failures=candidate['report']['failures'],
                checks_passed=candidate['report']['numerical_checks_passed'],fragment_count=candidate['report']['fragment_count']) if candidate else None,
            budget=asdict(budget),last_feedback=feedback,reserved_attachment_endpoints_camera_m=reserved_points,
            uncertainty_groups=inventory['uncertainty_groups'],hidden_geometry_is_measured=False)
        state['view_checks_order']=['explained','occluder','outside','mask_support']
        state['scale_row_order']=['tangent_scale','failure_codes','score','view_checks_left_then_right','outside_both_fraction']
        state['failure_code_legend']={v:k for k,v in failure_codes.items()}
        if candidate and candidate['report'].get('object_evidence_audit'):
            audit=candidate['report']['object_evidence_audit']
            state['object_evidence']=dict(warnings=audit['warnings'],raw_original_coverage=candidate['report']['original_visible_mask_coverage'],
                object_aware_coverage=candidate['report']['object_aware_original_mask_coverage'],
                ambiguous_short_needle_adjacent_pixels={s:v['ambiguous_needle_adjacent_skeleton_pixels'] for s,v in audit['views'].items()},
                ambiguity_is_not_confirmed_object_identity=True)
        return bind_state(state,images)


class SceneOrchestrator:
    def __init__(self,case_path,pipeline,needle_path,out,controller='agent',models=None,cache=None,budget=None):
        self.out=Path(out).resolve();self.out.mkdir(parents=True,exist_ok=False)
        self.needle=Path(needle_path).resolve();self.tools=SharedSceneToolsV7(case_path,pipeline,cache,needle_path=self.needle)
        self.controller=controller;self.models=models;self.budget=budget or Budget()
        if controller not in ('heuristic','agent') or (controller=='agent' and models is None):raise ValueError('Actual backend required for agent')
        self.ledger=EndpointLedger();self.attempted=set();self.history=[];self.reserved=[]
        self.attachment=None;self.feedback=None;self.snapshot=0
        self.gap_specialist=StereoGapSpecialist();self.attachment_specialist=NeedleAttachmentSpecialist()

    def log(self,row):
        self.history.append(deepcopy(row));write_json(self.out/'history.json',self.history)

    def observe(self,candidate):
        folder=self.out/f'observation_{self.snapshot:03d}';folder.mkdir();self.snapshot+=1
        inv=build_gap_inventory(self.tools.case,self.tools.pipeline,candidate,folder/'gaps')
        att=build_attachment_inventory(self.tools.case,self.tools.pipeline,candidate,self.needle,folder/'attachment')
        self.ledger.refresh(inv)
        self.tools.geometry_version=sha256(candidate['fragment_path']) if candidate else None
        write_json(self.out/'endpoint_ledger.json',self.ledger.entries)
        return inv,att

    def evidence(self,candidate,inventory=None,attachment=None):
        images=[('image.'+s,self.tools.original['inputs'][s]) for s in ('left','right')]
        extra=attachment.get('labeled_images',{}) if attachment else inventory.get('artifacts',{}) if inventory else {}
        images.extend([('labels.'+s,p) for s,p in extra.items()])
        if candidate:images.extend([('geometry.'+s,p) for s,p in candidate['images'].items()])
        if self.feedback and self.feedback.get('feedback_images'):
            images.extend([('tool_feedback.'+s,p) for s,p in self.feedback['feedback_images'].items()])
        return images

    def model_decision(self,role,state,images):
        folder=self.out/f'decision_{len(self.history):03d}_{role}';folder.mkdir()
        write_json(folder/'state.json',state)
        try:messages=messages_for(role,state,images)
        except Exception:
            self.log(dict(role=role,status='prompt_construction_failed',actual_api_attempt=False,budget=asdict(self.budget)))
            raise
        write_json(folder/'messages.json',messages)
        self.budget.consume('model')
        raw=self.models.infer(role,messages,self.budget,charged=True);(folder/'raw_response.txt').write_text(raw+'\n')
        decision=validate_decision(raw,state,role)
        if role=='gaps' and decision['action_id']=='repair':
            p=decision['parameters'];card=state['tool_map'][p['tool_id']]['card']
            if card['operation']=='occluder_polygons':
                canonical,audit=construct_polygon_request(p['tool_parameters'],original_image_sizes(images))
                write_json(folder/'polygon_request_audit.json',audit)
                p['tool_parameters']=canonical
        decoded,_=decode_attachment_response(raw)
        raw_action=json.loads(decoded)['action_id']
        if raw_action!=decision['action_id']:
            self.log(dict(role='host',operation='canonicalize_action_envelope',raw_action_id=raw_action,
                canonical_action_id=decision['action_id'],tool_id=decision['parameters']['tool_id'],
                reason='Operation name exactly matches the selected version-bound card; tool selection and parameters unchanged.'))
        return decision

    def attach(self,candidate,inventory,cached_candidate_id=None,construct=False):
        previous=deepcopy(self.attachment)
        images=self.evidence(candidate,attachment=inventory)
        state=self.attachment_specialist.state(inventory,self.budget,images)
        if cached_candidate_id:
            decision=dict(scene_version=state['scene_version'],action_id='attach',
                parameters=dict(candidate_id=cached_candidate_id,physical_swage_identified=True),
                evidence_ids=['image.left','image.right'],
                reason='Reuse the logged visual swage hypothesis on unchanged stereo/needle: unique eligible endpoint remains within 0.5mm and 2px in each view; rebuild against current geometry.')
            validate_decision(json.dumps(decision),state,'attachment')
        elif self.controller=='heuristic':
            # Proximity gates cannot independently identify the physical swage.
            decision=dict(scene_version=state['scene_version'],action_id='defer',parameters={},
                evidence_ids=['image.left','image.right'],reason='Fixed heuristic has no verified semantic swage detector; proximity alone is insufficient.')
        else:decision=self.model_decision('attachment',state,images)
        result=dict(status='deferred',decision=decision,hypothesis_not_ground_truth=True,
            decision_source='cached_visual_hypothesis' if cached_candidate_id else self.controller)
        if decision['action_id']=='attach':
            row=select_attachment_candidate(inventory,decision['parameters']['candidate_id'],
                physical_attachment_endpoint_confirmed=True,evidence_ids=decision['evidence_ids'])
            part=load_parts(candidate['fragment_path'])[row['fragment']]
            with np.load(self.needle,allow_pickle=False) as source:needle={k:source[k].copy() for k in source.files}
            inferred=part['geometry_inferred']|part['depth_inferred']
            geometry=dict(points_camera_m=part['points_m'],points_newton_m=transform(part['points_m'],CAMERA_TO_NEWTON),
                needle_points_camera_m=needle['points_m'],needle_points_newton_m=transform(needle['points_m'],CAMERA_TO_NEWTON),
                inferred=inferred,needle_inferred=needle['inferred'],observed=~inferred,reliability=(~inferred).astype(float),
                T_newton_from_camera=CAMERA_TO_NEWTON,K=np.asarray(inventory['K']))
            connected,metadata=prepare_connected_geometry(geometry,row,self.tools.case['baseline_m'])
            folder=self.out/f'attachment_{self.snapshot:03d}_{len(self.history):03d}';folder.mkdir()
            if construct:
                self.budget.consume('tool');np.savez_compressed(folder/'connected_fragment.npz',**connected)
            result.update(status='geometric_attachment_proposal' if construct else 'attachment_identity_hypothesis',selection=row,
                metadata=metadata if construct else None,artifact=str(folder/'connected_fragment.npz') if construct else None,
                geometry_constructed=construct,physics_validated=False,training_eligible=False,
                source_geometry_sha256=sha256(candidate['fragment_path']),source_attachment_inventory=inventory['scene_version'])
            endpoint=next(e for e in inventory['endpoints']['thread'] if e['endpoint_id']==row['thread_endpoint_id'])
            result['source_thread_endpoint']={k:endpoint[k] for k in ('points_camera_m','left_xy','right_xy')}
            result['semantic_anchor']=deepcopy(previous.get('semantic_anchor',previous['source_thread_endpoint'])) if cached_candidate_id else deepcopy(result['source_thread_endpoint'])
            result['semantic_decision']=deepcopy(previous.get('semantic_decision',previous['decision'])) if cached_candidate_id else deepcopy(decision)
            result['source_image_hashes']={s:sha256(self.tools.original['inputs'][s]) for s in ('left','right')}
            result['source_needle_hash']=sha256(self.needle)
            self.reserved=[part['points_m'][row['thread_endpoint_index']].tolist()]
            write_json(folder/'report.json',result)
        self.attachment=result;self.log(dict(role='attachment',result=result,budget=asdict(self.budget)))

    def reusable_attachment(self,old,inventory):
        if not old or old.get('source_image_hashes')!={s:sha256(self.tools.original['inputs'][s]) for s in ('left','right')}:return None
        if old.get('source_needle_hash')!=sha256(self.needle):return None
        saved=old.get('semantic_anchor',old.get('source_thread_endpoint'))
        if not saved:return None
        good=[]
        for row in inventory['candidates']:
            if not row['eligible_for_specialist_selection'] or row['needle_endpoint_index']!=old['selection']['needle_endpoint_index']:continue
            endpoint=next(e for e in inventory['endpoints']['thread'] if e['endpoint_id']==row['thread_endpoint_id'])
            if np.linalg.norm(np.asarray(endpoint['points_camera_m'])-saved['points_camera_m'])>.0005:continue
            if any(np.linalg.norm(np.asarray(endpoint[s+'_xy'])-saved[s+'_xy'])>2. for s in ('left','right')):continue
            good.append(row['candidate_id'])
        return good[0] if len(good)==1 else None

    def heuristic_gap(self,state):
        actions=state['tool_map'];p=state['pending']
        useful=[(alias,x) for alias,x in actions.items() if x['card']['operation']=='smooth_connection'
                or x['card']['operation']=='inferred_connection' and x['card']['parameters']['inference_mode']=='boundary']
        if not useful:
            useful=sorted([(alias,x) for alias,x in actions.items() if x['card']['operation']!='inferred_connection'],key=lambda x:(
                ['cross_view_recovery','topology_paths','occluder_mask','occluder_polygons','stereo_mask','resegment_thread'].index(x[1]['card']['operation']),
                x[1]['action_id']))
        for alias,row in useful:
            params={}
            if row['card']['operation']=='occluder_polygons':
                try:params=heuristic_polygons(self.tools)
                except (ValueError,FileNotFoundError):continue
            focus=[p[0]['focus_id']]
            parameters=dict(tool_id=alias,tool_parameters=params)
            if row['card']['operation'] not in ('smooth_connection','inferred_connection'):parameters['focus_ids']=focus
            return dict(scene_version=state['scene_version'],action_id='repair',
                parameters=parameters,
                evidence_ids=['image.left','image.right'],reason='Frozen rule order: feasible pair, alternate-view reconstruction, then perception repairs.')
        return dict(scene_version=state['scene_version'],action_id='assess',parameters=dict(assessments=[
            dict(focus_id=x['focus_id'],classification='defer',reason='No remaining supported shared repair under fixed heuristic.') for x in p]),
            evidence_ids=['image.left','image.right'],reason='Preserve each unresolved endpoint separately.')

    def run(self):
        start=time.perf_counter();needle_before=sha256(self.needle)
        candidate,initial,automatic=self.tools.reconstruct_first(self.out/'initial_reconstruction')
        if automatic:self.attempted.add(self.tools.action_key(automatic))
        inv,att=self.observe(candidate)
        self.log(dict(role='host',operation='reconstruct_first',result=initial,budget=asdict(self.budget)))
        images=self.evidence(candidate,inventory=inv)
        tasks=['needle_attachment']+['g'+str(i) for i,_ in enumerate(self.ledger.pending())]
        plan_state=bind_state(dict(task_ids=tasks,tasks=[dict(focus_id='g'+str(i),kind=self.ledger.entries[key]['gap']['kind'])
            for i,key in enumerate(self.ledger.pending())],budget=asdict(self.budget)),images)
        try:
            plan=(self.model_decision('orchestrator',plan_state,images) if self.controller=='agent' else
                dict(action_id='plan',parameters={'priority':tasks},reason='Deterministic coverage plan.'))
            self.log(dict(role='orchestrator',decision=plan,budget=asdict(self.budget)))
            # Attachment is assessed before a join to prevent consuming its
            # material endpoint. Main ordering determines the gap task order.
            priorities=plan['parameters']['priority'];order={key:i for i,key in enumerate(priorities)}
            ordered=list(self.ledger.entries.items())
            aliases={'g'+str(i):key for i,key in enumerate(self.ledger.pending())}
            rank={aliases[a]:order[a] for a in aliases}
            self.ledger.entries=dict(sorted(ordered,key=lambda item:rank.get(item[0],-1)))
        except Exception as error:
            self.log(dict(role='orchestrator',status='invalid_plan',error=f'{type(error).__name__}: {error}',budget=asdict(self.budget)))
            # Invalid main output is not a fabricated agent plan. The host keeps
            # complete inventory coverage and records its deterministic fallback.
        try:self.attach(candidate,att)
        except Exception as error:self.log(dict(role='attachment',status='failed',error=f'{type(error).__name__}: {error}',budget=asdict(self.budget)))
        while self.ledger.pending():
            if self.controller=='agent' and self.budget.model_calls>=self.budget.max_model_calls:break
            images=self.evidence(candidate,inventory=inv)
            state=None
            try:
                state=self.gap_specialist.state(inv,self.ledger,self.tools,candidate,
                    self.attempted|self.tools.automatic_keys,self.budget,self.reserved,images,self.feedback)
                if self.attachment and self.attachment['status']=='attachment_identity_hypothesis' and self.budget.tool_calls>=self.budget.max_tool_calls-1:
                    state['tools']=[];state['tool_map']={};state['final_attachment_slot_reserved']=True
                    state=bind_state({k:v for k,v in state.items() if k not in ('scene_version','full_state_sha256','evidence_ids')},images)
                decision=(self.model_decision('gaps',state,images) if self.controller=='agent' else self.heuristic_gap(state))
                validate_decision(json.dumps(decision),state,'gaps')
                if decision['action_id']=='assess':
                    self.ledger.assess(state['focus_map'],decision['parameters']['assessments'],decision,state['scene_version'])
                    self.log(dict(role='gaps',decision=decision,result='local_assessments_recorded',budget=asdict(self.budget)))
                    continue
                self.budget.consume('tool');p=decision['parameters'];chosen=state['tool_map'][p['tool_id']]
                card=deepcopy(chosen['card'])
                if card['operation']=='occluder_polygons':card['parameters']=p['tool_parameters']
                attempted_key=self.tools.action_key(card)
                if attempted_key in self.attempted:raise ValueError('Repeated unchanged tool request')
                # Polygon requests share one opportunity per case; feedback may
                # motivate other tools rather than polygon trial-and-error.
                self.attempted.add(attempted_key)
                if card['operation']=='occluder_polygons':self.attempted.add(self.tools.action_key(chosen['card']))
                folder=self.out/f'tool_{self.budget.tool_calls:03d}'
                snapshot={name:deepcopy(getattr(self.tools,name)) for name in
                    ('case','version','working_path','routed','last_audit','automatic_keys')}
                try:proposal,audit=self.tools.execute_repair(chosen['action_id'],card,candidate,folder,
                    selection_evidence=dict(controller=self.controller,evidence_ids=decision['evidence_ids'],reason=decision['reason']))
                except Exception:
                    for name,value in snapshot.items():setattr(self.tools,name,value)
                    raise
                passed=bool(proposal and proposal['report']['numerical_checks_passed'])
                committed=proposal is candidate or passed or candidate is None or not candidate['report']['numerical_checks_passed']
                retained_assembly=False
                if (card['operation']=='cross_view_recovery' and passed and candidate and candidate['report']['numerical_checks_passed']
                    and proposal['report']['fragment_count']>candidate['report']['fragment_count']):
                    # A default re-lift of unchanged perception cannot erase a
                    # checked join. Revised masks remain a new hypothesis route.
                    old_inputs=snapshot['case']['inputs'];new_inputs=self.tools.case['inputs']
                    same_masks=all(sha256(old_inputs['thread_'+s+'_mask'])==sha256(new_inputs['thread_'+s+'_mask']) for s in ('left','right'))
                    if same_masks:committed=False;retained_assembly=True
                if committed:candidate=proposal
                else:
                    for name,value in snapshot.items():setattr(self.tools,name,value)
                self.feedback=dict(tool_id=p['tool_id'],operation=card['operation'],numerical_checks_passed=passed,
                    candidate_committed=committed,failures=proposal['report']['failures'] if proposal else [],
                    prior_checked_assembly_retained=retained_assembly,
                    feedback_images=(audit or {}).get('feedback_images',{}))
                if card['operation'] in ('smooth_connection','inferred_connection'):
                    self.feedback['host_bound_endpoint_ids']=card['parameters']['proposal_id'].split('__')[:2]
                self.log(dict(role='gaps',decision=decision,tool=card,result=self.feedback,audit=audit,budget=asdict(self.budget)))
                # A committed replacement invalidates attachment endpoint IDs.
                # Reserve the physical point only if it still is a terminal.
                inv,att=self.observe(candidate)
                if self.attachment and self.attachment.get('status') in ('geometric_attachment_proposal','attachment_identity_hypothesis') and (
                    self.attachment['source_geometry_sha256']!=(sha256(candidate['fragment_path']) if candidate else None)
                    or self.attachment['source_attachment_inventory']!=att['scene_version']):
                    old_attachment=deepcopy(self.attachment)
                    self.attachment['status']='invalidated_by_reconstruction'
                    self.attachment['physics_validated']=False
                    self.log(dict(role='host',operation='invalidate_attachment',reason='Source geometry/evidence changed; stale attachment artifact cannot be reused.'))
                    reuse=self.reusable_attachment(old_attachment,att)
                    if reuse:
                        try:self.attach(candidate,att,reuse)
                        except Exception as error:self.log(dict(role='attachment',status='failed',error=f'{type(error).__name__}: {error}'))
                    elif self.controller=='agent' and self.budget.model_calls<self.budget.max_model_calls:
                        try:self.attach(candidate,att)
                        except Exception as error:self.log(dict(role='attachment',status='failed',error=f'{type(error).__name__}: {error}'))
                if self.reserved:
                    endpoints=[e['points_m'] for e in inv['endpoints_3d']]
                    if not all(any(np.linalg.norm(np.asarray(p)-q)<1e-8 for q in endpoints) for p in self.reserved):
                        self.reserved=[]
                        if self.attachment:self.attachment['status']='invalidated_by_reconstruction';self.attachment['physics_validated']=False
                        self.log(dict(role='host',operation='invalidate_attachment',reason='Source endpoint changed; previous attachment must not be reused.'))
            except Exception as error:
                self.feedback=dict(status='failed',error=f'{type(error).__name__}: {error}',
                    repair_contract='action_id is repair or assess. Pair repairs use tool_id and tool_parameters only; endpoint IDs are host-bound. Other repairs require current focus_ids.')
                self.log(dict(role='gaps',status='failed',error=f'{type(error).__name__}: {error}',budget=asdict(self.budget)))
                if self.budget.tool_calls>=self.budget.max_tool_calls:break
                if self.controller=='heuristic':break
                if state is None:break
                if str(error).startswith('Prompt bound:'):break
        # Reassess an invalidated/deferred attachment on changed evidence, once,
        # only while budget remains. Never silently reuse old indices.
        if self.controller=='agent' and self.budget.model_calls<self.budget.max_model_calls and self.snapshot>1 and (
                not self.attachment or self.attachment['status'] not in ('geometric_attachment_proposal','attachment_identity_hypothesis')):
            try:self.attach(candidate,att)
            except Exception as error:self.log(dict(role='attachment',status='failed',error=f'{type(error).__name__}: {error}',budget=asdict(self.budget)))
        if self.attachment and self.attachment['status']=='attachment_identity_hypothesis':
            reuse=self.reusable_attachment(self.attachment,att)
            if reuse and self.budget.tool_calls<self.budget.max_tool_calls:
                try:self.attach(candidate,att,reuse,construct=True)
                except Exception as error:self.log(dict(role='attachment',status='final_construction_failed',error=f'{type(error).__name__}: {error}'))
        try:completion=self.ledger.finish()
        except ValueError:completion=dict(all_current_endpoints_assessed=False,unassessed_count=len(self.ledger.pending()))
        write_json(self.out/'endpoint_ledger.json',self.ledger.entries)
        summary=dict(schema='scene_orchestrator_v7',controller=self.controller,case_id=self.tools.case['case_id'],
            initial_reconstruction=initial,candidate=candidate,attachment=self.attachment,completion=completion,
            status='assessed' if completion['all_current_endpoints_assessed'] else 'budget_or_failure_incomplete',
            budget=asdict(self.budget),model_accounting=self.models.accounting() if self.models else None,
            runtime_seconds=time.perf_counter()-start,history_path=str(self.out/'history.json'),
            endpoint_ledger_path=str(self.out/'endpoint_ledger.json'),
            independent_accuracy_measured=False,physics_validated=False,training_eligible=False,
            baseline_geometry_preserved=True,main_plan_is_runtime_code=True)
        if sha256(self.needle)!=needle_before:raise ValueError('Protected common needle changed')
        summary['needle_provenance']=dict(path=str(self.needle),sha256=needle_before,modified=False)
        self.tools.assert_unchanged(candidate);write_json(self.out/'summary.json',summary);return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('case','pipeline','needle','out'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--controller',choices=['heuristic','agent'],required=True)
    parser.add_argument('--cache',type=Path);parser.add_argument('--key-file',type=Path)
    parser.add_argument('--max-model-calls',type=int,default=12);parser.add_argument('--max-tool-calls',type=int,default=10)
    args=parser.parse_args()
    if not 1<=args.max_model_calls<=20 or not 1<=args.max_tool_calls<=12:raise ValueError('Pilot caps: 20 model calls,12 tools maximum')
    models=RoleModels(args.out.parent/(args.out.name+'_model_requests'),args.key_file) if args.controller=='agent' else None
    run=SceneOrchestrator(args.case,args.pipeline,args.needle,args.out,args.controller,models,args.cache,
        Budget(args.max_model_calls,args.max_tool_calls)).run()
    print(json.dumps({k:run[k] for k in ('case_id','status','budget','completion','model_accounting')},allow_nan=False))


if __name__=='__main__':main()
