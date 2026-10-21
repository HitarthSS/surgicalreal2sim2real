"""Version2: explicit image evidence, legal budgets and scene-level feedback.

The original protocol remains a fallback. Reconstruction tools, bounds and
numerical verifiers are unchanged. No image filename aliases or semantic repairs.
"""
from pathlib import Path

from compact_agent_protocol import (accepted_target, digest, encoded, make_state as legacy_state,
                                    signature, shared_request, token_usage,
                                    validate_decision as legacy_validate)
from reconstruction_tools import fingerprints

PROTOCOL = 'surgical_refinement_v2.1'


def image_evidence(session, target, result):
    images=[('image.left', 'Original rectified left view.', Path(session.inputs['left'])),
            ('image.right', 'Original rectified right view.', Path(session.inputs['right']))]
    if result:
        names=('shared_needle_left_overlay.png','shared_needle_right_overlay.png') if target=='needle' else (
            'shared_left_overlay.png','right_validation_overlay.png')
        for side,name in zip(('left','right'),names):
            path=session.out/result['call_id']/'backend'/name
            if path.exists():
                images.append(('overlay.'+side,'Current '+side+' input-support overlay; not independent accuracy.',path))
    return images


def make_state(observation, provenance, target, remaining, result=None, revision=0, attempted=(),
               images=(), scene_feedback=None, controller_feedback=None):
    for name in ('model_decisions','tool_calls','evidence_reads'):
        value=remaining.get(name,0)
        if isinstance(value,bool) or not isinstance(value,int) or value<0:
            raise ValueError('Invalid remaining budget: '+name)
    state,catalog=legacy_state(observation,provenance,target,remaining,result,revision,attempted)
    image_ids=[i for i,_,_ in images]
    if len(set(image_ids))!=len(image_ids) or set(image_ids)&set(catalog):
        raise ValueError('Duplicate evidence ID')
    image_hashes=fingerprints({i:path for i,_,path in images})
    for i,label,path in images:
        catalog[i]={'kind':'image','description':label,'already_shown':True,**image_hashes[i]}
    if scene_feedback is not None:
        if not isinstance(scene_feedback,dict) or not isinstance(scene_feedback.get('blocking_for_finish'),bool):
            raise ValueError('Scene feedback must explicitly declare its finish gate')
        catalog['scene.assembly']=scene_feedback.get('assembly',{})
        catalog['scene.attachment']=scene_feedback.get('attachment',{})
        catalog['scene.provenance']=scene_feedback.get('_provenance',{})
        # The shown aggregate is an actual evidence record, not an alias for a
        # rejected response. Its canonical ID matches the visible state field.
        catalog['scene_feedback']={k:v for k,v in scene_feedback.items() if k!='_provenance'}
    if controller_feedback is not None:
        catalog['controller.feedback']=controller_feedback
    compact_feedback=None if scene_feedback is None else {k:v for k,v in scene_feedback.items() if k!='_provenance'}
    state.update(protocol=PROTOCOL,scene_feedback=compact_feedback,
                 controller_feedback=controller_feedback,
                 evidence_ids=[k for k in catalog if not k.startswith('candidate.')])
    state['validity'].update(assembly_preflight_passed=bool(scene_feedback and scene_feedback.get('preflight_passed')),
        attachment_created=False)
    state['finish_allowed']=accepted_target(result,target) and not bool(
        scene_feedback and scene_feedback['blocking_for_finish'])
    active=remaining.get('model_decisions',0)>0
    state['operations']=state['operations'] if active and remaining.get('tool_calls',0)>0 else {}
    controls={}
    if active:
        controls['defer']={'purpose':'Preserve unresolved geometry or unavailable prerequisites.','parameters':{}}
        if state['finish_allowed']:
            controls['finish']={'purpose':'Complete the checked target only; not physics or training validation.','parameters':{}}
        # A last-decision inspection cannot inform a subsequent model decision.
        if remaining.get('evidence_reads',0)>0 and remaining.get('model_decisions',0)>1:
            controls['inspect_evidence']={'purpose':'Retrieve up to3 unshown detailed records for the next decision.','parameters':{}}
    state['control_operations']=controls
    state['legal_operations']=list(state['operations'])+list(controls)
    state['image_evidence_ids']=image_ids
    state['inspectable_evidence_ids']=[k for k in catalog if k not in image_ids and k!='controller.feedback'
                                      and not k.startswith('candidate.')]
    state['scene_version']=digest({'state':state,'catalog':catalog,'provenance':provenance,'full_result':result})
    return state,catalog


def validate_decision(raw,state,catalog):
    # The original full-document validator still enforces fields, versions,
    # evidence references and the same numeric tool parameter bounds.
    d=legacy_validate(raw,state,catalog)
    if d['operation'] not in state['legal_operations']:
        raise ValueError('Operation is not currently legal; choose from legal_operations')
    if d['operation']=='finish' and not state['finish_allowed']:
        raise ValueError('Assembly/attachment feedback blocks finish')
    if d['operation']=='inspect_evidence' and any(i not in state['inspectable_evidence_ids']
            and not i.startswith('candidate.') for i in d['evidence_ids']):
        raise ValueError('Images are already shown; inspect only inspectable_evidence_ids')
    if d['operation'] in state['operations']:
        sig=signature(d)
        if any(signature(x)==sig for x in state['attempted_operations']):
            raise ValueError('Duplicate effective reconstruction request; choose different parameters or defer')
    return d


def messages_for(state,images,requested_evidence=None):
    ids=[i for i,_,_ in images]
    if ids!=state['image_evidence_ids']:
        raise ValueError('Prompt images do not match the versioned evidence catalog')
    rules=(
        'Refine the current surgical reconstruction target. Return ONE JSON object with exactly '
        'scene_version (copy current), target, operation, parameters, evidence_ids (1-3 IDs), reason (<=500chars). '
        'Choose operation ONLY from legal_operations. Control parameters are {}. '
        'Cite IDs from evidence_ids or candidate.NNN records; NEVER use filenames or invent IDs. '
        'Images below are already shown and citable by their EVIDENCE_ID. Do not inspect an image again. '
        'inspect_evidence is available only with a read AND a subsequent model decision; '
        'cite only inspectable_evidence_ids or candidate.NNN. The last decision must act or defer. '
        'Use current numeric checks, paired images and scene_feedback (the aggregate evidence ID is scene_feedback; '
        'individual records are scene.assembly and scene.attachment). A reconstruction pass is not a valid assembled scene. '
        'If attachment exceeds the4mm bound, is ambiguous, or a required tool fit is missing, do not claim completion. '
        'If existing target tools cannot resolve the cause, defer and identify it. '
        'Do not repeat attempted effective parameters. Preserve uncertainty and inferred labels. '
        'Do not set needle size, switch estimators, extend hidden endpoints, invent occluders, or lower any check. '
        'Thread distance is diagnostic; original QP is the production baseline. '
        'Mask fit, residuals and model confidence are not independent accuracy; physics and policy remain unvalidated. '
        'Inspect competing candidate records or defer when summaries cannot resolve ambiguity.'
    )
    content=[{'type':'text','text':rules+'\nSTATE:'+encoded(state)}]
    if requested_evidence:
        content.append({'type':'text','text':'REQUESTED_EVIDENCE:'+encoded(requested_evidence)})
    for i,label,path in images:
        content.extend([{'type':'text','text':'EVIDENCE_ID='+i+'; already_shown=true; '+label},
                        {'type':'image','url':str(path)}])
    return [{'role':'user','content':content}]
