"""Causal, compact presentation of the unchanged shared action-card interface.

One failed backend is not evidence that every repair must fail. This module
exposes repair hypotheses and separates current-target dependencies from
independent scene blockers. It never chooses the action or relaxes a check.
"""
from compact_agent_protocol import digest, encoded
from compact_agent_protocol_v3 import (
    make_state as state_v3, image_evidence, validate_decision, resolve_action,
    shared_request, signature, heuristic_action,
)

PROTOCOL = 'surgical_refinement_v3_1'


def diagnosis(state):
    result = state.get('current_result') or {}
    failures = result.get('failures', [])
    scene = state.get('scene_feedback') or {}
    assembly = scene.get('assembly', {})
    prerequisites = assembly.get('missing_prerequisites', [])
    target_dependency = 'accepted_' + state['target'] + '_reconstruction'
    hypotheses = []
    for aid, card in state['actions'].items():
        if card['operation'] == 'thread_occlusion':
            addresses = {'missing_native_join_provenance', 'unresolved_fragments',
                         'visible_fragments_not_fully_represented', 'ambiguous_thread_topology'}
            explanation = ('QP stopping between visible fragments is a repair hypothesis, not a permanent failure. '
                           'This different tool generates and verifies occlusion-aware endpoint joins. '
                           'It may still reject ambiguous, unsupported or branched topology.')
        elif card['operation'] == 'thread_qp':
            addresses = {'unresolved_fragments', 'visible_fragments_not_fully_represented'}
            explanation = ('Only tests the reported native ordering gap. Increasing its cap does not '
                           'supply missing join provenance or resolve branch ambiguity.')
        else:
            continue
        hypotheses.append({'action_id': aid, 'addresses_current_failures': [f for f in failures if f in addresses],
                           'hypothesis': explanation, 'success_guaranteed': False})
    return {
        'current_target_failures': failures,
        'assembly_depends_on_current_target': target_dependency in prerequisites,
        'independent_missing_prerequisites': [x for x in prerequisites if x != target_dependency],
        'attachment': scene.get('attachment', {}),
        'repair_hypotheses': hypotheses,
        'no_repair_card_available': not hypotheses,
        'interpretation': ('Repair hypotheses require a tool execution and verification. A failed QP result '
                           'does not prove the occlusion tool fails. A target dependency is reassessed after '
                           'repair; independent missing inputs and oversized attachment gaps stay unresolved.'),
    }


def make_state(*args, **kwargs):
    state, catalog = state_v3(*args, **kwargs)
    state['protocol'] = PROTOCOL
    state['diagnosis'] = diagnosis(state)
    state['scene_version'] = digest({'parent_version': state['scene_version'], 'state': state, 'catalog': catalog})
    return state, catalog


def messages_for(state, images, requested_evidence=None):
    if [i for i, _, _ in images] != state['image_evidence_ids']:
        raise ValueError('Images do not match versioned evidence IDs')
    # All original numerical records and provenance remain in state/catalog logs.
    # Model and heuristic receive the same diagnosis; neither sees old outcomes.
    prompt = {k: state[k] for k in (
        'scene_version', 'target', 'remaining', 'observation', 'current_result',
        'diagnosis', 'evidence_ids', 'inspectable_evidence_ids', 'finish_allowed',
        'choice_is_constrained_to_defer', 'controller_feedback',
    )}
    prompt['actions'] = [dict(action_id=aid, **card) for aid, card in state['actions'].items()
                         if aid != 'defer'] + [dict(action_id='defer', **state['actions']['defer'])]
    rules = (
        'Return one JSON object: scene_version, target, action_id, evidence_ids (1-3 exact IDs), '
        'reason (brief, <=500 characters). Choose one exact card; parameters are fixed and logged. '
        'Diagnose the failed backend, then consider each repair hypothesis. A bounded repair is an '
        'experiment, not a success claim: verification can reject it. Do not treat an unmet current-target '
        'dependency as proof that a different reconstruction tool cannot repair it. If a supported hypothesis '
        'fits the stereo evidence, test it. If evidence is insufficient, inspect relevant unshown records '
        'when budget allows, or defer with the specific unsupported issue. Defer if no repair exists. '
        'Never repeat a closed action, change needle geometry, relax gates or invent attachment. '
        'A successful repair is still subject to scene preflight and later physics.\nSTATE:' + encoded(prompt)
    )
    content = [{'type': 'text', 'text': rules}]
    if requested_evidence:
        content.append({'type': 'text', 'text': 'REQUESTED_EVIDENCE:' + encoded(requested_evidence)})
    for eid, label, path in images:
        content.extend([{'type': 'text', 'text': 'EVIDENCE_ID=' + eid + '; ' + label},
                        {'type': 'image', 'url': str(path)}])
    return [{'role': 'user', 'content': content}]
