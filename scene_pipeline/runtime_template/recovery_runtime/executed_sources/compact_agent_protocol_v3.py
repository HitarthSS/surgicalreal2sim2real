"""Select exact, version-bound action cards instead of rewriting tool defaults.

The common menu does not choose an action. The model and scene-aware heuristic
receive the same cards, evidence and bounds. Rejected requests are never repaired.
"""
import math

from compact_agent_protocol import accepted_target, digest, encoded, signature
from compact_agent_protocol_v2 import (image_evidence, make_state as state_v2,
                                      validate_decision as validate_v2)
from reconstruction_tools import TOOLS, parameters_for, strict_json

PROTOCOL = 'surgical_refinement_v3'
FIELDS = {'scene_version', 'target', 'action_id', 'evidence_ids', 'reason'}


def action_cards(observation, target, remaining, result, attempted, feedback):
    """Generic causal proposals; no case IDs, outcome lookups or relaxed gates."""
    cards = {'defer': {'operation': 'defer', 'parameters': {},
                      'effect': 'Preserve unresolved geometry and state the missing evidence or unsupported repair.'}}
    excluded = []
    if remaining['model_decisions'] <= 0:
        return {}, [{'reason': 'model_budget_exhausted'}]
    if accepted_target(result, target):
        if feedback and feedback['blocking_for_finish']:
            excluded.append({'operation': 'target_reconstruction',
                'reason': 'Target already passes; current failure belongs to assembly/attachment. '
                          'No authorized target operation creates a missing tool fit or extends hidden needle geometry.'})
        else:
            cards['finish'] = {'operation': 'finish', 'parameters': {},
                               'effect': 'Finish the checked target/preflight only; physics remains unvalidated.'}
        return cards, excluded
    if remaining['tool_calls'] <= 0:
        return cards, [{'reason': 'reconstruction_budget_exhausted'}]
    used = {signature(a) for a in attempted}
    available = set(observation.get('available_tools', []))

    def propose(cid, op, params, effect, basis):
        effective = parameters_for(op, params)
        if signature({'operation': op, 'parameters': effective}) in used:
            excluded.append({'operation': op, 'parameters': effective, 'reason': 'already_attempted_effective_request'})
            return
        cards[cid] = {'operation': op, 'parameters': effective, 'effect': effect, 'basis': basis}

    if target != 'thread':
        return cards, [{'reason': 'This pilot refines thread only; needle estimator scope is unchanged.'}]
    if 'thread_qp' in available:
        if not any(a['operation'] == 'thread_qp' for a in attempted):
            propose('qp.default', 'thread_qp', {}, 'Run the unchanged production baseline.', 'baseline_not_attempted')
        stops = ((result or {}).get('backend_diagnostics') or {}).get('native_ordering_stops', [])
        gaps = [s.get('nearest_gap_px_reported') for s in stops]
        gaps = [g for g in gaps if isinstance(g, (int, float)) and not isinstance(g, bool) and math.isfinite(g)]
        if gaps:
            cap = math.ceil(max(gaps)) + 1  # Native reports are rounded; explicit one-pixel margin.
            _, low, high = TOOLS['thread_qp']['bounds']['connection_radius_px']
            if low <= cap <= high:
                propose('qp.reported_gap', 'thread_qp', {'connection_radius_px': cap},
                    'Change only the native ordering join cap. Cannot repair a branch, segmentation, depth or attachment failure. '
                    'Original visibility/topology/provenance verifiers still decide acceptance.',
                    {'native_reported_gaps_px': gaps, 'rounding_margin_px': 1, 'not_join_authorization': True})
            else:
                excluded.append({'operation': 'thread_qp', 'reason': 'reported_gap_exceeds_existing_parameter_bound',
                                 'required_cap_px': cap, 'existing_maximum_px': high})
        else:
            excluded.append({'operation': 'thread_qp', 'reason': 'no_native_join_stop_supports_a_changed_join_cap'})
    if ('thread_occlusion' in available and observation.get('thread_component_count', 0) > 1
            and observation.get('left_occluder_pixels', 0) > 0):
        propose('occlusion.default', 'thread_occlusion', {},
            'Test fragment joins under occluder, stereo, endpoint direction, metric-gap and ambiguity checks. '
            'Does not resolve cropped continuation or create tool geometry.', 'multiple_fragments_and_supplied_occluder')
    else:
        excluded.append({'operation': 'thread_occlusion', 'reason': 'no_separate_fragments_with_supplied_occluder'})
    if 'thread_distance' in available:
        excluded.append({'operation': 'thread_distance', 'reason': 'diagnostic_reference_not_a_production_repair_in_this_pilot'})
    # Inspect only when an actual repair alternative remains and another decision
    # can use the result. A lone defer menu does not waste a read.
    if len(cards) > 1 and remaining['evidence_reads'] > 0 and remaining['model_decisions'] > 1:
        cards['inspect'] = {'operation': 'inspect_evidence', 'parameters': {},
                            'effect': 'Read up to three unshown detailed records before the final decision.'}
    return cards, excluded


def make_state(observation, provenance, target, remaining, result=None, revision=0, attempted=(),
               images=(), scene_feedback=None, controller_feedback=None):
    state, catalog = state_v2(observation, provenance, target, remaining, result, revision, attempted,
                              images, scene_feedback, controller_feedback)
    cards, excluded = action_cards(observation, target, remaining, result, attempted, scene_feedback)
    state.update(protocol=PROTOCOL, actions=cards, excluded_actions=excluded,
                 decision_fields=sorted(FIELDS), choice_is_constrained_to_defer=list(cards) == ['defer'])
    # Retain the original validator and all its physical/numeric bounds.
    state['legal_operations'] = list(dict.fromkeys(c['operation'] for c in cards.values()))
    state['scene_version'] = digest({'state': state, 'catalog': catalog, 'provenance': provenance, 'result': result})
    return state, catalog


def validate_decision(raw, state, catalog):
    d = strict_json(raw) if isinstance(raw, str) else strict_json(encoded(raw))
    if not isinstance(d, dict) or set(d) != FIELDS:
        raise ValueError('Exactly five action-card fields required: ' + ', '.join(sorted(FIELDS)))
    aid = d['action_id']
    if not isinstance(aid, str) or aid not in state['actions']:
        raise ValueError('Choose an exact currently available action_id; closed/default repeats are not actions')
    card = state['actions'][aid]
    resolved = {k: v for k, v in d.items() if k != 'action_id'} | {
        'operation': card['operation'], 'parameters': card['parameters']}
    validate_v2(resolved, state, catalog)
    return d


def resolve_action(decision, state):
    """Materialize the exact selected card; no inferred replacement or fallback."""
    card = state['actions'][decision['action_id']]
    return {'tool': card['operation'], 'parameters': dict(card['parameters']), 'reason': decision['reason']}


def shared_request(decision, state):
    return resolve_action(decision, state)


def messages_for(state, images, requested_evidence=None):
    if [i for i, _, _ in images] != state['image_evidence_ids']:
        raise ValueError('Images do not match versioned evidence IDs')
    # Full state/catalog remains in the log. Avoid presenting already-spent
    # default schemas as tempting actions or repeating invariant tool bounds.
    prompt_state = {k: v for k, v in state.items() if k not in
                    {'operations', 'control_operations', 'legal_operations', 'decision_fields'}}
    rules = ('Select one exact action_id from actions. Return ONLY one JSON object with five fields: '
        'scene_version, target, action_id, evidence_ids (1-3 exact listed IDs), reason (brief, <=500 characters). '
        'Copy the current version and target. Do not return operation or parameters: each card already fixes '
        'the exact tool and parameters that will execute and be logged. Closed/default repeats are absent. '
        'Use stereo, numerical failures and each card\'s stated effect. A join-cap change only addresses a native '
        'join stop; it cannot solve a branch, crop truncation, missing tool fit or needle attachment gap. '
        'If no supported repair exists, select defer and cite its blockers; do not manufacture a connection. '
        'scene_feedback is a registered evidence ID. Shown images are already available; inspection may cite only '
        'unshown detailed records. Never change needle size, estimator, masks or protected checks. '
        'Input fit and export are not independent accuracy or valid physics.\nSTATE:' + encoded(prompt_state))
    content = [{'type': 'text', 'text': rules}]
    if requested_evidence:
        content.append({'type': 'text', 'text': 'REQUESTED_EVIDENCE:' + encoded(requested_evidence)})
    for eid, label, path in images:
        content += [{'type': 'text', 'text': 'EVIDENCE_ID=' + eid + '; ' + label},
                    {'type': 'image', 'url': str(path)}]
    return [{'role': 'user', 'content': content}]


def heuristic_action(state):
    """A stronger common-menu control; deterministic, transparent, no case IDs."""
    for aid in ('finish', 'occlusion.default', 'qp.reported_gap', 'qp.default', 'defer'):
        if aid in state['actions']:
            return {'scene_version': state['scene_version'], 'target': state['target'], 'action_id': aid,
                    'evidence_ids': ['verification' if state.get('current_result') is not None else 'observation'],
                    'reason': 'Shared-menu heuristic: prefer constrained occlusion join, then diagnosed native cap; defer otherwise.'}
    raise ValueError('No legal heuristic action')
