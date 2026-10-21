"""Expose only repairs that can satisfy the unchanged shared verifier.

This common capability gate cannot certify scene accuracy. It excludes native
QP cap changes that cannot create required join provenance, and endpoint-only
joins that cannot yield a two-terminal chain. The VLM remains free to choose
among feasible hypotheses or defer; the same gate applies to the heuristic.
"""
from compact_agent_protocol import digest
from compact_agent_protocol_v3_1 import (
    make_state as state_v3_1, diagnosis, image_evidence, validate_decision,
    resolve_action, shared_request, signature, heuristic_action, messages_for,
)

PROTOCOL = 'surgical_refinement_v3_2'


def make_state(*args, **kwargs):
    state, catalog = state_v3_1(*args, **kwargs)
    observation = state['observation']
    components = observation.get('thread_component_count')
    endpoints = observation.get('thread_endpoint_count')
    failures = (state.get('current_result') or {}).get('failures', [])
    # verify_native_qp cannot authorize a multi-component bridge: after=before,
    # joins=[] and missing_native_join_provenance is mandatory. A changed radius
    # cannot fix this contract. No verifier or physical threshold is changed.
    if 'qp.reported_gap' in state['actions'] and 'missing_native_join_provenance' in failures:
        card = state['actions'].pop('qp.reported_gap')
        state['excluded_actions'].append(dict(operation=card['operation'], parameters=card['parameters'],
            reason='native_qp_cap_cannot_supply_required_join_provenance'))
    # Joining C open fragments needs C-1 endpoint pairs and removes 2(C-1)
    # terminals. To end with exactly two terminals, E must equal 2C. Endpoint
    # joins cannot delete branches or cut a cycle. Counts come from the shared
    # retained skeleton inspection, not model confidence or case identifiers.
    if (isinstance(components, int) and isinstance(endpoints, int) and components > 0
            and endpoints != 2 * components):
        for aid in list(state['actions']):
            card = state['actions'][aid]
            if card['operation'] in ('thread_occlusion', 'thread_qp'):
                state['actions'].pop(aid)
                state['excluded_actions'].append(dict(operation=card['operation'], parameters=card['parameters'],
                    reason='endpoint_only_join_cannot_produce_two_terminal_chain',
                    retained_components=components, retained_endpoints=endpoints,
                    required_endpoints_for_join_only_repair=2 * components))
    repair_cards = [c for c in state['actions'].values() if c['operation'] in ('thread_occlusion', 'thread_qp')]
    if not repair_cards:
        state['actions'].pop('inspect', None)
    state.update(protocol=PROTOCOL, choice_is_constrained_to_defer=list(state['actions']) == ['defer'])
    state['legal_operations'] = list(dict.fromkeys(c['operation'] for c in state['actions'].values()))
    state['diagnosis'] = diagnosis(state)
    state['diagnosis']['excluded_actions'] = state['excluded_actions']
    state['scene_version'] = digest({'parent_version': state['scene_version'], 'state': state, 'catalog': catalog})
    return state, catalog
