"""Evidence repair is mandatory only when binding has no admissible tool.

Alternative rejected pairings do not veto other admissible connections. The
host menu/verifier still controls every numerical and ambiguity constraint.
"""
def required_evidence_tools(state):
    pair_operations={'smooth_connection','inferred_connection','coincident_endpoint_stitch'}
    if any(tool.get('operation') in pair_operations for tool in state.get('tools',[])):
        return []
    interior={e.get('tool_endpoint_id') for f in state['pending'] if f['kind']=='interior_endpoint'
              for e in f.get('endpoints',[]) if e.get('tool_endpoint_id')}
    background={code for code,name in state.get('failure_code_legend',{}).items()
                if name.endswith('_visible_background_contradiction')}
    blocked=False
    for pair in state.get('pairings',[]):
        if not interior.intersection(pair['tool_endpoint_ids']):continue
        if any(set(scale[1]) and set(scale[1])<=background for scale in pair['scales']):
            blocked=True;break
    if not blocked:return []
    return [t['tool_id'] for t in state['tools']
            if t['operation'] in ('occluder_mask','occluder_polygons')]
