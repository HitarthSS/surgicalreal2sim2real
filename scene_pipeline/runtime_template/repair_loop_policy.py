"""Schedule real repairs before endpoint bookkeeping; never fabricate a join."""
import hashlib
import json


def evidence_version(inventory):
    # Geometry changes do not invalidate a prior classification of unchanged
    # image-mask noise. Mask, calibration, depth and source changes still do.
    paths={k:v for k,v in inventory.get('provenance',{}).items() if k!='candidate_fragments'}
    return hashlib.sha256(json.dumps(paths,sort_keys=True).encode()).hexdigest()


def focus_groups(inventory,ledger,limit=8):
    """Group only nonmetric short mask features, preserving every ledger ID.

    A grouped decision may defer uncertain evidence; it cannot certify physical
    terminals or erase geometry. Metric endpoints always remain separate.
    """
    pending=ledger.pending();rows=ledger.entries
    components={e['endpoint_id']:e.get('component_id',e['endpoint_id']) for e in inventory.get('endpoints_2d',[])}
    endpoint_groups={e:g['group_id'] for g in inventory.get('uncertainty_groups',[])
                     for e in g['endpoint_ids']}
    groups={};singles=[]
    for key in pending:
        gap=rows[key]['gap'];ids=gap['endpoint_ids']
        if ids and all(e in endpoint_groups for e in ids) and gap['kind']!='topology_ambiguity':
            group=('short_mask',*sorted({endpoint_groups[e] for e in ids}))
            groups.setdefault(group,[]).append(key)
        elif ids and all(e in components for e in ids):
            group=('image_component',*sorted({components[e] for e in ids}))
            groups.setdefault(group,[]).append(key)
        else:singles.append(key)
    def rank(key):
        gap=rows[key]['gap']
        metric=any(e.startswith('endpoint3d.') for e in gap['endpoint_ids'])
        return (0 if metric and gap['kind']=='interior_endpoint' else 1 if metric else 2)
    result=[[k] for k in sorted(singles,key=rank)]+list(groups.values())
    return {f'g{i}':members[0] if len(members)==1 else members for i,members in enumerate(result if limit is None else result[:limit])}


def required_evidence_tools(state):
    """Route background-only connection failures to fresh occlusion evidence.

    Shape, gap, depth, intersection and out-of-frame failures are never waived.
    Only an interior metric endpoint is a mandatory repair dependency.
    """
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


def validate_assessment_dependency(decision,state):
    if decision['action_id']!='assess' or not state.get('required_evidence_tool_ids'):return
    focus={x['focus_id']:x for x in state['pending']}
    for value in decision['parameters'].get('assessments',[]):
        row=focus.get(value['focus_id'],{})
        if (value['classification']=='defer' and row.get('kind')=='interior_endpoint'
                and any(e.get('tool_endpoint_id') for e in row.get('endpoints',[]))):
            raise ValueError('Interior connection is blocked by missing occlusion evidence: use one of '+
                             ','.join(state['required_evidence_tool_ids'])+' then recompute pairs before deferring')


def reconstruction_completion(candidate,assessment_complete):
    report=(candidate or {}).get('report',{})
    count=report.get('fragment_count');checked=bool(report.get('numerical_checks_passed'))
    return dict(single_checked_thread=bool(checked and count==1),fragment_count=count,
                checked_visible_geometry=checked,endpoint_assessment_complete=assessment_complete,
                partial_reconstruction=bool(count and count>1),
                assessment_is_not_reconstruction_completion=True)


def preserve_fallback_assessments(source,tools):
    """Do not erase valid endpoint decisions when the best geometry stays put.

    Exact evidence equality is enough. New bounded occlusion-only evidence can
    also survive geometry fallback using the same existing retention contract.
    Revised thread masks require the original matching bundle on fallback.
    """
    from reconstruction_tools import sha256
    from evidence_transition import retain_occlusion_evidence
    old=source.tools
    if (old.case==tools.case and all(sha256(old.case['inputs'][k])==sha256(v)
                                  for k,v in tools.case['inputs'].items())):
        return True,'unchanged_evidence_assessments_preserved'
    snapshot={k:getattr(old,k) for k in ('case','version','working_path','routed','last_audit','automatic_keys')}
    keep,why=retain_occlusion_evidence(snapshot,tools,None,None,'occluder_mask',dict(no_op=False))
    return keep,why
