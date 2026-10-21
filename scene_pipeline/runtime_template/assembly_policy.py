"""Keep assembly distinct from endpoint bookkeeping, with bounded reopening."""
import hashlib,json
ASSEMBLY_OPERATIONS={'smooth_connection','inferred_connection','coincident_endpoint_stitch','cluster_topology'}

def reopen_assembly(ledger,candidate,tools,attempted,budget,seen):
 if not candidate or candidate['report'].get('fragment_count',0)<=1:return None
 if budget.model_calls>=budget.max_model_calls or budget.tool_calls>=budget.max_tool_calls:return None
 cards,_=tools.menu(candidate,attempted|tools.automatic_keys)
 possible={key:card for key,card in cards.items() if card['operation'] in ASSEMBLY_OPERATIONS}
 if not possible:return None
 signature=hashlib.sha256(json.dumps([candidate['fragment_path'],sorted(tools.action_key(c) for c in possible.values())],sort_keys=True).encode()).hexdigest()
 if signature in seen:return None
 seen.add(signature);reopened=[]
 for key,row in ledger.entries.items():
  if row['status']=='deferred':
   row['status']='pending';row['reopened_for_assembly']=True;reopened.append(key)
 return dict(available_assembly_cards=list(possible),reopened_gap_ids=reopened,previous_assessments_preserved=True) if reopened else None

def required_evidence_tools(state):
 depth=[t['tool_id'] for t in state.get('tools',[]) if t['operation']=='stereo_depth_repair']
 if depth and any('depth' in f for f in (state.get('current_geometry') or {}).get('failures',[])):return depth
 if any(t['operation'] in ASSEMBLY_OPERATIONS for t in state.get('tools',[])):return []
 interior={e.get('tool_endpoint_id') for f in state['pending'] if f['kind']=='interior_endpoint' for e in f.get('endpoints',[]) if e.get('tool_endpoint_id')}
 background={code for code,name in state.get('failure_code_legend',{}).items() if name.endswith('_visible_background_contradiction')}
 blocked=any(interior.intersection(p['tool_endpoint_ids']) and any(set(s[1]) and set(s[1])<=background for s in p['scales']) for p in state.get('pairings',[]))
 return [t['tool_id'] for t in state['tools'] if t['operation'] in ('occluder_mask','occluder_polygons')] if blocked else []
