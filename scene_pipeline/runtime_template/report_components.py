"""Execution completion never implies successful scene conversion."""
from pathlib import Path
import json

def read(p):return json.loads(Path(p).read_text()) if Path(p).exists() else {}
def report(folder,root):
 folder=Path(folder);root=Path(root);rows={}
 for method in ('heuristic','agent'):
  summary=read(folder/method/'summary.json');candidate=summary.get('candidate') or {};geometry=candidate.get('report',{});count=geometry.get('fragment_count',0);attachment=summary.get('attachment') or {}
  needle=summary.get('needle_provenance') or {};context=root/'export_context'/method/folder.name
  needle_path=needle.get('path') or (str(context/'common_needle.npz') if (context/'common_needle.npz').exists() else None)
  package=root/'all_geometry_export/advisor_handoff'/folder.name/method
  scene=read(package/'scene.json');tools=scene.get('visual_tools',[])
  rows[method]=dict(thread_available=bool(candidate),thread_checked=geometry.get('numerical_checks_passed',False),thread_fragment_count=count,thread_assembly_complete=bool(count==1 and geometry.get('numerical_checks_passed')),thread_failures=geometry.get('failures',[]),needle_available=bool(needle_path and Path(needle_path).exists()),attachment_proposed=attachment.get('status')=='geometric_attachment_proposal',tool_meshes=sum(bool(t.get('geometry')) for t in tools),accepted_tool_hypotheses=sum(bool(t.get('accepted_tool_hypothesis')) for t in tools),all_geometry_export_verified=(package/'package_verified.json').exists(),independent_accuracy=None,physics_validated=False)
 stages={p.stem:read(p) for p in (folder/'stages').glob('*.json')}
 result=dict(case_id=folder.name,processing_terminal=all(v.get('state') in ('complete','failed') for v in stages.values()),failed_stages=[k for k,v in stages.items() if v.get('state')=='failed'],controllers=rows,stage_timing_seconds={k:v.get('elapsed_seconds') for k,v in stages.items()},conversion_is_not_processing_completion=True)
 (folder/'component_report.json').write_text(json.dumps(result,indent=2)+'\n');return result
