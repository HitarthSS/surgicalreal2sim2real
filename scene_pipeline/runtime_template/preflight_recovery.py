"""Actual-data CPU regression checks; API disabled and old outputs read-only."""
from pathlib import Path
import sys,os,json,copy
import numpy as np
C=Path(__file__).resolve().parent;D=Path('/data/xliang/ismr_2027/sam3_pipeline10_20261019_v1')
sys.path[:0]=[str(C),str(C/'scene_sources'),str(C/'executed_sources'),str(C/'perception_runtime')]
assert os.environ.get('SLURM_JOB_ID') and os.environ.get('ISMR_NO_API')=='1'
from contract import read,save
from run_scene_orchestrator_v7 import SceneOrchestrator,Budget
from connection_finalize import verified_tool_masks
from fixed_needle_fitter import build_catalog,load_npz,CATALOG
from rigid_needle_contract import audit
from needle_size_stage import geometry_choice
old=Path('/home/xliang/ismr_agent70_20261018_v1/day20261015');checks=[]
cid='snapshot_20261015_145924';base=read(old/cid/'perception/baseline/assessment.json');case=read(base['case_path'])
r=SceneOrchestrator(base['case_path'],C/'pipeline-tools',None,D/'preflight/thread_without_needle',controller='heuristic',budget=Budget(max_model_calls=0,max_tool_calls=0))
summary=r.run();assert summary.get('candidate') and summary['needle_provenance']['path'] is None
checks.append(dict(test='thread_repair_runs_without_needle',passed=True,fragment_count=summary['candidate']['report']['fragment_count']))
selected=read(C/cid/'tools/complete_tools.json')
for item in selected['tools']['heuristic'].values():item['accepted_tool_hypothesis']=False
masks,receipts,rejected=verified_tool_masks(case,C/cid,'heuristic',selected)
assert any(x.any() for x in masks['left']) and any(x.any() for x in masks['right'])
checks.append(dict(test='actual_tool_masks_survive_unaccepted_pose',passed=True,receipts=len(receipts),rejected=rejected))
for short in ('145924','154257'):
 f=old/('snapshot_20261015_'+short);base=read(f/'perception/baseline/assessment.json');case=read(base['case_path'])
 catalog=build_catalog(case,f/'perception/baseline/common_needle.npz',[],D/'preflight'/short/'catalog')
 assert all(audit(load_npz(row['needle_path']),CATALOG,require_pose=True)['usable'] for row in catalog['candidates'])
 row,verified=geometry_choice(catalog)
 checks.append(dict(test='joint_rigid_arc_stereo_refinement',case=short,passed=True,geometry_eligible=verified,
  selected=row['candidate_id'] if row else None,views=row['report']['views'] if row else None,candidates=len(catalog['candidates'])))
save(D/'preflight/verification.json',dict(checks=checks,API_calls=0,GPU_used=False));print(json.dumps(checks,indent=2))
