"""Stage a four-scene repair revision; old scientific outputs remain immutable."""
from pathlib import Path
import os,json,shutil,hashlib,py_compile
C=Path('/home/xliang/ismr_sam3_pipeline10_20261019_v1');D=Path('/data/xliang/ismr_2027/sam3_pipeline10_20261019_v1')
B=Path('/home/xliang/ismr_agent70_20261018_v1/day20261015');O=Path('/data/xliang/ismr_2027/agent70_20261018_v1/day20261015')
CASES=['snapshot_20261015_'+x for x in ['145924','152035','154257','155222']]
def rd(p):return json.loads(Path(p).read_text())
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cp(a,b):b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b)
assert not (D/'staged.json').exists() and not (D/'started.json').exists()
C.mkdir(exist_ok=True);D.mkdir(parents=True,exist_ok=True)
for p in B.glob('*.py'):cp(p,C/p.name)
for name in ['scene_sources','executed_sources','frozen_sources','pipeline-tools','tool_template','perception_runtime','transport_hook']:
 shutil.copytree(B/name,C/name,dirs_exist_ok=True,symlinks=False,ignore=shutil.ignore_patterns('__pycache__'))
for name in ['recovery_runtime','all_geometry_export']:
 (C/name).mkdir(exist_ok=True)
 for p in (B/name).glob('*.py'):cp(p,C/name/p.name)
 for link in ['executed_sources','tool_template']:(C/name/link).symlink_to(C/link,target_is_directory=True)
for p in (C/'patches').rglob('*.py'):
 cp(p,C/p.relative_to(C/'patches'))
backend=C/'scene_sources/openai_scene_backend.py'
for p in list(C.rglob('openai_scene_backend.py')):
 if 'patches' not in p.parts and p!=backend:cp(backend,p)
for name in ['preregistration.json','job_manifest.json']:cp(B/name,C/name)
cp(B.parent/'split_manifest.json',C/'split_manifest.json')
manifest=rd(C/'job_manifest.json');manifest['cases']=[x.rsplit('_',1)[1] for x in CASES];save(C/'job_manifest.json',manifest)
save(C/'recovery_scope.json',dict(cases=CASES,API_per_case=23,maximum_new_scene_API_attempts=92,health_requests=2,source_run=str(B.parent),scope='four reported scenes; no full70 rerun'))
for name in ['cache','ready','export_context','advisor_handoff']:
 dest=D/name;dest.mkdir(exist_ok=True);(C/name).symlink_to(dest,target_is_directory=True)
for name in ['advisor_handoff','review']:
 dest=D/'all_geometry_export'/name;dest.mkdir(parents=True,exist_ok=True);(C/'all_geometry_export'/name).symlink_to(dest,target_is_directory=True)
for cid in CASES:
 f=D/cid;f.mkdir();(C/cid).symlink_to(f,target_is_directory=True)
 cp(O/cid/'case.json',f/'case.json');cp(O/cid/'K.csv',f/'K.csv')
 marker=rd(O/cid/'ready.json');save(C/'ready'/(cid+'.json'),marker)
 # Preserve only the original deterministic tool baseline; fresh agent decisions
 # may resegment/refit as necessary. Native fits are never repeated gratuitously.
 shutil.copytree(O/cid/'tools',f/'tools',symlinks=True)
 tools=rd(f/'tools/complete_tools.json');save(f/'tools/source_complete_tools.json',tools)
 baseline=tools['tools'].get('heuristic',{});tools['tools']=dict(heuristic=baseline,agent=json.loads(json.dumps(baseline)))
 save(f/'tools/complete_tools.json',tools)
 save(f/'reused_stage_provenance.json',dict(original_case_sha256=sha(O/cid/'case.json'),source_tools_sha256=sha(O/cid/'tools/complete_tools.json'),
  initial_preprocessing_only=True,deterministic_native_tool_baseline_reused=True,prior_agent_outputs_reused=False,
  protected_mesh_hashes={n:sha(v['mesh']) for n,v in baseline.items() if v.get('mesh')}))
# Early source-bound tool-mask evidence can help needle recovery before poses
# are improved. Images, calibration, depth, raw labels and cached masks stay fixed.
p=C/'case_worker.py';s=p.read_text();old="ok=stage('initial_assessment',lambda:per.assess(F/'case.json',ROOT/'pipeline-tools',F/'perception/baseline'))"
new="""from needle_tool_handoff import prepare
  initial_case,requires_visual,working=prepare(read(F/'case.json'),None,F,'heuristic',F/'initial_tool_observations')
  assert not requires_visual
  ok=stage('initial_assessment',lambda:per.assess(F/'case.json',ROOT/'pipeline-tools',F/'perception/baseline',working=Path(working)))"""
assert old in s;s=s.replace(old,new);p.write_text(s)
for p in C.rglob('*.py'):
 if '__pycache__' not in p.parts and 'patches' not in p.parts and not any(x.startswith('snapshot_') for x in p.relative_to(C).parts):py_compile.compile(str(p),doraise=True)
files={str(p):sha(p) for p in C.rglob('*') if p.is_file() and p.suffix in ('.py','.sbatch','.json') and '__pycache__' not in p.parts and 'patches' not in p.parts and not any(x.startswith('snapshot_') for x in p.relative_to(C).parts)}
for cid in CASES:files[str(C/cid/'case.json')]=sha(C/cid/'case.json')
save(D/'source_manifest.json',files);save(D/'original_run_source_manifest.json',rd(O.parent/'protected_sources.json'))
save(D/'staged.json',dict(cases=CASES,source_files=len(files),API_calls=0,geometry_gates_relaxed=False,original_outputs_modified=False))
print(json.dumps(rd(D/'staged.json')))
