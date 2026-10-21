"""Create an isolated full-pipeline revision with the selected fine-tuned SAM3."""
from pathlib import Path
import hashlib,json,shutil,py_compile
C=Path('/home/xliang/ismr_sam3_pipeline10_20261019_v1')
D=Path('/data/xliang/ismr_2027/sam3_pipeline10_20261019_v1')
B=Path('/home/xliang/ismr_sam3_pipeline10_20261019_v1')
BD=Path('/data/xliang/ismr_2027/sam3_pipeline10_20261019_v1')
OLD=Path('/data/xliang/ismr_2027/agent70_20261018_v1/day20261015')
def rd(p):return json.loads(Path(p).read_text())
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def cp(a,b):b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b)
assert not (D/'staged.json').exists() and not (D/'started.json').exists()
model=rd(D/'model/checkpoint.json');assert sha(model['checkpoint'])==model['sha256']
selection=rd(C/'selection.json');cases=[r['case_id'] for r in selection['cases']]
assert len(cases)==10
for p in B.glob('*.py'):cp(p,C/p.name)
for name in ['scene_sources','executed_sources','frozen_sources','pipeline-tools','tool_template','perception_runtime','transport_hook']:
 shutil.copytree(B/name,C/name,symlinks=False,ignore=shutil.ignore_patterns('__pycache__'))
for name in ['recovery_runtime','all_geometry_export']:
 (C/name).mkdir(exist_ok=True)
 for p in (B/name).glob('*.py'):cp(p,C/name/p.name)
 for link in ['executed_sources','tool_template']:(C/name/link).symlink_to(C/link,target_is_directory=True)
for p in (C/'followup_sources').glob('*.py'):
 if p.name in ['assess_perception.py','run_perception.py','segment_guided.py']:cp(p,C/'perception_runtime'/p.name)
 else:cp(p,C/p.name)
for name in ['preregistration.json','job_manifest.json','split_manifest.json']:cp(B/name,C/name)
for p in C.rglob('*.py'):
 if '__pycache__' in p.parts:continue
 text=p.read_text().replace(str(BD),str(D)).replace(str(B),str(C))
 text=text.replace('/data/xliang/ismr_2027/sam3_pipeline10_20261019_v1/model/sam3_human27_best_epoch2.pt',model['checkpoint'])
 text=text.replace('64617502feb29df7ecd4b0ee831aa512f86f0b86c67cdd3af1a85d9e2802b9f7',model['sha256'])
 p.write_text(text)
save(C/'model.json',model)
manifest=rd(C/'job_manifest.json');manifest['cases']=[c.rsplit('_',1)[1] for c in cases]
manifest['batches']=[[x] for x in manifest['cases']];save(C/'job_manifest.json',manifest)
scope=dict(cases=cases,API_per_case=23,maximum_new_scene_API_attempts=230,source_run=str(B),
 checkpoint=model,scope='ten approved out-of-finetuning scenes, five easy/five hard; all masks and tool fits freshly computed')
save(C/'recovery_scope.json',scope);save(D/'run_scope.json',scope)
for name in ['cache','ready','export_context','advisor_handoff']:
 dest=D/name;dest.mkdir(exist_ok=True);(C/name).symlink_to(dest,target_is_directory=True)
for name in ['advisor_handoff','review']:
 dest=D/'all_geometry_export'/name;dest.mkdir(parents=True,exist_ok=True);(C/'all_geometry_export'/name).symlink_to(dest,target_is_directory=True)
for row in selection['cases']:
 cid=row['case_id'];f=D/cid;f.mkdir();(C/cid).symlink_to(f,target_is_directory=True)
 original=rd(OLD/cid/'case.json');case=json.loads(json.dumps(original))
 cp(OLD/cid/'K.csv',f/'K.csv');case['inputs']['K']=str(C/cid/'K.csv')
 for side in ['left','right']:
  assert sha(case['inputs'][side])==row['files'][side+'_rect.png']['sha256']
  for target in ['thread','needle']:case['inputs'][target+'_'+side+'_mask']=str(C/cid/'segmentation'/f'{side}_{target}_mask.png')
 save(f/'case.json',case)
 files={k:dict(path=original['inputs'][k],sha256=sha(original['inputs'][k])) for k in ['left','right','K','depth_npy']}
 save(f/'preprocessing_sources.json',dict(source_case=str(OLD/cid/'case.json'),source_case_sha256=sha(OLD/cid/'case.json'),files=files,only_depth_and_calibration_reused=True))
# Persist the native confidences without changing the initial segmentation rule.
p=C/'executed_sources/run_two_arm_segmentation_checkpoint.py';text=p.read_text()
needle="tm, nm, overlap = existing.resolve_class_overlap(thread, needle)"
assert text.count(needle)==1
text=text.replace(needle,needle+"\n                np.save(a.out / (side + '_thread_confidence.npy'), thread['confidence'])\n                np.save(a.out / (side + '_needle_confidence.npy'), needle['confidence'])")
p.write_text(text)
# Recompute heuristic tool fits before early occlusion/needle assessment.
p=C/'case_worker.py';text=p.read_text()
old="save(F/'ready.json',marker);baseline_path=F/'perception/baseline/assessment.json'"
assert text.count(old)==1
text=text.replace(old,"save(F/'ready.json',marker)\n if not (F/'tools/complete_tools.json').exists():\n  stage('heuristic_tools',lambda:frozen.heuristic_tools(F,Path(marker['cache'])))\n baseline_path=F/'perception/baseline/assessment.json'")
old="assert not requires_visual\n  ok=stage('initial_assessment'"
new="""assert not requires_visual
  confidence=read(F/'class_confidence_state.json')
  assert confidence and all(initial_case['inputs'][k]==read(F/'case.json')['inputs'][k] for k in confidence['bindings'])
  confidence['case_sha256']=sha(working)
  save(Path(working).parent/'class_confidence_state.json',confidence)
  ok=stage('initial_assessment'"""
assert old in text;text=text.replace(old,new);p.write_text(text)
# The native ten-scene driver retains raw attempts, continuation and provider guards.
p=C/'run_recovery.py';text=p.read_text()
text=text.replace('total=4','total=10').replace("'total':4","'total':10").replace('<=92','<=230')
text=text.replace("if any(r['errors'] or r['API_execution']['agent_work_pending'] for r in rows)","if any(r['errors'] or r['status']!='attempt_complete' or r['API_execution']['agent_work_pending'] for r in rows)")
old="f=C/cid\n  if (f/'case_result.json').exists():continue"
new="""f=C/cid
  if (f/'case_result.json').exists():continue
  try:
   from execution import execute
   execute([sys.executable,str(C/'fresh_preprocessing.py'),cid],f/'fresh_preprocessing.log',timeout=1800)
  except Exception as error:
   save(f/'case_result.json',dict(case_id=cid,status='fresh_preprocessing_failed',errors=[dict(stage='fresh_SAM3',error=str(error))],model_attempts=0))
   continue"""
assert old in text;text=text.replace(old,new)
# Metrics must also represent a preprocessing failure rather than omit it.
text=text.replace("total=4,API_physical_attempts", "total=10,API_physical_attempts")
text=text.replace("def audit():", "def audit():\n assert sha(read(C/'model.json')['checkpoint'])==read(C/'model.json')['sha256'],'Selected SAM3 checkpoint changed'")
p.write_text(text)
for p in C.rglob('*.py'):
 if '__pycache__' not in p.parts and not any(x.startswith('snapshot_') for x in p.relative_to(C).parts):py_compile.compile(str(p),doraise=True)
protected={str(p):sha(p) for p in C.rglob('*') if p.is_file() and p.suffix in ['.py','.sbatch','.json'] and '__pycache__' not in p.parts and not any(x.startswith('snapshot_') for x in p.relative_to(C).parts)}
for cid in cases:protected[str(C/cid/'case.json')]=sha(C/cid/'case.json')
save(D/'source_manifest.json',protected)
save(D/'original_run_source_manifest.json',rd(BD/'source_manifest.json'))
save(D/'staged.json',dict(cases=cases,source_files=len(protected),checkpoint=model,all_masks_fresh=True,old_results_unchanged=True,API_calls=0))
print(json.dumps(rd(D/'staged.json')))
