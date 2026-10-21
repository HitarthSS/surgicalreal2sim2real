"""Materialize a fresh runtime from a pinned source snapshot and explicit assets."""
from pathlib import Path
import hashlib,json,shutil,py_compile,re
from .runner import save
PACKAGE=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def doctor(path):
 cfg=json.loads(Path(path).read_text());required=['sam3_checkpoint','native_tracker','legacy_assets','container_image','ffs_root','ffs_checkpoint'];checks={k:Path(cfg[k]).exists() for k in required};print(json.dumps(checks,indent=2));assert all(checks.values()),'Configured external dependencies missing'
 reference=PACKAGE/'resources/native_reference_hashes.json'
 if reference.exists() and cfg.get('verify_native_reference',True):
  for name,expected in json.loads(reference.read_text()).items():
   assert sha(Path(cfg['native_tracker'])/name)==expected,'Native dependency differs from pinned reference: '+name
 assert sha(cfg['sam3_checkpoint'])==cfg['sam3_sha256'],'SAM3 checkpoint hash mismatch'
 return cfg

def main(manifest,config,output):
 cfg=doctor(config);manifest=json.loads(Path(manifest).read_text());rows=manifest['scenes'];ids=[r['case_id'] for r in rows]
 assert ids and len(ids)==len(set(ids));assert all(re.fullmatch(r'snapshot_\d{8}_\d{6}',cid) for cid in ids),'Scene IDs must be snapshot_YYYYMMDD_HHMMSS';root=Path(output).resolve();root.mkdir(exist_ok=False);save(root/'scope.json',dict(scenes=rows,provider='gpt',max_calls_per_scene=23,original_inputs_preserved=True));save(root/'config.json',cfg)
 source=PACKAGE/'runtime_template';assert source.exists(),'Bundled runtime sources missing'
 for day in sorted({cid.split('_')[1] for cid in ids}):
  C=root/'gpt'/('day'+day);shutil.copytree(source,C)
  replacements=dict(cfg.get('path_overrides',{}));replacements.update({
   '/home/hitarth/online_dvrk_tracking_clean_extracted/online_dvrk_tracking_clean':cfg['native_tracker'],
   '/data/xliang/ismr_2027/sam3_pipeline10_20261019_v1/model/sam3_human27_best_epoch2.pt':cfg['sam3_checkpoint'],
   '/home/xliang/ismr_sam3_pipeline10_20261019_v1':str(C),
   '/home/hitarth':cfg['legacy_assets']})
  for p in C.rglob('*.py'):
   text=p.read_text()
   for old,new in sorted(replacements.items(),key=lambda x:-len(x[0])):text=text.replace(old,new)
   text=text.replace('snapshot_20261015_','snapshot_'+day+'_');p.write_text(text)
  # Generic manifest replaces the historical locked30-only gate. Public entry points
  # still validate exact scope and immutable stereo inputs before every scene.
  p=C/'contract.py';text=p.read_text();start=text.index('def test_rows(');end=text.index('def api_attempts(',start)
  text=text[:start]+'''def test_rows(root):
    return read(Path(root)/'scene_scope.json')['scenes']
def require_test(root,cid):
    rows=[r for r in test_rows(root) if r['case_id']==cid]
    if len(rows)!=1:raise ValueError('Outside explicit run manifest')
    return rows[0]

'''+text[end:];p.write_text(text)
  selected=[r for r in rows if r['case_id'].split('_')[1]==day];save(C/'scene_scope.json',dict(scenes=selected));save(C/'recovery_scope.json',dict(cases=[r['case_id'] for r in selected],partition='development'));save(C/'job_manifest.json',dict(cases=[r['case_id'].rsplit('_',1)[1] for r in selected]));save(C/'model.json',dict(checkpoint=cfg['sam3_checkpoint'],sha256=cfg['sam3_sha256']))
  save(C/'pilot_manifest.json',dict(cases={r['case_id'].rsplit('_',1)[1]:dict(previous_calls=0,max_new_calls=5) for r in selected}))
  for folder in ['ready','cache','export_context','advisor_handoff','all_geometry_export/review','all_geometry_export/advisor_handoff']:(C/folder).mkdir(parents=True,exist_ok=True)
  for row in selected:
   cid=row['case_id'];F=C/cid;F.mkdir();case=json.loads(Path(row['source_case']).read_text()) if row.get('source_case') else dict(row['case']);case['case_id']=cid
   source_hashes={}
   for key in ['left','right','K','depth_npy']:
    if case['inputs'].get(key) and Path(case['inputs'][key]).exists():source_hashes[key]=sha(case['inputs'][key])
   for side in ['left','right']:
    assert source_hashes[side]==row['image_sha256'][side]
    for target in ['thread','needle']:case['inputs'][target+'_'+side+'_mask']=str(F/'segmentation'/f'{side}_{target}_mask.png')
   shutil.copy2(case['inputs']['K'],F/'K.csv');case['inputs']['K']=str(F/'K.csv')
   depth=Path(case['inputs'].get('depth_npy','/missing'))
   reused=depth.is_file()
   if not reused:case['inputs']['depth_npy']=str(F/'stereo/depth_meter.npy')
   case['immutable_input_sha256']=source_hashes
   case['observation_metadata']={}
   for filename,key in [('calibration.json','calibration_path'),('psm_state.json','robot_state_path')]:
    path=Path(row[key]) if row.get(key) else Path(case['inputs']['left']).parent/filename
    if row.get(key) and not path.is_file():raise FileNotFoundError('Explicit observation metadata is missing: '+str(path))
    if path.is_file():case['observation_metadata'][filename]=dict(path=str(path.resolve()),sha256=sha(path))
   save(F/'case.json',case);save(F/'input_scope.json',dict(difficulty=row['difficulty'],partition=row.get('partition','development'),raw_hashes=row['image_sha256'],depth_source=dict(reused_verified_depth=reused,**(dict(path=str(depth),sha256=source_hashes['depth_npy']) if reused else dict(fresh_inference_required=True)))))
  for p in C.rglob('*.py'):py_compile.compile(str(p),doraise=True)
 frozen=list(root.rglob('*.py'))+[root/'scope.json']+[p for day in (root/'gpt').glob('day*') for p in [day/'preregistration.json',day/'model.json',day/'scene_scope.json',day/'recovery_scope.json']]
 save(root/'source_hashes.json',{str(p.relative_to(root)):sha(p) for p in frozen});print(json.dumps(dict(root=str(root),scenes=len(rows))))
