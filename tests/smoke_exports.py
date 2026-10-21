"""Read-only integration test against one existing completed scene; no inference."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('prepared_run',type=Path);p.add_argument('existing_scene',type=Path);p.add_argument('output',type=Path);a=p.parse_args();D=a.output;D.mkdir(exist_ok=False);cid=a.existing_scene.name;day=cid.split('_')[1];C=D/'gpt'/('day'+day);C.mkdir(parents=True);F=C/cid;F.mkdir()
for n in ['tool_template','executed_sources']:(C/n).symlink_to(a.prepared_run/'gpt'/('day'+day)/n,target_is_directory=True)
for n in ['heuristic','agent','tools']:(F/n).symlink_to(a.existing_scene/n,target_is_directory=True)
case=json.loads((a.existing_scene/'case.json').read_text());raw=Path(case['inputs']['left']).parent;case['observation_metadata']={}
for name in ['calibration.json','psm_state.json']:
 path=raw/name
 if path.exists():case['observation_metadata'][name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(F/'case.json').write_text(json.dumps(case,indent=2));(D/'scope.json').write_text(json.dumps(dict(scenes=[dict(case_id=cid,difficulty='hard')])));shutil.copy2(a.prepared_run/'config.json',D/'config.json')
for command in ['review','collect']:subprocess.run([sys.executable,'-m','scene_pipeline',command,str(D)],check=True)
receipt=json.loads((D/'reconstructions/verification.json').read_text());assert receipt[0]['controller_packages_verified']==2 and receipt[0]['zip_integrity'];assert receipt[0]['all_tool_meshes_have_angles'];print(json.dumps(receipt))
