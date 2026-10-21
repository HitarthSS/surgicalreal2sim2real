"""CPU-only provenance and cache readability check; no inference."""
from pathlib import Path
import sys,json,hashlib,time,py_compile
from PIL import Image
R=Path(__file__).resolve().parent
sys.path[:0]=[str(R),str(R/'scene_sources'),str(R/'executed_sources')]
from shared_scene_tools_v7 import SharedSceneToolsV7
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
job=read(R/'job_manifest.json');assert len(job['cases'])==33
assert [c for batch in job['batches'] for c in batch]==job['cases']
assert all(sha(R/n)==h for n,h in read(R/'runtime_manifest.json').items())
cards=['segment.precision','segment.recall','occluder.sam3.instrument','occluder.sam3.grasper'];count=0
for c in job['cases']:
 F=R/('snapshot_20261015_'+c);ready=read(R/'ready'/(F.name+'.json'))
 assert ready['status']=='ready'
 tools=SharedSceneToolsV7(F/'case.json',R/'pipeline-tools',cache=ready['cache'])
 for card in cards:
  masks,report=tools.read_cache(card);assert set(masks)=={'left','right'}
  assert all(Image.open(mask).size==(640,480) for mask in masks.values());count+=1
for n in ['overnight_master.py','overnight_driver.py','overnight_execution.py','overnight_pack.py','transport_hook/sitecustomize.py']:
 compile((R/n).read_text(),str(R/n),'exec')
result=dict(passed=True,cases=33,cached_cards_read=count,source_hashes_verified=True,full_stereo=True,real_API_calls=0,new_GPU_preprocessing=False,checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
(R/'overnight_preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
