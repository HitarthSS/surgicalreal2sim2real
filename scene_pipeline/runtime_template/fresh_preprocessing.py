"""Fresh SAM3 observations and shared repair masks; reuse only verified stereo depth."""
from pathlib import Path
import json,sys,time,os
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'executed_sources')]
from contract import read,save,sha,require_test
from execution import execute,gpu_execute

def main():
 cid=sys.argv[1];require_test(ROOT,cid);f=ROOT/cid
 marker=ROOT/'ready'/(cid+'.json')
 if read(marker).get('status')=='ready':return
 model=read(ROOT/'model.json');assert sha(model['checkpoint'])==model['sha256']
 case=read(f/'case.json');baseline=read(f/'preprocessing_sources.json')
 for name,item in baseline['files'].items():assert sha(item['path'])==item['sha256'],name+' changed'
 began=time.perf_counter()
 raw=Path(case['inputs']['left']).parent
 gpu_execute(None,[sys.executable,str(ROOT/'executed_sources/run_two_arm_segmentation_checkpoint.py'),
  '--snapshot',str(raw),'--pipeline-tools',str(ROOT/'pipeline-tools'),'--checkpoint',model['checkpoint'],
  '--out',str(f/'segmentation')],f/'fresh_segmentation.log',cid,'initial_segmentation')
 cache=ROOT/'cache'/cid
 gpu_execute(None,[sys.executable,str(ROOT/'executed_sources/produce_shared_mask_repairs_v6.py'),
  '--input-root',str(raw.parent),'--pipeline-tools',str(ROOT/'pipeline-tools'),'--checkpoint',model['checkpoint'],
  '--out',str(cache),'--case-ids',cid],f/'fresh_shared_masks.log',cid,'shared_mask_proposals')
 bindings=['left','right']+[t+'_'+s+'_mask' for t in ['thread','needle'] for s in ['left','right']]
 artifacts={t+'.'+s:dict(path=str(f/'segmentation'/f'{s}_{t}_confidence.npy'),
  sha256=sha(f/'segmentation'/f'{s}_{t}_confidence.npy')) for t in ['thread','needle'] for s in ['left','right']}
 save(f/'class_confidence_state.json',dict(schema='native_class_confidence_state_v1',case_sha256=sha(f/'case.json'),
  bindings={key:dict(path=case['inputs'][key],sha256=sha(case['inputs'][key])) for key in bindings},artifacts=artifacts))
 for name,item in baseline['files'].items():assert sha(item['path'])==item['sha256'],name+' changed'
 save(f/'preprocessing.json',dict(status='complete',fresh_SAM3_initial_masks=True,fresh_SAM3_repair_cache=True,
  old_masks_or_tools_reused=False,depth_only_reused=True,source_bindings=baseline,
  checkpoint=model,elapsed_seconds=time.perf_counter()-began,slurm_job_id=os.environ['SLURM_JOB_ID']))
 save(marker,dict(status='ready',cache=str(cache),checkpoint_sha256=model['sha256'],old_masks_reused=False))

if __name__=='__main__':main()
