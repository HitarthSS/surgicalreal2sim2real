"""Independent fresh SAM3+repair-cache observations; explicit frozen depth provenance."""
from pathlib import Path
import os,sys,time
C=Path(__file__).parent;sys.path[:0]=[str(C),str(C/'executed_sources')]
from contract import read,save,sha,require_test
from execution import execute,gpu_execute

def main():
 cid=sys.argv[1];require_test(C,cid);F=C/cid;marker=C/'ready'/(cid+'.json')
 if read(marker).get('status')=='ready':return
 began=time.perf_counter();case=read(F/'case.json');scope=read(F/'input_scope.json');model=read(C/'model.json');assert sha(model['checkpoint'])==model['sha256']
 for side in ('left','right'):assert sha(case['inputs'][side])==scope['raw_hashes'][side]
 depth=Path(case['inputs']['depth_npy']);D=Path('/data/xliang/ismr_2027/dev30_dual_20261019_v1');provider=C.parent.name
 if not depth.exists():
  if scope['depth_source'].get('reused_verified_depth'):raise RuntimeError('Protected cached depth missing')
  weights=Path(os.environ['ISMR_FFS_CHECKPOINT']);assert weights.exists()
  gpu_execute(None,[sys.executable,str(C/'pipeline-tools/run_fast_foundationstereo_depth.py'),'--repo',os.environ['ISMR_FFS_ROOT'],'--model',str(weights),'--left',case['inputs']['left'],'--right',case['inputs']['right'],'--K',case['inputs']['K'],'--baseline-m',str(case['baseline_m']),'--out-dir',str(depth.parent),'--valid-iters','32','--max-disp','416','--scale','1.0'],F/'fresh_depth.log',cid,'stereo_depth')
 assert depth.exists()
 if scope['depth_source'].get('reused_verified_depth'):assert sha(depth)==scope['depth_source']['sha256']
 save(F/'preprocessing_sources.json',dict(only_depth_and_calibration_reused=scope['depth_source'].get('reused_verified_depth',False),files={name:dict(path=case['inputs'][name],sha256=sha(case['inputs'][name])) for name in ('left','right','K','depth_npy')}))
 execute([sys.executable,str(C/'fresh_preprocessing.py'),cid],F/'fresh_preprocessing.log',timeout=1800)
 save(F/'preprocessing_timing.json',dict(elapsed_seconds=time.perf_counter()-began,depth_source=scope['depth_source'],depth_sha256=sha(depth),cold_SAM3=True,human_masks_used=False))
if __name__=='__main__':main()
