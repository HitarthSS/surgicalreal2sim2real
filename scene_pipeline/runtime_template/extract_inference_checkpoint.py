"""Extract exact model tensors; leave training checkpoint and optimizer untouched."""
from pathlib import Path
import hashlib
import json
import os
import time
import torch

ROOT=Path('/data/xliang/ismr_2027/sam3_pipeline10_20261019_v1')
TRAIN=Path('/data/xliang/ismr_2027/sam3_snapshot30_20261019_v1/finetune27_v1/run')

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for raw in iter(lambda:f.read(8*1024**2),b''):h.update(raw)
 return h.hexdigest()

def main():
 assert not torch.cuda.is_available(), 'Checkpoint extraction is CPU-only; no GPU access is needed'
 torch.set_num_threads(4)
 selection=json.loads((TRAIN/'selected_checkpoint.json').read_text())
 assert selection['phase']=='finetuned' and selection['completed_epochs']==2
 source=Path(selection['checkpoint']);before=sha(source);began=time.time()
 out=ROOT/'model/sam3_human27_best_epoch2.pt';out.parent.mkdir(parents=True,exist_ok=True)
 assert not out.exists(),'Do not overwrite an existing inference checkpoint'
 full=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
 state={k.removeprefix('module.'):v for k,v in full['model'].items()}
 del full
 assert len(state)==1134 and all(torch.isfinite(v).all() for v in state.values())
 temp=out.with_suffix('.partial');torch.save({'model':state},temp);temp.replace(out)
 restored=torch.load(out,map_location='cpu',weights_only=True)['model']
 assert set(restored)==set(state)
 assert all(torch.equal(state[k],restored[k]) for k in state)
 assert sha(source)==before
 report=dict(checkpoint=str(out),sha256=sha(out),source_checkpoint=str(source),source_sha256=before,
  tensor_count=len(state),all_tensors_exactly_equal=True,source_unchanged=True,selected_epoch=2,
  runtime_seconds=time.time()-began,bytes=out.stat().st_size)
 (ROOT/'model/checkpoint.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
