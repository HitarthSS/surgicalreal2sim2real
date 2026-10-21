"""Generate a Slurm/Apptainer job from explicit portable configuration."""
from pathlib import Path
import json,shlex,subprocess,fcntl

def script(root):
 root=Path(root).resolve();cfg=json.loads((root/'config.json').read_text());package=Path(__file__).resolve().parents[1];q=shlex.quote
 binds=list(dict.fromkeys([str(root.parent),str(package.parent),'/data']+cfg.get('bind_paths',[])))
 mount=' '.join('--bind '+q(p+':'+p) for p in binds)
 return f'''#!/bin/bash
#SBATCH --job-name=surgical-scenes
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output={root}/run-%j.log
set -euo pipefail
export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export APPTAINERENV_SLURM_JOB_ID="$SLURM_JOB_ID"
export APPTAINERENV_PYTHONPATH={q(str(package)+':/opt/ismr_sam3')}
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_OMP_NUM_THREADS=8
cd {q(str(package))}
apptainer exec --nv {mount} {q(cfg['container_image'])} python3 {q(str(package/'scene_pipeline/credential_entry.py'))} {q(str(root))}
apptainer exec {mount} {q(cfg['container_image'])} python3 -m scene_pipeline review {q(str(root))}
apptainer exec {mount} {q(cfg['container_image'])} python3 -m scene_pipeline collect {q(str(root))}
'''
def main(root):
 root=Path(root).resolve();target=root/'run.sbatch'
 with (root/'submission.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  receipt=root/'submission.json'
  if receipt.exists():raise ValueError('Submission already recorded; inspect existing Slurm job instead of duplicating it')
  target.write_text(script(root));receipt.write_text(json.dumps(dict(state='submission_started')))
  result=subprocess.run(['sbatch','--parsable',str(target)],check=True,capture_output=True,text=True)
  receipt.write_text(json.dumps(dict(state='submitted',job_id=result.stdout.strip()),indent=2));print(result.stdout.strip())
