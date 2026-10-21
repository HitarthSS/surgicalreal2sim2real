"""Run the existing native Bayesian initializer/tracker and export its mesh."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent/'tool_template'
sys.path[:0]=[str(ROOT/'native_pipeline'),str(ROOT/'dependencies')]


def main():
    parser=argparse.ArgumentParser()
    for name in ['repo','run','K','request']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    import torch,numpy as np
    torch.manual_seed(0);np.random.seed(0)
    torch.cuda.set_per_process_memory_fraction(min(1,8*1024**3/torch.cuda.get_device_properties(0).total_memory))
    import run_online_tracker_frame as native
    sys.argv=['native','--repo',str(args.repo),'--run',str(args.run),'--K',str(args.K),
              '--seed-source','original','--initialization-request',str(args.request)]
    native.main()
    import gc
    gc.collect();torch.cuda.empty_cache()
    import export_native_tool_mesh_checkpoint as exporter
    sys.argv=['export','--repo',str(args.repo),'--K',str(args.K),
              '--result',str(args.run/'robot/online_tracker/tracker_result.npz'),'--out',str(args.run/'mesh')]
    exporter.main()


if __name__=='__main__':main()
