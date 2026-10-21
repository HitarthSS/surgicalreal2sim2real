"""Numerical feedback, bounded shared refinement and final selected tool state."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from tool_protocol import LIMITS
from shared_tool_estimation import verify,refine_stereo_pose,save

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--method',choices=['heuristic','agent'],required=True)
    parser.add_argument('--instance',choices=['tool_A','tool_B'],required=True)
    args=parser.parse_args();config=json.loads((ROOT/'pilot_config.json').read_text());case=json.loads(Path(config['case']).read_text())
    folder=ROOT/args.method/args.instance;path=folder/'mesh/tool_camera_m.npz'
    with np.load(path,allow_pickle=False) as data:mesh={k:data[k].copy() for k in data.files}
    masks={s:cv2.imread(str(folder/(s+'_mask.png')),0)>0 for s in ['left','right']}
    images={s:cv2.imread(case['inputs'][s]) for s in masks};K=np.loadtxt(case['inputs']['K'],delimiter=',');depth=np.load(case['inputs']['depth_npy'])
    before=verify(mesh,masks,images,K,case['baseline_m'],depth,folder/'verification_initial')
    save(folder/'selected.json',dict(mesh=str(path),verification=before,status='numerically_checked_tool_proposal' if before['accepted_numerical_hypothesis'] else 'rejected_tool_fit',
        visual_instance=args.instance,psm_identity=None,controller=args.method,independent_accuracy_measured=False,physics_validated=False))


if __name__=='__main__':main()
