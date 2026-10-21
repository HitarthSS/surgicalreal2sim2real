"""Trusted host fitting adapter. No model-authored code or altered CAD dimensions."""
from pathlib import Path
import sys,json,shutil,hashlib
import numpy as np,cv2
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'tool_template'),str(ROOT/'executed_sources')]
from tool_repair import NativeTool,refine,verify,LOW,HIGH
from reinitialize import align_axis,pose_seed,SEARCH
from instance_masks import select_instance
from stereo_tool_refit import refit as coverage_refit

def load(path):
 with np.load(path,allow_pickle=False) as d:return {k:d[k].copy() for k in d.files}
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))
def main():
 config=json.loads(Path(sys.argv[2]).read_text());out=Path(config['out']);case=json.loads(Path(config['case']).read_text())
 masks={s:cv2.imread(config['masks'][s],0)>0 for s in ['left','right']};images={s:cv2.imread(case['inputs'][s]) for s in masks}
 K=np.loadtxt(case['inputs']['K'],delimiter=',');depth=np.load(case['inputs']['depth_npy']);native=NativeTool(config['repo'])
 if sys.argv[1]=='seed':
  out.mkdir(parents=True,exist_ok=True)
  selections={}
  for side in masks:
   if 'boxes' in config['views'][side]:
    cv2.imwrite(str(out/(side+'_sam3_union_mask.png')),masks[side].astype('uint8')*255)
    native_report=Path(config['masks'][side]).parent/'report.json'
    native_data=json.loads(native_report.read_text()) if native_report.exists() else {}
    identity=native_data.get('views',{}).get(side,{}).get(config['instance_id'],{}).get('native_instance_selection',{})
    if identity.get('native_instance_identity_preserved'):
     assert native_data['actual_model_producer'] and native_data['full_frame']
     assert native_data['views'][side][config['instance_id']]['mask_sha256']==hashlib.sha256(Path(config['masks'][side]).read_bytes()).hexdigest()
     selections[side]=identity
    else:masks[side],selections[side]=select_instance(masks[side],config['views'][side]['boxes'],entry_border='left' if config['instance_id']=='tool_A' else 'right')
  save(out/'instance_mask_selection.json',dict(views=selections,full_frame=True,source_is_native_sam3=True,source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),selector_sha256=hashlib.sha256((ROOT/'instance_masks.py').read_bytes()).hexdigest()))
  for s,m in masks.items():
   if not 64<=int(m.sum())<=m.size*.30:raise ValueError(s+': unusable segmented instance mask')
   cv2.imwrite(str(out/(s+'_mask.png')),m.astype('uint8')*255)
  mask_path=out/'robot/sam3_tool/tool_mask.png';mask_path.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(mask_path),masks['left'].astype('uint8')*255)
  save(out/'native_request.json',dict(input_origin='agent',jaw_tips_xy=config['views']['left']['jaw_tips_xy'],keypoints_required=config['views']['left'].get('jaw_tip_visibility')=='both_visible',source_sha256={str(mask_path):hashlib.sha256(mask_path.read_bytes()).hexdigest()},visual_instance=config['instance_id'],psm_identity=None))
  return
 # Refine candidate on proposed masks; then reverify BOTH old/new geometries on those SAME masks.
 masks={side:cv2.imread(str(out/(side+'_mask.png')),0)>0 for side in ['left','right']}
 original=load(out/'mesh/tool_camera_m.npz');search=dict(SEARCH,max_seconds=90,max_evaluations=1000)
 native.validate_source(original)
 report=verify(original,masks,images,K,case['baseline_m'],depth,out/'verify_native')
 selected=dict(label='native',mesh=str(out/'mesh/tool_camera_m.npz'),report=report)
 rows=[selected]
 baseline=None
 if config.get('baseline_mesh'):baseline=verify(load(config['baseline_mesh']),masks,images,K,case['baseline_m'],depth,out/'baseline_on_same_masks')
 save(out/'evaluation.json',dict(candidate=selected,baseline=baseline,alternatives=rows,masks_are_proposals=True,independent_accuracy_measured=False,physics_validated=False))
 pair=np.hstack([cv2.imread(selected['report']['feedback_images'][s]) for s in ['left','right']]);cv2.imwrite(str(out/'candidate_full_stereo.jpg'),pair)
if __name__=='__main__':main()
