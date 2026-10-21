"""Prepare shared silhouette baseline with no model/API call."""
import json,sys,hashlib
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'dependencies')]
from shared_tool_estimation import heuristic_initialization,masks_from_request,save
from instance_masks import select_instance

def main():
 cfg=json.loads((ROOT/'pilot_config.json').read_text());case=json.loads(Path(cfg['case']).read_text())
 images={s:cv2.imread(case['inputs'][s]) for s in ['left','right']};sizes={s:[im.shape[1],im.shape[0]] for s,im in images.items()}
 cache=Path(cfg['cache']);masks={s:cv2.imread(str(cache/(s+'_mask.png')),0)>0 for s in images}
 out=ROOT/'heuristic';out.mkdir(exist_ok=True)
 try:request,sources=heuristic_initialization(images,masks)
 except ValueError as e:
  save(ROOT/'initialization_status.json',dict(heuristic=dict(status='failed',error=str(e)),agent=dict(status='inherits_heuristic')));return
 instance_masks=masks_from_request(request,sizes);selections={}
 for item in request['instances']:
  name=item['instance_id'];selections[name]={}
  for side in images:
   if sources.get(name+'.'+side)!='cached_sam3':continue
   w,h=sizes[side]
   try:
    selected,audit=select_instance(masks[side],[dict(xyxy=[0,0,w,h],positive=True)],entry_border=item['entry_border'])
    instance_masks[name][side]=selected;selections[name][side]=audit
   except ValueError as error:
    selections[name][side]=dict(status='preserved_polygon_baseline_after_ambiguous_grouping',error=str(error))
 save(out/'multipart_mask_selection.json',selections)
 save(out/'initialization.json',request);save(out/'heuristic_sources.json',sources)
 for item in request['instances']:
  name=item['instance_id'];folder=out/name;directory=folder/'robot/sam3_tool';directory.mkdir(parents=True)
  for side in images:cv2.imwrite(str(folder/(side+'_mask.png')),instance_masks[name][side].astype('uint8')*255)
  path=directory/'tool_mask.png';cv2.imwrite(str(path),instance_masks[name]['left'].astype('uint8')*255)
  save(folder/'native_request.json',dict(input_origin='heuristic',visual_instance=name,psm_identity=None,jaw_tips_xy=item['views']['left']['jaw_tips_xy'],independent_ground_truth=False,source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest(),case['inputs']['left']:hashlib.sha256(Path(case['inputs']['left']).read_bytes()).hexdigest()}))
 save(ROOT/'initialization_status.json',dict(heuristic=dict(status='prepared',instances=[x['instance_id'] for x in request['instances']]),agent=dict(status='inherits_final_heuristic_geometry')))
if __name__=='__main__':main()
