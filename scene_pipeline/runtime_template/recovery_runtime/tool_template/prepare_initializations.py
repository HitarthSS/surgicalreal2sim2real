"""Fixed heuristic and actual VLM preparation, same two-instance fitting tools."""
import hashlib
import json
import os
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'dependencies'))
from openai_scene_backend import OpenAISceneModel
from shared_tool_estimation import heuristic_initialization,masks_from_request,save
from tool_protocol import validate_initialization,digest,LIMITS

PROMPT='''You are the two-instrument initialization specialist in a surgical real-to-sim pipeline.
Use BOTH full original stereo images and cached-mask overlays. Find each actual tool's silhouette. Jaw tips are OPTIONAL: provide [] if two distinct tips are not visible.
tool_A is the instrument entering the left image border; tool_B enters the right border. These are visual IDs, NOT PSM1/PSM2 assignments.
Use original pixel-center coordinates x=0..639,y=0..479. No crops or normalized coordinates. Do not label the needle/thread as tool.
The common fitter is the existing articulated native LND tool model: a silhouette initializes Bayesian fitting; visible jaw tips optionally add landmark evidence.
Inputs are uncertain machine proposals, never a human-reviewed mask or measured ground truth. Do not invent a visible jaw tip if indeterminate.
For initialize_tools return parameters={instances:[{instance_id:tool_A|tool_B,entry_border:left|right,views:{left:{polygon:[[x,y],...],jaw_tips_xy:[[x,y],[x,y]]},right:{polygon:[[x,y],...],jaw_tips_xy:[[x,y],[x,y]]}},reason:...}, ...]}.
Each polygon has 3–16 vertices, but use at most10 if adequate to stay concise; must meet its entering frame border and cover only visible instrument surface. Include each independently available instance with BOTH views; one difficult tool must not block another.
Use silhouette-only fitting with jaw_tips_xy=[] when jaw tips are uncertain. Return one instance when only one usable silhouette exists. Defer only when no usable tool silhouette can be supplied. Never invent observed tips.
Return ONLY JSON fields scene_version,action_id,parameters,evidence_ids,reason. Cite image.left,image.right. Top-level reason<=400chars, each instance reason<=400chars.
'''


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    config=json.loads((ROOT/'pilot_config.json').read_text());case=json.loads(Path(config['case']).read_text())
    images={s:cv2.imread(case['inputs'][s]) for s in ['left','right']};sizes={s:[im.shape[1],im.shape[0]] for s,im in images.items()}
    cached={s:cv2.imread(str(Path(config['cache'])/(s+'_mask.png')),0)>0 for s in images}
    protected={p:sha(p) for p in list(case['inputs'].values())+[config['case']]}
    save(ROOT/'protected_inputs.json',protected)
    results={}
    for method in ['heuristic','agent']:
        out=ROOT/method;out.mkdir(exist_ok=False)
        request=None
        try:
            if method=='heuristic':
                request,sources=heuristic_initialization(images,cached);save(out/'heuristic_sources.json',sources)
            else:
                key=sys.stdin.readline().strip();assert key
                os.environ['OPENAI_API_KEY']=key;del key
                backend=OpenAISceneModel(out/'model_requests',max_calls=2)
                feedback=None
                for attempt in range(2):
                    state=dict(image_sizes=sizes,limits=LIMITS,case_id=case['case_id'],last_contract_error=feedback,
                               psm_identity_assigned=False,available_operations=['initialize_tools','silhouette_only_initialization','defer_if_no_usable_silhouette'])
                    state['scene_version']='t'+digest(state)[:12]
                    content=[dict(type='text',text=PROMPT+'\nSTATE:'+json.dumps(state,separators=(',',':')))]
                    for s in images:
                        content.extend([dict(type='text',text='EVIDENCE_ID=image.'+s),dict(type='image',url=case['inputs'][s]),
                                        dict(type='text',text='EVIDENCE_ID=cached_mask.'+s),dict(type='image',url=str(Path(config['cache'])/(s+'_overlay.png')))])
                    raw=backend([dict(role='user',content=content)])
                    try:
                        value=json.loads(raw)
                        if set(value)!={'scene_version','action_id','parameters','evidence_ids','reason'} or value['scene_version']!=state['scene_version']:raise ValueError('Exact fields and version required')
                        if not {'image.left','image.right'}<=set(value['evidence_ids']) or not set(value['evidence_ids'])<={'image.left','image.right','cached_mask.left','cached_mask.right'}:raise ValueError('Registered stereo evidence required')
                        if not isinstance(value['reason'],str) or not 1<=len(value['reason'])<=400:raise ValueError('Reason bound')
                        if value['action_id']=='defer' and not value['parameters']:raise RuntimeError('Explicit agent deferral: '+value['reason'])
                        if value['action_id']!='initialize_tools':raise ValueError('Unknown initialization action')
                        valid=[];instance_errors=[]
                        for item in value['parameters'].get('instances',[]):
                            try:
                                single=validate_initialization(dict(instances=[item]),sizes);masks_from_request(single,sizes)
                                valid.extend(single['instances'])
                            except ValueError as error:instance_errors.append(dict(instance_id=item.get('instance_id'),error=str(error)))
                        if not valid:raise ValueError('No usable silhouettes: '+str(instance_errors))
                        request=validate_initialization(dict(instances=valid),sizes);masks_from_request(request,sizes)
                        if instance_errors:save(out/'partial_initialization_errors.json',instance_errors)
                        save(out/'initial_decision.json',value);break
                    except ValueError as error:
                        feedback=str(error);save(out/f'contract_failure_{attempt+1}.json',dict(error=feedback,raw=raw))
                        if attempt==1:raise
                save(out/'model_budget.json',dict(actual_attempts=backend.index,max_attempts=2,estimated_api_cost_usd=backend.estimated_api_cost_usd))
            masks=masks_from_request(request,sizes);save(out/'initialization.json',request)
            for item in request['instances']:
                name=item['instance_id'];folder=out/name;directory=folder/'robot/sam3_tool';directory.mkdir(parents=True)
                for side in images:
                    path=folder/(side+'_mask.png');cv2.imwrite(str(path),masks[name][side].astype('uint8')*255)
                mask_path=directory/'tool_mask.png';cv2.imwrite(str(mask_path),masks[name]['left'].astype('uint8')*255)
                save(folder/'native_request.json',dict(input_origin=method,visual_instance=name,psm_identity=None,
                    jaw_tips_xy=item['views']['left']['jaw_tips_xy'],independent_ground_truth=False,
                    source_sha256={str(mask_path):sha(mask_path),case['inputs']['left']:sha(case['inputs']['left'])}))
            results[method]=dict(status='prepared',instances=[x['instance_id'] for x in request['instances']])
        except Exception as error:
            results[method]=dict(status='failed_initialization',error=f'{type(error).__name__}: {error}')
            save(out/'initialization_failure.json',results[method])
    if results.get('agent',{}).get('status')!='prepared' and results.get('heuristic',{}).get('status')=='prepared':
        import shutil
        failed=dict(results['agent'])
        request=json.loads((ROOT/'heuristic/initialization.json').read_text())
        save(ROOT/'agent/initialization.json',request)
        for item in request['instances']:
            name=item['instance_id'];folder=ROOT/'agent'/name
            shutil.copytree(ROOT/'heuristic'/name,folder,dirs_exist_ok=True)
            native=json.loads((folder/'native_request.json').read_text())
            mask=folder/'robot/sam3_tool/tool_mask.png'
            native.update(input_origin='agent',initialization_source='shared_automatic_fallback',source_sha256={str(mask):sha(mask),case['inputs']['left']:sha(case['inputs']['left'])})
            save(folder/'native_request.json',native)
        save(ROOT/'agent/initialization_fallback.json',dict(policy='best_effort_v2',original_failure=failed,source='shared deterministic heuristic initialization',jaw_tips_are_estimated=True,independent_ground_truth=False))
        results['agent']=dict(status='prepared',instances=[x['instance_id'] for x in request['instances']],fallback='shared_automatic_initialization',original_failure=failed)
    save(ROOT/'initialization_status.json',results)
    assert all(sha(p)==h for p,h in protected.items())
    print(json.dumps(results))


if __name__=='__main__':main()
