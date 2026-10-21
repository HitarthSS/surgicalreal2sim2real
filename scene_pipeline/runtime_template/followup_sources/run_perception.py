"""Actual VLM perception feedback before the needle gate, isolated from frozen runs."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'frozen_sources'))
from openai_scene_backend import OpenAISceneModel, hash_file
from needle_thread_perception_protocol import PROMPTS, THRESHOLDS, digest, validate_decision, numerical_eligibility

PHYSICS = 'sha256:bd675f584cb274db81af062e8fc0934fdfbdf87b228f57a6ff99e6a8529bb12e'
SAM3 = 'sha256:901ee73c7e4b7904e4c52d93224635f471bb14fd1ad536cd207971ff530e14b2'
CHECKPOINT = '/data/xliang/ismr_2027/sam3_pipeline10_20261019_v1/model/sam3_human27_best_epoch2.pt'

PROMPT = '''You are a surgical stereo perception repair specialist, before reconstruction gating.
Inspect BOTH full original camera images, class overlays and red-removed/green-added change overlays.
Green class overlay=thread; cyan=needle; overlap and segmentation scores do not establish physical truth.
Fix visible object identity/extent with native SAM3 text and positive/negative bounding boxes; never draw a mask yourself.
Box coordinates are xyxy in ORIGINAL full-frame pixels, not normalized. Boxes guide SAM3; images are never cropped.
Needle is the rigid curved metal object; thread is flexible and may be attached at one needle end. Use both views independently.
Do not remove real visible thread merely to reduce branch counts. Do not classify the needle arc as thread to obtain continuity.
Original masks may be wrong: inspect every major removed region against the original images and preserve all real visible objects.
A rigid circular-arc catalog fitter runs after each proposal. It uses a fixed radius/span and one rigid pose; occluded portions are labeled inferred. Different scenes may use different needle models.
request.objects may contain thread, needle or both. For each target supply left/right with prompt,confidence_threshold,boxes.
Allowed prompts/thresholds are in state. At most4 boxes/object/view, each {xyxy:[x0,y0,x1,y1],positive:true|false}; boxes may be empty.
Use positive boxes around the target and negative boxes only on visually different objects; do not exclude the target's own visible portion.
Actions: repair_masks(parameters={objects:{...}}), accept_proposal(parameters={reviewed_removed_pixels:true,visible_objects_preserved:{left:true,right:true}}), or defer(parameters={}).
accept_proposal is only an explicitly uncertain working-mask hypothesis and requires state.numerically_eligible=true; never claim measured accuracy.
Use numerical failure feedback to select a different targeted repair; do not repeat unchanged requests. Defer if no supported correction exists.
mask_component_inventory lists foreground components and bounding boxes. Scattered distant components can corrupt a rigid fit even when the main metal arc is correct. Inspect them in BOTH originals before proposing different confidence thresholds or negative boxes on other objects. Preserve genuine occluded needle fragments; never discard components merely because they are smaller.
Native class confidences persist between repairs. If class_resolution is pending because old confidences are unavailable, request both thread and needle together to resolve their overlapping labels; an updated needle mask alone must not leave the same visible metal labeled as thread.
Return ONLY JSON with EXACT fields scene_version,action,parameters,evidence_ids,reason. Cite image.left,image.right. Reason<=700characters.
'''


class InfrastructureFailure(RuntimeError):
    pass

def mask_component_inventory(assessment):
    import cv2,numpy as np
    case=json.loads(Path(assessment['case_path']).read_text());result={}
    for target in ('thread','needle'):
        result[target]={}
        for side in ('left','right'):
            mask=cv2.imread(case['inputs'][target+'_'+side+'_mask'],0)
            count,_,stats,centers=cv2.connectedComponentsWithStats((mask>0).astype('uint8'))
            order=sorted(range(1,count),key=lambda i:int(stats[i,4]),reverse=True)
            result[target][side]=dict(component_count=count-1,total_pixels=int((mask>0).sum()),
                components=[dict(pixels=int(stats[i,4]),xyxy=[int(stats[i,0]),int(stats[i,1]),int(stats[i,0]+stats[i,2]-1),int(stats[i,1]+stats[i,3]-1)],
                    centroid=centers[i].tolist()) for i in order[:12]],truncated=count-1>12,
                components_are_observations_not_object_identities=True)
    return result


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def compact(assessment):
    thread = assessment['thread']
    return dict(needle=assessment['needle'], thread={k:thread[k] for k in ['checks_passed', 'status', 'fragment_count', 'failures', 'reference_preconditions', 'initial_mask_audit_diagnostic_only'] if k in thread},
                mask_changes=assessment['mask_changes'], sanity_failures=assessment['sanity_failures'],class_resolution=assessment.get('class_resolution',{}))


def command(script, arguments, gpu=False):
    cmd = ['docker', 'run', '--rm', '--cpus', '4', '--memory', '16g',
           '--user', str(ROOT.stat().st_uid)+':'+str(ROOT.stat().st_gid),
           '-e', 'HOME=/tmp', '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'OMP_NUM_THREADS=4',
           '-e', 'USER=ismr', '-e', 'LOGNAME=ismr', '-e', 'TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_ismr',
           '-e', 'OPENBLAS_NUM_THREADS=2', '-e', 'MPLBACKEND=Agg',
           '-v', '/home/hitarth:/home/hitarth:ro', '-v', str(ROOT.parent)+':'+str(ROOT.parent)]
    if gpu:
        cmd += ['--gpus', 'device=1']
    else:
        cmd += ['--network', 'none']
    return cmd + ['--entrypoint', 'python3', SAM3 if gpu else PHYSICS, str(ROOT / script)] + arguments


def execute(cmd, log, timeout=600):
    with log.open('w') as stream:
        return subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout).returncode


def assess(original, pipeline, folder, working=None, proposal=None):
    args = ['--baseline-case', str(original), '--pipeline', str(pipeline), '--out', str(folder)]
    if working: args += ['--working-case', str(working)]
    if proposal: args += ['--proposal-dir', str(proposal)]
    if execute(command('assess_perception.py', args), folder.parent / (folder.name+'_stdout.log')) != 0:
        raise RuntimeError('Assessment failed; inspect '+str(folder.parent / (folder.name+'_stdout.log')))
    return json.loads((folder / 'assessment.json').read_text())


def gpu_request(case_path, request_path, pipeline, destination):
    # Separate GPU1 worker; frozen remaining53 uses GPU1. Never touch its queue.
    with (ROOT.parent / 'gpu1_pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        began = time.monotonic()
        while True:
            rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
            free = {int(a):int(b) for a,b in (line.split(',') for line in rows.splitlines())}
            if free.get(1, 0) >= 6144:break
            if time.monotonic()-began > 900:raise RuntimeError('GPU1 admission timeout; no inference attempted')
            save(destination.parent / 'gpu_wait.json', dict(state='waiting', gpu=1, free_mib=free.get(1), required_mib=6144))
            time.sleep(10)
        return execute(command('segment_guided.py', ['--case', str(case_path), '--request', str(request_path),
             '--checkpoint', CHECKPOINT, '--pipeline', str(pipeline), '--out', str(destination)], gpu=True),
             destination.parent / (destination.name+'_stdout.log'))


def run(case, pipeline, out, baseline=None, max_model_attempts=5):
    out.mkdir(parents=True, exist_ok=baseline is not None)
    protected = {str(case): hash_file(case)}
    original = json.loads(case.read_text())
    protected.update({path:hash_file(path) for path in original['inputs'].values()})
    # PNG dimensions from the file header; no third-party modules on the host.
    import struct
    sizes = {s:list(struct.unpack('>II', Path(original['inputs'][s]).read_bytes()[16:24])) for s in ['left', 'right']}
    baseline = baseline or assess(case, pipeline, out / 'baseline')
    if baseline['needle']['accepted'] and baseline['thread']['checks_passed'] and not baseline['sanity_failures']:
        result=dict(case_id=original['case_id'],status='baseline_usable_no_perception_repair',accepted=False,
            selected_case=None,selected_needle=None,model_calls=0,segmentation_calls=0,
            baseline=compact(baseline),estimated_api_cost_usd=0.,independent_accuracy_measured=False)
        save(out/'result.json',result)
        return result
    current = baseline; working = Path(current['case_path']); candidate = None
    backend = OpenAISceneModel(out / 'model_requests', max_calls=max_model_attempts)
    history = []; attempted = set(); tools = 0; accepted = False; last_error = None
    for attempt in range(1, max_model_attempts+1):
        images = [('image.'+s, original['inputs'][s]) for s in ['left', 'right']]
        images += [('initial_mask.'+s, baseline['overlays'][s+'.after']) for s in ['left', 'right']]
        if candidate:
            images += [(label+'.'+s, current['overlays'][s+'.'+label]) for label in ['after', 'changes'] for s in ['left', 'right']]
        eligibility = numerical_eligibility(baseline, current, candidate['objects']) if candidate else dict(eligible=False)
        if current['sanity_failures']:eligibility['eligible'] = False
        state = dict(image_sizes=sizes, allowed_prompts=PROMPTS, confidence_thresholds=THRESHOLDS,
                     baseline=compact(baseline), current=compact(current), segmentation_calls=tools,
                     max_segmentation_calls=2, model_attempt=attempt, max_model_attempts=max_model_attempts,
                     proposal_available=candidate is not None, numerical_eligibility=eligibility,
                     numerically_eligible=eligibility['eligible'], last_error=last_error,
                     geometry_accuracy_measured=False, evidence_ids=[eid for eid,_ in images])
        state['mask_component_inventory']=mask_component_inventory(current)
        state['scene_version'] = 'p'+digest(state)[:12]
        folder = out / f'decision_{attempt:02d}';folder.mkdir();save(folder / 'state.json', state)
        content = [dict(type='text', text=PROMPT+'\nSTATE:'+json.dumps(state, separators=(',', ':')))]
        for eid, path in images:content += [dict(type='text', text='EVIDENCE_ID='+eid), dict(type='image', url=path)]
        decision = None
        try:
            raw = backend([dict(role='user', content=content)])
            (folder / 'raw_response.txt').write_text(raw)
            if raw.strip().startswith('```'):raw = '\n'.join(raw.strip().splitlines()[1:-1])
            decision = validate_decision(json.loads(raw), state)
            save(folder / 'decision.json', decision);last_error = None
            if decision['action'] == 'defer':
                history.append(dict(decision=decision, result='deferred'));break
            if decision['action'] == 'accept_proposal':
                accepted = True;history.append(dict(decision=decision, result='perception_hypothesis_accepted'));break
            params = decision['parameters'];key = digest(params)
            if key in attempted:raise ValueError('Repeated unchanged segmentation request')
            attempted.add(key);tools += 1
            request = out / f'request_{tools:02d}.json';save(request, params)
            proposal = out / f'sam3_{tools:02d}'
            code = gpu_request(working, request, pipeline, proposal)
            history.append(dict(decision=decision, tool='sam3_guided_stereo', exit_code=code, request=str(request), report=str(proposal / 'report.json')))
            if code != 0:raise InfrastructureFailure('SAM3 execution failed; stop for infrastructure review, not another model guess')
            current = assess(case, pipeline, out / f'assessment_{tools:02d}', working, proposal)
            candidate = params;working = Path(current['case_path'])
        except InfrastructureFailure as error:
            history.append(dict(decision=decision, status='infrastructure_failed', error=str(error)))
            raise
        except Exception as error:
            last_error = f'{type(error).__name__}: {error}'
            history.append(dict(decision=decision, status='failed', error=last_error))
        finally:
            save(out / 'history.json', history)
            save(out / 'progress.json', dict(model_calls=attempt, segmentation_calls=tools, accepted=accepted, error=last_error))
    assert all(hash_file(path) == value for path, value in protected.items()), 'Original data changed'
    result = dict(case_id=original['case_id'], status='accepted_perception_hypothesis' if accepted else 'deferred_or_incomplete',
                  accepted=accepted, selected_case=str(working) if accepted else None,
                  selected_needle=current['needle']['common_needle'] if accepted else None,
                  baseline=compact(baseline), final=compact(current), model_calls=backend.index,
                  segmentation_calls=tools, estimated_api_cost_usd=backend.estimated_api_cost_usd,
                  original_sha256=protected, model_visual_review_is_not_ground_truth=True,
                  independent_accuracy_measured=False, physics_validated=False, training_eligible=False)
    save(out / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ['case', 'pipeline', 'out']:parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    key = sys.stdin.readline().strip()
    if not key:raise RuntimeError('Missing protected stdin credential')
    os.environ['OPENAI_API_KEY'] = key;del key
    print(json.dumps(run(args.case, args.pipeline, args.out), indent=2))


if __name__ == '__main__':
    main()
