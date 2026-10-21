"""Evaluate both objects independently; original-mask agreement is diagnostic only."""
from pathlib import Path
import argparse
import json
import shutil
import sys
import numpy as np
import cv2
from PIL import Image
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'frozen_sources'))
from needle_quality import recover_needle
from reconstruction_tools import ReconstructionSession, fingerprints
from repair_aware_scene_tools import RepairAwareSceneTools as SharedSceneToolsV7
from thread_object_evidence_v8 import evaluate_thread_object_evidence
from mask_identity_state import resolve_incremental_classes,load_confidence_state,sha as confidence_sha


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    for name in ['baseline-case', 'pipeline', 'out']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--proposal-dir', type=Path)
    parser.add_argument('--working-case', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    original = json.loads(args.baseline_case.read_text())
    protected = fingerprints(dict(original['inputs'], case=args.baseline_case))
    working = json.loads(args.working_case.read_text()) if args.working_case else original
    assert working['case_id'] == original['case_id'] and working['baseline_m'] == original['baseline_m']
    assert all(working['inputs'][key] == original['inputs'][key] for key in ['left', 'right', 'depth_npy', 'K'])
    working_protected = fingerprints(working['inputs'])
    carried_confidence,confidence_paths=load_confidence_state(args.working_case or args.baseline_case,working)
    confidence_protected=fingerprints(confidence_paths)
    next_confidence_paths={}
    case = json.loads(json.dumps(working))
    request = {}
    if args.proposal_dir:
        result = json.loads((args.proposal_dir / 'report.json').read_text())
        assert result['status'] == 'proposal_generated' and result['full_frame'] and result['image_crop'] is None
        request = result['request']
        actual = fingerprints({k:v['path'] for k,v in result['source_fingerprints'].items()})
        assert actual == result['source_fingerprints'], 'Executed request/source changed'
    changes, overlays, sanity, class_resolution = {}, {}, [], {}
    for side in ['left', 'right']:
        image = np.asarray(Image.open(case['inputs'][side]).convert('RGB'))
        before, after = {}, {}
        changes[side] = {}
        for target in ['thread', 'needle']:
            key = target+'_'+side+'_mask'
            before[target] = np.asarray(Image.open(original['inputs'][key]).convert('L')) > 0
            proposed = args.proposal_dir / f'{side}_{target}_mask.png' if args.proposal_dir else None
            after[target] = np.asarray(Image.open(proposed if proposed and proposed.is_file() else working['inputs'][key]).convert('L')) > 0
            assert after[target].shape == before[target].shape == image.shape[:2]
        confidences={}
        for target in ['thread','needle']:
            confidence_key=target+'.'+side
            new_path=args.proposal_dir/f'{side}_{target}_confidence.npy' if args.proposal_dir else None
            if new_path and new_path.is_file():
                confidences[target]=np.load(new_path,allow_pickle=False);next_confidence_paths[confidence_key]=str(new_path)
            elif confidence_key in carried_confidence:
                confidences[target]=carried_confidence[confidence_key]
                next_confidence_paths[confidence_key]=confidence_paths['confidence.'+confidence_key]
        after,class_resolution[side]=resolve_incremental_classes(after,confidences)
        if args.proposal_dir and class_resolution[side]['class_resolution_pending']:
            sanity.append(side+'.overlapping_classes_need_joint_native_confidence')
        for target in ['thread', 'needle']:
            b, a = before[target], after[target]
            if target in request.get('objects', {}) and (not a.any() or a.mean() > (.15 if target == 'needle' else .25)):
                sanity.append(f'{side}.{target}.empty_or_excessive_mask')
            path = args.out / f'{side}_{target}_mask.png'
            Image.fromarray(a.astype('uint8')*255).save(path)
            case['inputs'][target+'_'+side+'_mask'] = str(path)
            old_sk = skeletonize(b)
            dist = cv2.distanceTransform((~a).astype('uint8'), cv2.DIST_L2, 5)
            changes[side][target] = dict(before_pixels=int(b.sum()), after_pixels=int(a.sum()),
                 removed_pixels=int((b & ~a).sum()), added_pixels=int((a & ~b).sum()),
                 old_skeleton_supported_fraction=float((dist[old_sk] <= 2).mean()) if old_sk.any() else None,
                 initial_mask_is_ground_truth=False)
        for label, masks in [('before', before), ('after', after)]:
            overlay = image.copy()
            for target, color in [('thread', [50, 240, 60]), ('needle', [0, 220, 255])]:
                overlay[masks[target]] = (overlay[masks[target]]*.35 + np.asarray(color)*.65).astype('uint8')
            path = args.out / f'{side}_{label}.png';Image.fromarray(overlay).save(path);overlays[side+'.'+label] = str(path)
        delta = image.copy()
        removed = (before['thread'] & ~after['thread']) | (before['needle'] & ~after['needle'])
        added = (after['thread'] & ~before['thread']) | (after['needle'] & ~before['needle'])
        delta[removed] = [255, 50, 50];delta[added] = [70, 255, 80]
        path = args.out / f'{side}_changes.png';Image.fromarray(delta).save(path);overlays[side+'.changes'] = str(path)
    # A new baseline is an explicit perception hypothesis with its own immutable
    # provenance, not a claim that revised masks are ground truth. Audit the old
    # masks separately; never delete the original evidence or change verifier limits.
    case['manual_interventions'] = 'Separate perception-feedback development hypothesis; initial masks retained and changes audited; accuracy unmeasured.'
    case_path = args.out / 'case.json';save(case_path, case)
    confidence_state=args.out/'class_confidence_state.json'
    binding_keys=['left','right']+[t+'_'+s+'_mask' for t in ['thread','needle'] for s in ['left','right']]
    save(confidence_state,dict(schema='native_class_confidence_state_v1',case_sha256=confidence_sha(case_path),
        bindings={key:dict(path=case['inputs'][key],sha256=confidence_sha(case['inputs'][key])) for key in binding_keys},
        artifacts={key:dict(path=path,sha256=confidence_sha(path)) for key,path in next_confidence_paths.items()}))
    stripped = {k:v for k,v in case.items() if k != 'scene_priors'}
    save(args.out / 'needle_case.json', stripped)
    session = ReconstructionSession(args.out / 'needle_case.json', args.out / 'needle_fit', 'agent',
                  'Same existing estimator after explicit perception repair', args.pipeline, max_calls=1, purpose="evaluation" if original.get("split")=="test" else "development")
    needle_result = session.call(dict(tool='needle_arc', parameters={}, reason='Fit existing free-radius estimator; no size prior, hidden extension or estimator switch.'))
    accepted = needle_result['accepted']
    needle_path = args.out / 'needle_fit/call_001/backend/shared_needle_arc.npz'
    if accepted:
        shutil.copy2(needle_path, args.out / 'common_needle.npz')
        needle_path = args.out / 'common_needle.npz'
    else:
        needle_path = None
    needle_quality=recover_needle(case,needle_path,args.out/'full_needle_recovery')
    selected=needle_quality['selected_path']
    if selected and (needle_path is None or Path(selected)!=needle_path):
        if needle_path:shutil.copy2(needle_path,args.out/'common_needle_original.npz')
        needle_path=args.out/'common_needle.npz'
        shutil.copy2(selected,needle_path)
    elif not selected:
        # An archived free-radius native seed is not a valid rigid fallback.
        if needle_path:shutil.copy2(needle_path,args.out/'legacy_free_radius_needle.npz')
        needle_path=None
    accepted=bool(needle_quality['complete_visible_needle'])
    thread = dict(checks_passed=False, status='not_executed')
    try:
        tools = SharedSceneToolsV7(case_path, args.pipeline, needle_path=needle_path)
        candidate, initial, _ = tools.reconstruct_first(args.out / 'thread_initial')
        thread = dict(checks_passed=bool(candidate and candidate['report']['numerical_checks_passed']),
                      status=initial['status'], fragment_count=candidate['report']['fragment_count'] if candidate else None,
                      failures=candidate['report']['failures'] if candidate else ['ordered_thread_path_unavailable'],
                      candidate=candidate, reference_preconditions=tools.reference_preconditions)
        if candidate:
            old_audit = evaluate_thread_object_evidence(original, args.pipeline, candidate['fragment_path'], needle_path)
            thread['initial_mask_audit_diagnostic_only'] = {side:dict(raw_coverage=v['raw_original_mask_coverage'],
                object_aware_coverage=v['object_aware_thread_coverage']) for side,v in old_audit['views'].items()}
    except Exception as error:
        thread.update(status='failed', failures=[f'{type(error).__name__}: {error}'])
    result = dict(case_id=case['case_id'], case_path=str(case_path), original_case=str(args.baseline_case),
                  original_fingerprints=protected, changed_masks=request.get('objects', {}), mask_changes=changes,
                  sanity_failures=sanity, overlays=overlays,class_resolution=class_resolution,class_confidence_state=str(confidence_state),
                  needle=dict(accepted=accepted, visible_quality=needle_quality, complete_visible_needle=needle_quality.get('complete_visible_needle',False), failures=[] if accepted else ['visible_needle_incomplete'], native_failures=needle_result.get('verification', {}).get('failures', []),
                              result_path=str(args.out / 'needle_fit/call_001/result.json'),
                              common_needle=str(needle_path) if needle_path else None),
                  thread=thread, revised_mask_consistency_is_not_accuracy=True,
                  independent_accuracy_measured=False, training_eligible=False)
    assert fingerprints(dict(original['inputs'], case=args.baseline_case)) == protected
    assert fingerprints(working['inputs']) == working_protected
    assert fingerprints(confidence_paths)==confidence_protected
    save(args.out / 'assessment.json', result)
    print(json.dumps(dict(needle_accepted=accepted, thread_checked=thread['checks_passed'], sanity_failures=sanity)))


if __name__ == '__main__':
    main()
