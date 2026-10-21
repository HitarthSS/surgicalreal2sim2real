"""Small shared-QP/adaptive-join checkpoint; no model, physics or training.

Original data and production fitter remain read-only. Pilot occluder polygons
are manually inspected development annotations, not benchmark ground truth.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

from reconstruction_tools import ReconstructionSession, fingerprints, heuristic_request, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--pipeline-tools', type=Path, required=True)
    parser.add_argument('--original-src', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    annotations = json.loads(a.annotations.read_text())
    if not 1 <= len(annotations['cases']) <= 5:
        raise ValueError('Development pilot of at most five cases')
    a.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for item in annotations['cases']:
        case_id = item['case_id']
        if Path(case_id).name != case_id:
            raise ValueError('Invalid case ID')
        folder = a.out / case_id; folder.mkdir()
        inputs = {}
        for view in ('left', 'right'):
            original = a.input_root / case_id / (view + '_rect.png')
            if sha256(original) != item['image_sha256'][view]:
                raise ValueError('Annotation image mismatch')
            image = cv2.imread(str(original))
            mask = np.zeros(image.shape[:2], np.uint8)
            for polygon in item['polygons_xy'][view]:
                xy = np.asarray(polygon, dtype=np.int32)
                if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3:
                    raise ValueError('Malformed polygon')
                if np.any(xy < 0) or np.any(xy[:, 0] >= mask.shape[1]) or np.any(xy[:, 1] >= mask.shape[0]):
                    raise ValueError('Polygon outside image')
                cv2.fillPoly(mask, [xy], 255)
            path = folder / (view + '_occluder.png'); cv2.imwrite(str(path), mask)
            overlay = image.copy()
            overlay[mask > 0] = (.6*image[mask > 0] + .4*np.array([255, 100, 0])).astype('uint8')
            cv2.imwrite(str(folder / (view + '_occluder_review.png')), overlay)
            inputs[view] = str(original)
            inputs[view + '_occluder'] = str(path)
        stereo = a.data_root / 'stereo_pilot_v2' / case_id
        for target in ('thread', 'needle'):
            for view in ('left', 'right'):
                inputs[target + '_' + view + '_mask'] = str(a.data_root / 'segmentation_pilot' / case_id / f'{view}_{target}_mask.png')
        inputs.update(K=str(stereo / 'K.csv'), depth_npy=str(stereo / 'stereo/depth_meter.npy'))
        case = dict(case_id=case_id, session_id='dvrk_20261009_uploaded_collection', split='development',
                    calibration_id='prior_matched_calibration_units_provisional',
                    baseline_m=json.loads((stereo / 'report.json').read_text())['baseline_m_hypothesis'],
                    inputs=inputs,
                    manual_interventions='Assistant manually traced visible instrument polygons in both cameras. Approximate development annotations; no thread mask edits, hidden geometry labels, or tool pose fitting. Metric units provisional.')
        case_path = folder / 'case.json'; case_path.write_text(json.dumps(case, indent=2)+'\n')
        results = {}
        for controller in ('fixed', 'heuristic'):
            session = ReconstructionSession(case_path, folder / controller, controller,
                                            'production_qp_defaults' if controller == 'fixed' else 'initial_untuned_occlusion_rule',
                                            a.pipeline_tools, max_calls=2, original_src=a.original_src)
            request = dict(tool='thread_qp', parameters={}, reason='Unchanged production QP baseline through shared checks.') if controller == 'fixed' else heuristic_request(session.observe())
            result = session.call(request)
            results[controller] = dict(request=request, result=result)
        native = folder / 'fixed/call_001/backend/thread_samples.npz'
        prior = a.data_root / 'reconstruction_pilot_v2' / case_id / 'native_original_thread/thread_samples.npz'
        same = False
        if native.is_file() and prior.is_file():
            with np.load(native) as current, np.load(prior) as previous:
                same = all(np.array_equal(current[k], previous[k]) for k in ('points_m', 'pixel_xy'))
        candidates = results['heuristic']['result'].get('backend_report', {}).get('candidates', [])
        reasons = Counter(reason for candidate in candidates for reason in candidate.get('rejections', []))
        row = dict(case_id=case_id, results=results, production_arrays_identical_to_prior=same,
                   join_candidate_count=len(candidates), candidate_rejection_counts=dict(reasons),
                   input_fingerprints=fingerprints(inputs))
        rows.append(row)
        (folder / 'summary.json').write_text(json.dumps(row, indent=2)+'\n')
        print(json.dumps(dict(case_id=case_id, qp=results['fixed']['result']['status'],
                             adaptive=results['heuristic']['result']['status'],
                             unchanged=same, candidate_rejections=dict(reasons))), flush=True)
    report = dict(rows=rows, llm_invoked=False, physics_run=False, training_eligible=False,
                  independent_accuracy_measured=False, api_cost_usd=0,
                  source_sha256=sha256(__file__), annotations_sha256=sha256(a.annotations),
                  limitations=['Approximate manually inspected instrument masks, shared with controllers.',
                               'Preselected development cases; not held-out or a dataset yield estimate.',
                               'Native QP lacks a verified join ledger; unsupported joins reject.',
                               'Calibration units provisional; no complete scene or demonstration.'])
    (a.out / 'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
