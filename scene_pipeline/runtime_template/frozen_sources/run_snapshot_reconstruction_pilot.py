"""Existing needle/thread operations and native QP baseline on new snapshots.

Partial reconstruction pilot only. No tool-pose review, agent inference or physics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--stereo-root', type=Path, required=True)
    p.add_argument('--segmentation-root', type=Path, required=True)
    p.add_argument('--shared-code', type=Path, required=True)
    p.add_argument('--pipeline-tools', type=Path, required=True)
    p.add_argument('--original-src', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--case-ids', nargs='+', required=True)
    a = p.parse_args()
    if not 1 <= len(a.case_ids) <= 5:
        raise ValueError('Small development pilot only')
    sys.path.insert(0, str(a.shared_code))
    from reconstruction_tools import ReconstructionSession, sha256
    a.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for case_id in a.case_ids:
        if Path(case_id).name != case_id:
            raise ValueError('Case IDs must be directory names')
        raw = a.input_root / case_id
        stereo = a.stereo_root / case_id
        masks = a.segmentation_root / case_id
        case = {'case_id': case_id, 'session_id': 'dvrk_20261009_uploaded_collection',
                'split': 'development', 'calibration_id': 'prior_matched_calibration_units_provisional',
                'baseline_m': json.loads((stereo / 'report.json').read_text())['baseline_m_hypothesis'],
                'manual_interventions': 'No new mask edits or tool landmarks. Calibration units inferred from matching prior pipeline; not approved for physics or training.',
                'inputs': {'left': str(raw / 'left_rect.png'), 'right': str(raw / 'right_rect.png'),
                           'K': str(stereo / 'K.csv'), 'depth_npy': str(stereo / 'stereo/depth_meter.npy'),
                           **{f'{target}_{view}_mask': str(masks / f'{view}_{target}_mask.png')
                              for target in ('needle', 'thread') for view in ('left', 'right')}}}
        case_path = a.out / (case_id + '.json'); case_path.write_text(json.dumps(case, indent=2) + '\n')
        row = {'case_id': case_id, 'unit_hypothesis_provisional': True,
               'independent_accuracy_measured': False, 'training_eligible': False}
        try:
            session = ReconstructionSession(case_path, a.out / case_id / 'shared', 'fixed',
                                            'shared_defaults_no_agent', a.pipeline_tools, max_calls=2)
            for tool in ('needle_arc', 'thread_distance'):
                result = session.call({'tool': tool, 'parameters': {}, 'reason': 'New-data defaults pilot; no repair or fallback.'})
                row[tool] = {'status': result['status'], 'accepted': result['accepted'],
                             'verification': result['verification'], 'error': result.get('error'),
                             'backend_report': result.get('backend_report'),
                             'runtime_seconds': result['runtime_seconds']}
        except Exception as error:
            row['shared_preflight_error'] = type(error).__name__ + ': ' + str(error)
        out = a.out / case_id / 'native_original_thread'; out.mkdir(parents=True)
        command = [sys.executable, str(a.pipeline_tools / 'reconstruct_thread_sam3_original.py'),
                   '--original-src', str(a.original_src), '--left', case['inputs']['left'],
                   '--right', case['inputs']['right'], '--left-mask', case['inputs']['thread_left_mask'],
                   '--right-mask', case['inputs']['thread_right_mask'], '--depth-npy', case['inputs']['depth_npy'],
                   '--K', case['inputs']['K'], '--baseline-m', str(case['baseline_m']), '--samples', '200',
                   '--segment-connection-radius-px', '80', '--min-depth-m', '.03', '--max-depth-m', '.5',
                   '--min-valid-depth-fraction', '.70', '--out', str(out)]
        start = time.perf_counter()
        with (out / 'stdout.txt').open('w') as stdout, (out / 'stderr.txt').open('w') as stderr:
            try:
                result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=120)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = None
        row['native_original_thread'] = {'command': command, 'exit_code': code,
                                        'runtime_seconds': time.perf_counter()-start,
                                        'samples_exist': (out / 'thread_samples.npz').is_file(),
                                        'shared_acceptance_not_applied': True}
        rows.append(row)
        (a.out / case_id / 'summary.json').write_text(json.dumps(row, indent=2) + '\n')
        print(json.dumps({'case_id': case_id, 'needle': row.get('needle_arc', {}).get('status'),
                          'thread_distance': row.get('thread_distance', {}).get('status'),
                          'native_thread_exit': code}), flush=True)
    source_files = [Path(__file__), *a.shared_code.glob('*.py'), *a.pipeline_tools.glob('*.py'),
                    *a.original_src.rglob('*.py')]
    (a.out / 'summary.json').write_text(json.dumps({'rows': rows,
        'source_sha256': {str(path): sha256(path) for path in source_files},
        'independent_accuracy_measured': False, 'physics_run': False, 'training_eligible': False,
        'limitations': ['Metric unit hypothesis and timing semantics unconfirmed.',
                        'Native QP output has not passed shared observed/inferred labeling checks.',
                        'Distance connector is a diagnostic comparison, not the production QP baseline.',
                        'No newest manual tool pose or complete Newton scene.']}, indent=2) + '\n')


if __name__ == '__main__':
    main()
