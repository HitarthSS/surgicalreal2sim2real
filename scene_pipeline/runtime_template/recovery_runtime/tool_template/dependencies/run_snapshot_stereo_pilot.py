"""Existing FFS wrapper on three preregistered snapshot pairs; no scene training.

Metric scale is a documented hypothesis from the matching prior calibration.
No changes to the original FFS checkout, reconstruction or neural weights.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--pipeline-tools', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--prior-K', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--case-ids', nargs='+', required=True)
    a = p.parse_args()
    model_config = a.model.parent / 'cfg.yaml'
    if not a.model.is_file() or not model_config.is_file():
        raise FileNotFoundError('Existing FFS requires both the model and adjacent cfg.yaml')
    import numpy as np
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one idle GPU required')
    if not 1 <= len(a.case_ids) <= 5 or len(set(a.case_ids)) != len(a.case_ids):
        raise ValueError('One to five distinct cases only')
    prior_K = np.loadtxt(a.prior_K, delimiter=',')
    cases = []
    for case_id in a.case_ids:
        if Path(case_id).name != case_id:
            raise ValueError('Invalid case directory')
        root = a.input_root / case_id
        c = json.loads((root / 'calibration.json').read_text())
        K = np.asarray(c['P1'], dtype=float)[:, :3]
        baseline_mm = -float(c['P2'][0][3]) / float(c['P2'][0][0])
        if not np.allclose(K, prior_K, rtol=1e-7, atol=1e-4) or not np.isclose(baseline_mm*.001, .006820275085, rtol=1e-7):
            raise ValueError('New calibration differs from the prior calibration; do not guess scale')
        for view in ('left', 'right'):
            if not (root / (view + '_rect.png')).is_file():
                raise FileNotFoundError(root / (view + '_rect.png'))
        cases.append((case_id, root, K, baseline_mm*.001))
    a.out.mkdir(parents=True, exist_ok=False)
    manifest = {'runner_sha256': sha256(__file__),
                'existing_stereo_wrapper_sha256': sha256(a.pipeline_tools / 'run_fast_foundationstereo_depth.py'),
                'model_sha256': sha256(a.model), 'model_config_sha256': sha256(model_config),
                'prior_K_sha256': sha256(a.prior_K),
                'run_demo_before_sha256': sha256(a.repo / 'scripts/run_demo.py'),
                'units_evidence': 'Millimeters inferred from exact intrinsics/baseline agreement with existing pipeline calibration.',
                'physical_baseline_units_confirmed_by_collector': False,
                'valid_iters': 32, 'max_disp': 416, 'scale': 1.,
                'split': 'development', 'independent_accuracy_measured': False,
                'training_eligible': False, 'api_cost_usd': 0., 'compute_cost_usd': None}
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    rows = []
    for case_id, root, K, baseline in cases:
        out = a.out / case_id; out.mkdir()
        np.savetxt(out / 'K.csv', K, delimiter=',')
        command = [sys.executable, str(a.pipeline_tools / 'run_fast_foundationstereo_depth.py'),
                   '--repo', str(a.repo), '--model', str(a.model),
                   '--left', str(root / 'left_rect.png'), '--right', str(root / 'right_rect.png'),
                   '--K', str(out / 'K.csv'), '--baseline-m', str(baseline),
                   '--valid-iters', '32', '--max-disp', '416', '--out-dir', str(out / 'stereo')]
        start = time.perf_counter()
        with (out / 'stdout.txt').open('w') as stdout, (out / 'stderr.txt').open('w') as stderr:
            try:
                process = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=300)
                code = process.returncode
            except subprocess.TimeoutExpired:
                code = None
        row = {'case_id': case_id, 'command': command, 'exit_code': code,
               'runtime_seconds': time.perf_counter()-start, 'baseline_m_hypothesis': baseline,
               'calibration_sha256': sha256(root / 'calibration.json'),
               'units_provisional': True, 'training_eligible': False}
        for name in ('disp.npy', 'depth_meter.npy'):
            path = out / 'stereo' / name
            row[name] = {'exists': path.is_file(), 'sha256': sha256(path) if path.is_file() else None}
        row['status'] = 'stereo_executed' if code == 0 and row['depth_meter.npy']['exists'] else 'failed'
        (out / 'report.json').write_text(json.dumps(row, indent=2) + '\n')
        rows.append(row); print(json.dumps(row), flush=True)
    manifest['run_demo_after_sha256'] = sha256(a.repo / 'scripts/run_demo.py')
    (a.out / 'summary.json').write_text(json.dumps({**manifest, 'rows': rows}, indent=2) + '\n')


if __name__ == '__main__':
    main()
