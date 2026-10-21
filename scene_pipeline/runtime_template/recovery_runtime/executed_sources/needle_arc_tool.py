"""Compose the existing stereo point extractor and free-radius arc fitter.

No new estimator, template size, endpoint extension or occlusion fallback.
The shared verifier decides acceptance after both native operations finish.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('left', 'right', 'depth-npy', 'K', 'left-mask', 'right-mask', 'pipeline-tools', 'out-dir'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--baseline-m', type=float, required=True)
    p.add_argument('--samples', type=int, default=96)
    p.add_argument('--depth-sample-radius', type=int, default=3)
    p.add_argument('--ransac-threshold-m', type=float, default=.002)
    p.add_argument('--ransac-min-inlier-fraction', type=float, default=2/3)
    args = p.parse_args()
    root, out = Path(args.pipeline_tools), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    point_command = [sys.executable, str(root / 'reconstruct_needle_from_masks.py')]
    for flag in ('left', 'right', 'depth-npy', 'K', 'left-mask', 'right-mask', 'baseline-m', 'samples', 'depth-sample-radius'):
        point_command += ['--' + flag, str(getattr(args, flag.replace('-', '_')))]
    point_command += ['--out-dir', str(out), '--min-valid-depth-fraction', '.7', '--min-right-mask-fraction', '.5']
    fit_command = [sys.executable, str(root / 'make_needle_arc_npz_from_points3d.py'),
                   '--points3d', str(out / 'needle_points3d.csv'), '--out', str(out / 'needle_arc.npz'),
                   '--report', str(out / 'native_arc_report.json'), '--samples', '64',
                   '--tube-radius', '.00035', '--extend-start-degrees', '0', '--extend-end-degrees', '0',
                   '--ransac', '--ransac-threshold-m', str(args.ransac_threshold_m),
                   '--ransac-hypotheses', '4096', '--ransac-min-inliers', '5',
                   '--ransac-min-inlier-fraction', str(args.ransac_min_inlier_fraction),
                   '--ransac-seed', '0', '--unordered-points']
    report = dict(status='failed', stages=[], new_estimator=False, fixed_radius_prior=False,
                  hidden_endpoint_extension=False, tube_radius_is_unmeasured_assumption=True)
    try:
        for name, command in [('existing_stereo_points', point_command), ('existing_free_radius_arc', fit_command)]:
            before = time.perf_counter()
            with (out / (name + '_stdout.txt')).open('w') as stdout, (out / (name + '_stderr.txt')).open('w') as stderr:
                result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False, timeout=100)
            report['stages'].append(dict(name=name, command=command, exit_code=result.returncode,
                                         runtime_seconds=time.perf_counter() - before))
            if result.returncode:
                raise RuntimeError(name + ' failed; no fallback')
        report['native_arc'] = json.loads((out / 'native_arc_report.json').read_text())
        report['native_points'] = json.loads((out / 'mask_backprojection_report.json').read_text())
        report['status'] = 'passed'
    except Exception as exc:
        report['error'] = str(exc)
    (out / 'quality_report.json').write_text(json.dumps(report, indent=2) + '\n')
    if report['status'] != 'passed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
