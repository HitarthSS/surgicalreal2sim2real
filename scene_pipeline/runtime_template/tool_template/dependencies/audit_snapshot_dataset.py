"""Read-only stereo snapshot intake: hashes, decoding, calibration and timing.

Does not estimate geometry, infer calibration units, or construct trajectories.
All supplied snapshots remain together in one explicitly named collection.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
from PIL import Image


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def timestamp_ns(header):
    """Prefer integer ROS timestamps to avoid epoch-float cancellation."""
    if not isinstance(header, dict):
        return None
    sec, nsec = header.get('sec'), header.get('nanosec')
    if (type(sec) is int and type(nsec) is int and sec >= 0
            and 0 <= nsec < 1_000_000_000):
        return sec * 1_000_000_000 + nsec
    stamp = header.get('stamp')
    if type(stamp) in (int, float) and math.isfinite(stamp) and stamp >= 0:
        return round(stamp * 1_000_000_000)
    return None


def calibration_check(calibration, dimensions, baseline_units):
    failures, unresolved = [], []
    values = {}
    try:
        p1, p2, q = [np.asarray(calibration[k], dtype=float) for k in ('P1', 'P2', 'Q')]
        if (p1.shape != (3, 4) or p2.shape != (3, 4) or q.shape != (4, 4)
                or not all(np.isfinite(a).all() for a in (p1, p2, q))
                or min(p1[0, 0], p1[1, 1], p2[0, 0], p2[1, 1]) <= 0):
            raise ValueError('invalid matrix shape, value or focal length')
        if not np.allclose(p1[:, :3], p2[:, :3], rtol=1e-5, atol=1e-4):
            failures.append('unsupported_different_rectified_intrinsics')
        if (not np.allclose(p1[:, 3], 0, atol=1e-6)
                or not np.allclose(p2[1:, 3], 0, atol=1e-6)
                or not np.allclose(p1[2, :], [0, 0, 1, 0], atol=1e-6)):
            failures.append('unsupported_rectified_projection_convention')
        baseline = float(-p2[0, 3] / p2[0, 0])
        if baseline <= 0 or not math.isfinite(baseline):
            failures.append('invalid_stereo_baseline')
        if not np.isclose(float(calibration['baseline']), baseline, rtol=1e-5, atol=1e-6):
            failures.append('baseline_projection_disagreement')
        if not np.isclose(q[3, 2] * baseline, 1, rtol=1e-5, atol=1e-5):
            failures.append('Q_projection_baseline_disagreement')
        for key, actual in [('fx', p1[0, 0]), ('fy', p1[1, 1]),
                            ('cx', p1[0, 2]), ('cy', p1[1, 2])]:
            if not np.isclose(float(calibration[key]), actual, rtol=1e-5, atol=1e-4):
                failures.append(key + '_projection_disagreement')
        if len(set(dimensions.values())) != 1:
            failures.append('left_right_image_dimensions_differ')
        if any(tuple(calibration['image_size']) != size for size in dimensions.values()):
            failures.append('declared_image_dimensions_differ')
        if baseline_units is None:
            unresolved.append('physical_baseline_units_unconfirmed')
        baseline_m = None if baseline_units is None else baseline * (0.001 if baseline_units == 'mm' else 1.)
        if baseline_m is not None and not .001 <= baseline_m <= .1:
            failures.append('baseline_outside_declared_1_to_100mm_range')
        values.update(K=p1[:, :3].tolist(), baseline_recorded=baseline,
                      baseline_units=baseline_units, baseline_m=baseline_m)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        failures.append('invalid_calibration: ' + str(error))
    return {'failures': failures, 'unresolved': unresolved, **values,
            'internal_consistency_only': True, 'calibration_independently_validated': False}


def audit(root, collection_id, baseline_units=None, sync_screen_ms=10.):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError('Snapshot root is not a directory')
    if baseline_units not in (None, 'm', 'mm'):
        raise ValueError('Units must be explicitly m or mm')
    if not math.isfinite(sync_screen_ms) or sync_screen_ms < 0:
        raise ValueError('Invalid screening threshold')
    rows = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        row = {'case_id': folder.name, 'collection_id': collection_id,
               'split': 'development', 'files': {}, 'failures': [], 'unresolved': []}
        dimensions = {}
        for name in ('left_rect.png', 'right_rect.png', 'calibration.json', 'psm_state.json'):
            path = folder / name
            if not path.is_file():
                row['failures' if name != 'psm_state.json' else 'unresolved'].append('missing_' + name)
                continue
            row['files'][name] = {'path': str(path), 'bytes': path.stat().st_size,
                                 'sha256': digest(path)}
            if name.endswith('.png'):
                try:
                    with Image.open(path) as im:
                        im.load()
                        dimensions[name] = im.size
                        row['files'][name].update(size=list(im.size), mode=im.mode, decoded=True)
                except (OSError, ValueError) as error:
                    row['failures'].append('image_decode_failed: ' + name + ': ' + str(error))
        cal = None
        try:
            cal = json.loads((folder / 'calibration.json').read_text())
            row['calibration'] = calibration_check(cal, dimensions, baseline_units)
            row['failures'].extend(row['calibration']['failures'])
            row['unresolved'].extend(row['calibration']['unresolved'])
            stamps = {side: timestamp_ns(cal.get('image_headers', {}).get(side)) for side in ('left', 'right')}
            row['image_timestamps_ns'] = stamps
            row['stereo_offset_ms'] = None if None in stamps.values() else abs(stamps['left'] - stamps['right']) / 1_000_000
            if row['stereo_offset_ms'] is None:
                row['unresolved'].append('missing_stereo_timestamps')
            elif row['stereo_offset_ms'] > sync_screen_ms:
                row['unresolved'].append('stereo_offset_exceeds_screening_threshold_motion_unknown')
        except (OSError, ValueError, TypeError) as error:
            row['failures'].append('calibration_read_failed: ' + str(error))
        row['robot_image_offsets_ms'] = {}
        state_path = folder / 'psm_state.json'
        if state_path.is_file():
            try:
                state = json.loads(state_path.read_text())
                arms = state.get('arms', {})
                row['robot_arms'] = sorted(arms)
                image_ns = row.get('image_timestamps_ns', {}).get('left')
                for arm, arm_state in arms.items():
                    for key in ('joint_state', 'jaw_state', 'cartesian_state'):
                        message = arm_state.get(key)
                        stamp = timestamp_ns(message.get('header')) if isinstance(message, dict) else None
                        row['robot_image_offsets_ms'][arm + '/' + key] = None if stamp is None or image_ns is None else (stamp - image_ns) / 1_000_000
            except (ValueError, TypeError, AttributeError) as error:
                row['failures'].append('robot_state_read_failed: ' + str(error))
        row.update(pixel_reconstruction_inputs_decoded=not row['failures'],
                   metric_calibration_ready=not row['failures'] and baseline_units is not None,
                   synchronized_motion_confirmed=False, policy_demonstration=False,
                   training_eligible=False)
        rows.append(row)
    offsets = [row['stereo_offset_ms'] for row in rows if row.get('stereo_offset_ms') is not None]
    return {'schema': 'ismr_snapshot_intake_v1', 'root': str(root), 'collection_id': collection_id,
            'split': 'development', 'snapshot_count': len(rows),
            'decoded_pair_count': sum(len([f for f in r['files'].values() if f.get('decoded')]) == 2 for r in rows),
            'cases_with_format_failures': sum(bool(r['failures']) for r in rows),
            'screening_threshold_ms': sync_screen_ms, 'threshold_is_validated_acceptance_criterion': False,
            'stereo_offset_ms': None if not offsets else {'min': min(offsets), 'median': statistics.median(offsets), 'max': max(offsets)},
            'unresolved_counts': dict(Counter(f for r in rows for f in r['unresolved'])),
            'independent_accuracy_measured': False, 'source_completeness_verified': False,
            'policy_demonstration_count': 0, 'rows': rows,
            'limitations': ['Snapshot metadata does not establish motion or action alignment.',
                            'Internal matrix agreement does not establish camera calibration accuracy.',
                            'All snapshots are grouped together; physical arrangement/session identities require review.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--collection-id', required=True)
    p.add_argument('--baseline-units', choices=('m', 'mm'))
    p.add_argument('--sync-screen-ms', type=float, default=10.)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('Preserve prior audits: output already exists')
    result = audit(a.root, a.collection_id, a.baseline_units, a.sync_screen_ms)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'limitations')}, indent=2))


if __name__ == '__main__':
    main()
