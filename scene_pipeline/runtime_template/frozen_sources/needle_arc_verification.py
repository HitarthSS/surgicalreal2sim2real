"""Reject inconsistent/poorly supported existing arc fits; no hidden repair."""
from pathlib import Path

import cv2
import numpy as np

from reconstruction_verification import load_base, verify_needle, write_curve_obj


def verify_needle_arc(inputs, output, tools, backend_report, limits):
    out = Path(output)
    point_check = verify_needle(inputs, out, tools, backend_report, limits)
    failures = list(point_check['failures'])
    path = out / 'needle_arc.npz'
    if not path.exists():
        return dict(accepted=False, failures=failures + ['missing_needle_arc'], metrics=point_check['metrics'],
                    independent_accuracy_measured=False)
    with np.load(path, allow_pickle=False) as data:
        points = data['needle_centerline'].copy()
        source = data['source_points3d'].copy()
        inliers = data['source_inlier_mask'].copy()
        center, normal = data['fit_center'].copy(), data['fit_normal'].copy()
        radius = float(data['fit_radius'])
        span = float(np.diff(data['fit_theta_start_end'])[0])
        residual = data['source_combined_residual_m'].copy()
    if (points.shape != (64, 3) or not np.isfinite(points).all() or
            not np.isfinite(center).all() or not np.isfinite(normal).all() or
            not np.isfinite([radius, span]).all() or radius <= 0 or
            source.ndim != 2 or source.shape[1:] != (3,) or
            inliers.shape != (len(source),) or inliers.dtype != bool or
            residual.shape != (len(source),) or not np.isfinite(residual).all()):
        return dict(accepted=False, failures=failures + ['invalid_arc_arrays'], metrics={},
                    independent_accuracy_measured=False)
    report = backend_report.get('native_arc', {})
    fraction = float(inliers.mean())
    if inliers.sum() < 5 or fraction < 2/3:
        failures.append('insufficient_arc_consensus')
    if not np.isclose(np.linalg.norm(normal), 1., atol=1e-5):
        failures.append('invalid_arc_normal')
    if not 45 <= np.degrees(abs(span)) <= 360:
        failures.append('insufficient_arc_angular_support')
    threshold = report.get('ransac', {}).get('threshold_m', 0.)
    if not .0003 <= threshold <= .002 or np.any(residual[inliers] > threshold + 1e-8):
        failures.append('arc_consensus_residual_failed')
    centered = source[inliers] - source[inliers].mean(axis=0) if inliers.any() else np.zeros((3, 3))
    singular = np.linalg.svd(centered, compute_uv=False)
    ratio = float(singular[1] / max(singular[0], 1e-12)) if len(singular) >= 2 else 0.
    if ratio < .05:
        failures.append('arc_points_nearly_collinear')
    base = load_base(tools)
    K = base.load_intrinsics(inputs['K'])
    left, right = base.load_rgb(inputs['left']), base.load_rgb(inputs['right'])
    z = points[:, 2]
    if np.any((z < limits['min_depth_m']) | (z > limits['max_depth_m'])):
        return dict(accepted=False, failures=failures + ['arc_depth_outside_limits'], metrics={},
                    independent_accuracy_measured=False)
    if (not np.allclose(np.linalg.norm(points-center, axis=1), radius, atol=1e-8) or
            not np.allclose((points-center) @ normal, 0., atol=1e-8)):
        failures.append('arc_geometry_inconsistent_with_fit')
    def support(image, mask_path, baseline):
        xy = np.c_[(points[:, 0] - baseline) * K[0, 0] / z + K[0, 2],
                   points[:, 1] * K[1, 1] / z + K[1, 2]]
        mask = base.load_mask(mask_path, image.shape[:2])
        dist = cv2.distanceTransform((~mask).astype('uint8'), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        distances = cv2.remap(dist, xy[:, 0].astype('float32')[None], xy[:, 1].astype('float32')[None],
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1e6).ravel()
        return xy, distances
    xy, ld = support(left, inputs['needle_left_mask'], 0.)
    rxy, rd = support(right, inputs['needle_right_mask'], inputs['baseline_m'])
    ls, rs = ld <= limits['right_tolerance_px'], rd <= limits['right_tolerance_px']
    if float(ls.mean()) < .5 or float(rs.mean()) < .5:
        failures.append('fitted_arc_stereo_support_failed')
    # Acceptance tolerance cannot turn a nearby hidden bridge into visible support.
    inferred = (ld > 1.5) | (rd > 1.5)
    write_curve_obj(out / 'needle_arc_centerline.obj', points, inferred)
    np.savez_compressed(out / 'shared_needle_arc.npz', points_m=points, inferred=inferred,
                        visible_stereo_support=~inferred, reliability=np.where(inferred, 0., 1.),
                        fit_center_m=center, fit_normal=normal, fit_radius_m=radius,
                        tube_radius_m=.00035, independent_accuracy_measured=False)
    for label, image, pixel in [('left', left, xy), ('right', right, rxy)]:
        overlay = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for i in range(len(points) - 1):
            color = (0, 165, 255) if inferred[i] or inferred[i+1] else (0, 200, 0)
            cv2.line(overlay, tuple(np.rint(pixel[i]).astype(int)), tuple(np.rint(pixel[i+1]).astype(int)), color, 2)
        cv2.imwrite(str(out / ('shared_needle_' + label + '_overlay.png')), overlay)
    metrics = dict(point_check['metrics'], radius_estimate_m=radius, span_degrees=float(np.degrees(abs(span))),
                   inlier_fraction=fraction, planar_singular_ratio=ratio,
                   fitted_left_support_fraction=float(ls.mean()), fitted_right_support_fraction=float(rs.mean()),
                   inferred_samples=int(inferred.sum()), visible_label_tolerance_px=1.5)
    return dict(accepted=not failures, failures=list(dict.fromkeys(failures)), metrics=metrics,
                geometry_is_estimated=True, pose_kind='estimated circle center and plane, not full needle pose',
                independent_accuracy_measured=False, physics_validated=False, fixed_radius_prior=False,
                endpoint_extension=False, hidden_geometry_is_inferred=True,
                reliability_is_support_label_not_probability=True, tube_radius_is_unmeasured_assumption=True)
