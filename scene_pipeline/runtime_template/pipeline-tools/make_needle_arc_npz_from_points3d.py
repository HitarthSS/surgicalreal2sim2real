#!/usr/bin/env python3
"""Build a smooth needle centerline NPZ from backprojected 3D needle points.

This avoids the pose-template ambiguity in newton_rigid_needle_from_pose.py:
instead of converting the fitted circle into a quaternion and re-generating a
local template, it samples the fitted 3D circle directly between the ordered
manual points. The output can be passed to generate_newton_frame_scene.py as
--needle-npz.
"""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np


def load_points3d(path):
    rows = []
    with Path(path).open(newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            try:
                values = [float(v) for v in row[:3]]
            except ValueError:
                continue
            if len(values) == 3 and np.all(np.isfinite(values)):
                rows.append(values)
    points = np.asarray(rows, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 5:
        raise RuntimeError(f"Expected at least 5 xyz rows in {path}, got {points.shape}")
    return points


def fit_plane(points):
    origin = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - origin, full_matrices=False)
    u_axis = vh[0]
    u_axis /= np.linalg.norm(u_axis)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    v_axis = np.cross(normal, u_axis)
    v_axis /= np.linalg.norm(v_axis)
    return origin, u_axis, v_axis, normal


def fit_circle_2d(points2d):
    x = points2d[:, 0]
    y = points2d[:, 1]
    a = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    cx, cy, c = np.linalg.lstsq(a, b, rcond=None)[0]
    radius = float(np.sqrt(max(c + cx * cx + cy * cy, 0.0)))
    residual = np.abs(np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - radius)
    return np.asarray([cx, cy], dtype=np.float64), radius, residual


def fit_ordered_arc(points):
    plane_origin, u_axis, v_axis, normal = fit_plane(points)
    centered = points - plane_origin
    points2d = np.column_stack([centered @ u_axis, centered @ v_axis])
    center2d, radius, radial_residual = fit_circle_2d(points2d)
    center = plane_origin + center2d[0] * u_axis + center2d[1] * v_axis
    rel2d = np.column_stack([(points - center) @ u_axis, (points - center) @ v_axis])
    theta = np.unwrap(np.arctan2(rel2d[:, 1], rel2d[:, 0]))
    plane_residual = np.abs((points - plane_origin) @ normal)
    return {
        "center": center,
        "radius": radius,
        "u_axis": u_axis,
        "v_axis": v_axis,
        "normal": normal,
        "theta": theta,
        "radial_residual": radial_residual,
        "plane_residual": plane_residual,
    }


def set_unordered_arc_span(arc, points):
    """Set the shortest circular interval containing unordered mask samples."""
    rel2d = np.column_stack(
        [(points - arc["center"]) @ arc["u_axis"], (points - arc["center"]) @ arc["v_axis"]]
    )
    angles = np.sort(np.mod(np.arctan2(rel2d[:, 1], rel2d[:, 0]), 2.0 * np.pi))
    wrapped = np.concatenate([angles, [angles[0] + 2.0 * np.pi]])
    largest_gap = int(np.argmax(np.diff(wrapped)))
    start_index = (largest_gap + 1) % len(angles)
    ordered = np.concatenate([angles[start_index:], angles[:start_index] + 2.0 * np.pi])
    updated = dict(arc)
    updated["theta"] = ordered
    return updated


def circle_from_three_points(points):
    """Return the unique 3D circle through three non-collinear points."""
    p0, p1, p2 = np.asarray(points, dtype=np.float64)
    a = p1 - p0
    b = p2 - p0
    normal_unnormalized = np.cross(a, b)
    normal_norm = float(np.linalg.norm(normal_unnormalized))
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1.0e-12)
    if normal_norm <= 1.0e-8 * scale * scale:
        return None

    normal = normal_unnormalized / normal_norm
    denominator = 2.0 * normal_norm * normal_norm
    center = p0 + (
        np.dot(a, a) * np.cross(b, normal_unnormalized)
        + np.dot(b, b) * np.cross(normal_unnormalized, a)
    ) / denominator
    radius = float(np.linalg.norm(center - p0))
    if not np.isfinite(center).all() or not np.isfinite(radius) or radius <= 0.0:
        return None
    return center, radius, normal


def circle_residuals(points, center, radius, normal):
    """Measure plane, radial, and combined distance from a 3D circle."""
    points = np.asarray(points, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    relative = points - center
    signed_plane = relative @ normal
    projected = relative - signed_plane[:, None] * normal[None, :]
    plane = np.abs(signed_plane)
    radial = np.abs(np.linalg.norm(projected, axis=1) - float(radius))
    combined = np.hypot(plane, radial)
    return plane, radial, combined


def arc_residuals(points, arc):
    return circle_residuals(points, arc["center"], arc["radius"], arc["normal"])


def candidate_triplets(point_count, max_hypotheses, seed):
    total = math.comb(point_count, 3)
    if total <= max_hypotheses:
        return itertools.combinations(range(point_count), 3), total

    rng = np.random.default_rng(seed)
    triplets = set()
    max_attempts = max_hypotheses * 20
    attempts = 0
    while len(triplets) < max_hypotheses and attempts < max_attempts:
        triplet = tuple(sorted(rng.choice(point_count, size=3, replace=False).tolist()))
        triplets.add(triplet)
        attempts += 1
    return sorted(triplets), len(triplets)


def fit_ordered_arc_ransac(
    points,
    threshold_m,
    max_hypotheses,
    min_inliers,
    seed=0,
    min_radius_m=None,
    max_radius_m=None,
):
    """Fit a 3D circle with RANSAC, then refit it using all consensus points."""
    points = np.asarray(points, dtype=np.float64)
    if threshold_m <= 0.0:
        raise ValueError("RANSAC threshold must be positive")
    if max_hypotheses < 1:
        raise ValueError("RANSAC hypotheses must be at least 1")
    if min_inliers < 5 or min_inliers > len(points):
        raise ValueError(f"RANSAC min inliers must be in [5,{len(points)}]")

    triplets, hypothesis_count = candidate_triplets(len(points), max_hypotheses, seed)
    best_mask = None
    best_score = None
    valid_hypotheses = 0
    for indices in triplets:
        candidate = circle_from_three_points(points[list(indices)])
        if candidate is None:
            continue
        center, radius, normal = candidate
        if min_radius_m is not None and radius < min_radius_m:
            continue
        if max_radius_m is not None and radius > max_radius_m:
            continue
        valid_hypotheses += 1
        _, _, combined = circle_residuals(points, center, radius, normal)
        mask = combined <= threshold_m
        count = int(np.count_nonzero(mask))
        if count < min_inliers:
            continue
        inlier_residual = combined[mask]
        score = (count, -float(np.median(inlier_residual)), -float(np.mean(inlier_residual)))
        if best_score is None or score > best_score:
            best_score = score
            best_mask = mask

    if best_mask is None:
        raise RuntimeError(
            "RANSAC found no valid 3D needle circle with at least "
            f"{min_inliers} inliers at threshold {threshold_m:g} m"
        )

    # Recompute the least-squares plane and circle on the consensus set. A few
    # update passes allow the inlier mask to follow the refined model.
    mask = best_mask
    for _ in range(3):
        arc = fit_ordered_arc(points[mask])
        _, _, combined = arc_residuals(points, arc)
        refined_mask = combined <= threshold_m
        if np.count_nonzero(refined_mask) < min_inliers or np.array_equal(refined_mask, mask):
            break
        mask = refined_mask

    arc = fit_ordered_arc(points[mask])
    plane_residual, radial_residual, combined_residual = arc_residuals(points, arc)
    return {
        "arc": arc,
        "inlier_mask": mask,
        "plane_residual": plane_residual,
        "radial_residual": radial_residual,
        "combined_residual": combined_residual,
        "hypotheses_requested": int(hypothesis_count),
        "hypotheses_valid": int(valid_hypotheses),
    }


def adaptive_min_inliers(point_count, requested, fraction):
    if point_count < 5:
        raise ValueError("RANSAC requires at least 5 points")
    if requested < 5:
        raise ValueError("RANSAC min inliers must be at least 5")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("RANSAC min-inlier fraction must be in (0,1]")
    proportional = max(5, int(np.ceil(fraction * point_count)))
    return min(requested, proportional, point_count)


def sample_arc(arc, samples, extend_start_degrees, extend_end_degrees):
    theta = arc["theta"]
    start = float(theta[0]) - np.deg2rad(float(extend_start_degrees))
    end = float(theta[-1]) + np.deg2rad(float(extend_end_degrees))
    dense_theta = np.linspace(start, end, int(samples), dtype=np.float64)
    center = arc["center"].reshape(1, 3)
    return center + arc["radius"] * (
        np.cos(dense_theta)[:, None] * arc["u_axis"].reshape(1, 3)
        + np.sin(dense_theta)[:, None] * arc["v_axis"].reshape(1, 3)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points3d", required=True, help="CSV written by stereo_manual_points_arc_pose.py")
    parser.add_argument("--out", required=True, help="NPZ accepted by generate_newton_frame_scene.py --needle-npz")
    parser.add_argument("--report", default=None)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--tube-radius", type=float, default=0.00035)
    parser.add_argument("--extend-start-degrees", type=float, default=0.0)
    parser.add_argument("--extend-end-degrees", type=float, default=0.0)
    parser.add_argument("--ransac", action="store_true", help="reject 3D circle outliers before the final fit")
    parser.add_argument("--ransac-threshold-m", type=float, default=0.002)
    parser.add_argument("--ransac-hypotheses", type=int, default=4096)
    parser.add_argument("--ransac-min-inliers", type=int, default=5)
    parser.add_argument("--ransac-min-inlier-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--ransac-seed", type=int, default=0)
    parser.add_argument("--ransac-min-radius-m", type=float, default=None)
    parser.add_argument("--ransac-max-radius-m", type=float, default=None)
    parser.add_argument(
        "--unordered-points",
        action="store_true",
        help="derive arc span from the shortest interval covering unordered mask samples",
    )
    args = parser.parse_args()

    points = load_points3d(args.points3d)
    if args.ransac:
        effective_min_inliers = adaptive_min_inliers(
            len(points), args.ransac_min_inliers, args.ransac_min_inlier_fraction
        )
        if effective_min_inliers != args.ransac_min_inliers:
            print(
                "Adjusted RANSAC minimum from "
                f"{args.ransac_min_inliers} to {effective_min_inliers} for "
                f"{len(points)} available points."
            )
        robust_fit = fit_ordered_arc_ransac(
            points,
            threshold_m=args.ransac_threshold_m,
            max_hypotheses=args.ransac_hypotheses,
            min_inliers=effective_min_inliers,
            seed=args.ransac_seed,
            min_radius_m=args.ransac_min_radius_m,
            max_radius_m=args.ransac_max_radius_m,
        )
        arc = robust_fit["arc"]
        inlier_mask = robust_fit["inlier_mask"]
        plane_residual = robust_fit["plane_residual"]
        radial_residual = robust_fit["radial_residual"]
        combined_residual = robust_fit["combined_residual"]
    else:
        arc = fit_ordered_arc(points)
        inlier_mask = np.ones(len(points), dtype=bool)
        plane_residual, radial_residual, combined_residual = arc_residuals(points, arc)
        robust_fit = None
    if args.unordered_points:
        arc = set_unordered_arc_span(arc, points[inlier_mask])
    centerline = sample_arc(
        arc,
        args.samples,
        args.extend_start_degrees,
        args.extend_end_degrees,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        initial_world_centerline=centerline,
        needle_centerline=centerline,
        needle_states=centerline.reshape(1, *centerline.shape),
        needle_tube_radius=np.asarray(args.tube_radius, dtype=np.float64),
        tube_radius=np.asarray(args.tube_radius, dtype=np.float64),
        fit_center=arc["center"],
        fit_radius=np.asarray(arc["radius"], dtype=np.float64),
        fit_normal=arc["normal"],
        fit_theta_start_end=np.asarray([arc["theta"][0], arc["theta"][-1]], dtype=np.float64),
        source_points3d=points,
        source_inlier_mask=inlier_mask,
        source_plane_residual_m=plane_residual,
        source_radial_residual_m=radial_residual,
        source_combined_residual_m=combined_residual,
    )

    inlier_indices = np.flatnonzero(inlier_mask)
    outlier_indices = np.flatnonzero(~inlier_mask)
    report = {
        "points3d": str(Path(args.points3d).resolve()),
        "out": str(out.resolve()),
        "samples": int(args.samples),
        "tube_radius_m": float(args.tube_radius),
        "center_m": arc["center"].tolist(),
        "radius_m": float(arc["radius"]),
        "normal": arc["normal"].tolist(),
        "arc_span_degrees": float(np.rad2deg(arc["theta"][-1] - arc["theta"][0])),
        "ransac_enabled": bool(args.ransac),
        "unordered_points": bool(args.unordered_points),
        "inlier_count": int(len(inlier_indices)),
        "outlier_count": int(len(outlier_indices)),
        "inlier_indices_1based": (inlier_indices + 1).tolist(),
        "outlier_indices_1based": (outlier_indices + 1).tolist(),
        "mean_radial_residual_m": float(np.mean(radial_residual[inlier_mask])),
        "max_radial_residual_m": float(np.max(radial_residual[inlier_mask])),
        "mean_plane_residual_m": float(np.mean(plane_residual[inlier_mask])),
        "max_plane_residual_m": float(np.max(plane_residual[inlier_mask])),
        "mean_combined_residual_m": float(np.mean(combined_residual[inlier_mask])),
        "max_combined_residual_m": float(np.max(combined_residual[inlier_mask])),
    }
    if robust_fit is not None:
        report["ransac"] = {
            "threshold_m": float(args.ransac_threshold_m),
            "hypotheses_requested": int(robust_fit["hypotheses_requested"]),
            "hypotheses_valid": int(robust_fit["hypotheses_valid"]),
            "requested_min_inliers": int(args.ransac_min_inliers),
            "min_inlier_fraction": float(args.ransac_min_inlier_fraction),
            "effective_min_inliers": int(effective_min_inliers),
            "seed": int(args.ransac_seed),
            "min_radius_m": args.ransac_min_radius_m,
            "max_radius_m": args.ransac_max_radius_m,
        }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
