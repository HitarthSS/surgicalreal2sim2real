#!/usr/bin/env python3
"""Reconstruct a thread from ordered manual points and metric stereo depth.

This module is intentionally independent from the automatic SAM/skeleton thread
reconstructor. The selected pipeline method determines which module writes the
shared ``thread_samples.npz`` scene input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


ALGORITHM = "manual_points_foundationstereo_backprojection"


def load_points(path: str | Path) -> np.ndarray:
    points = np.loadtxt(path, delimiter=",").astype(np.float64)
    if points.ndim != 2:
        raise RuntimeError(f"Expected a 2D point CSV, got shape {points.shape}")
    if points.shape[0] == 2:
        points = points.T
    elif points.shape[1] != 2:
        raise RuntimeError(
            f"Expected point CSV shape (2,N) or (N,2), got {points.shape}"
        )
    if len(points) < 4:
        raise RuntimeError(f"Need at least 4 ordered thread points, got {len(points)}")
    if not np.all(np.isfinite(points)):
        raise RuntimeError("Thread points contain NaN or infinity")

    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate([[True], segment > 1.0e-6])
    points = points[keep]
    if len(points) < 4:
        raise RuntimeError("Need at least 4 distinct ordered thread points")
    return points


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def load_intrinsics(path: str | Path) -> np.ndarray:
    intrinsic = np.loadtxt(path, delimiter=",").astype(np.float64)
    if intrinsic.shape != (3, 3):
        raise RuntimeError(f"Expected K shape (3,3), got {intrinsic.shape}")
    if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
        raise RuntimeError("Camera focal lengths must be positive")
    return intrinsic


def resample_ordered_points(points_xy: np.ndarray, count: int) -> np.ndarray:
    if count < len(points_xy):
        raise RuntimeError(
            f"Sample count {count} must be at least the {len(points_xy)} manual points"
        )
    segment = np.linalg.norm(np.diff(points_xy, axis=0), axis=1)
    distance = np.concatenate([[0.0], np.cumsum(segment)])
    if distance[-1] <= 0.0:
        raise RuntimeError("Manual thread path has zero length")
    target = np.linspace(0.0, distance[-1], count)
    return np.column_stack(
        [PchipInterpolator(distance, points_xy[:, axis])(target) for axis in range(2)]
    )


def sample_nearest_depths(
    depth: np.ndarray,
    points_xy: np.ndarray,
    radius: int,
    nearest_count: int,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    if radius < 0:
        raise RuntimeError("Depth sample radius cannot be negative")
    if nearest_count < 1:
        raise RuntimeError("Depth nearest count must be positive")

    height, width = depth.shape
    sampled = np.full(len(points_xy), np.nan, dtype=np.float64)
    support = np.zeros(len(points_xy), dtype=np.float64)
    for index, (col_f, row_f) in enumerate(points_xy):
        col = int(round(col_f))
        row = int(round(row_f))
        row0, row1 = max(0, row - radius), min(height, row + radius + 1)
        col0, col1 = max(0, col - radius), min(width, col + radius + 1)
        rows, cols = np.mgrid[row0:row1, col0:col1]
        local = depth[row0:row1, col0:col1]
        distance_sq = (rows - row_f) ** 2 + (cols - col_f) ** 2
        valid = (
            (distance_sq <= radius**2)
            & np.isfinite(local)
            & (local >= min_depth_m)
            & (local <= max_depth_m)
        )
        if not np.any(valid):
            continue
        values = local[valid]
        distances = distance_sq[valid]
        order = np.argsort(distances, kind="stable")[:nearest_count]
        sampled[index] = float(np.median(values[order]))
        support[index] = len(order) / nearest_count
    return sampled, support


def odd_window(requested: int, length: int, minimum: int) -> int:
    window = min(requested, length if length % 2 else length - 1)
    if window % 2 == 0:
        window -= 1
    return window if window >= minimum else 0


def clean_depth(
    raw_depth: np.ndarray, median_window: int, smooth_window: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    valid = np.isfinite(raw_depth)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) < 2:
        raise RuntimeError("Fewer than two valid manual thread depth samples")

    sample_index = np.arange(len(raw_depth))
    filled = np.interp(sample_index, valid_indices, raw_depth[valid])
    med_window = odd_window(median_window, len(filled), 3)
    local_median = (
        median_filter(filled, size=med_window, mode="nearest")
        if med_window
        else filled
    )
    residual = np.abs(raw_depth - local_median)
    finite_residual = residual[valid]
    residual_median = float(np.median(finite_residual))
    residual_mad = float(np.median(np.abs(finite_residual - residual_median)))
    threshold = max(0.003, residual_median + 4.0 * 1.4826 * residual_mad)
    accepted = valid & (residual <= threshold)
    accepted_indices = np.flatnonzero(accepted)
    if len(accepted_indices) < 2:
        raise RuntimeError("Depth outlier rejection removed all manual thread samples")

    cleaned = np.interp(sample_index, accepted_indices, raw_depth[accepted])
    if med_window:
        cleaned = median_filter(cleaned, size=med_window, mode="nearest")
    smooth = odd_window(smooth_window, len(cleaned), 5)
    if smooth:
        cleaned = savgol_filter(cleaned, window_length=smooth, polyorder=2, mode="interp")
    return cleaned, accepted, {
        "raw_valid_count": int(np.count_nonzero(valid)),
        "depth_outlier_count": int(np.count_nonzero(valid & ~accepted)),
        "depth_outlier_threshold_m": float(threshold),
    }


def backproject(points_xy: np.ndarray, depth_m: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    x = (points_xy[:, 0] - cx) * depth_m / fx
    y = (points_xy[:, 1] - cy) * depth_m / fy
    return np.column_stack([x, y, depth_m])


def draw_overlays(
    left: np.ndarray,
    right: np.ndarray,
    controls_xy: np.ndarray,
    path_xy: np.ndarray,
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    baseline_m: float,
    out_dir: Path,
) -> float:
    left_overlay = left.copy()
    polyline = np.round(path_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(left_overlay, [polyline], False, (255, 255, 0), 2, cv2.LINE_AA)
    for index, point in enumerate(np.round(controls_xy).astype(np.int32), start=1):
        cv2.circle(left_overlay, tuple(point), 3, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(
            left_overlay,
            str(index),
            tuple(point + np.asarray([4, -4])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out_dir / "manual_left_overlay.png"), left_overlay)

    disparity = float(intrinsic[0, 0]) * baseline_m / depth_m
    right_xy = path_xy.copy()
    right_xy[:, 0] -= disparity
    height, width = right.shape[:2]
    in_bounds = (
        (right_xy[:, 0] >= 0.0)
        & (right_xy[:, 0] < width)
        & (right_xy[:, 1] >= 0.0)
        & (right_xy[:, 1] < height)
    )
    right_overlay = right.copy()
    valid_points = np.round(right_xy[in_bounds]).astype(np.int32).reshape(-1, 1, 2)
    if len(valid_points) > 1:
        cv2.polylines(right_overlay, [valid_points], False, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "manual_right_projection_overlay.png"), right_overlay)
    return float(np.mean(in_bounds))


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--points", required=True)
    parser.add_argument("--depth-npy", required=True)
    parser.add_argument("--K", required=True)
    parser.add_argument("--baseline-m", type=float, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--depth-sample-radius", type=int, default=3)
    parser.add_argument("--depth-nearest-count", type=int, default=5)
    parser.add_argument("--min-depth-m", type=float, default=0.03)
    parser.add_argument("--max-depth-m", type=float, default=0.50)
    parser.add_argument("--min-valid-depth-fraction", type=float, default=0.70)
    parser.add_argument("--depth-median-window", type=int, default=9)
    parser.add_argument("--depth-smooth-window", type=int, default=21)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "quality_report.json"
    left = load_image(args.left)
    right = load_image(args.right)
    if left.shape != right.shape:
        raise RuntimeError(f"Stereo image shape mismatch: {left.shape} vs {right.shape}")
    controls = load_points(args.points)
    height, width = left.shape[:2]
    inside = (
        (controls[:, 0] >= 0.0)
        & (controls[:, 0] < width)
        & (controls[:, 1] >= 0.0)
        & (controls[:, 1] < height)
    )
    if not np.all(inside):
        bad = (np.flatnonzero(~inside) + 1).tolist()
        raise RuntimeError(f"Manual thread point(s) outside the left image: {bad}")

    depth = np.squeeze(np.load(args.depth_npy).astype(np.float64))
    if depth.shape != (height, width):
        raise RuntimeError(f"Depth shape {depth.shape} does not match image {(height, width)}")
    intrinsic = load_intrinsics(args.K)
    path_xy = resample_ordered_points(controls, args.samples)
    raw_depth, support = sample_nearest_depths(
        depth,
        path_xy,
        args.depth_sample_radius,
        args.depth_nearest_count,
        args.min_depth_m,
        args.max_depth_m,
    )
    valid_fraction = float(np.mean(np.isfinite(raw_depth)))
    failures = []
    if valid_fraction < args.min_valid_depth_fraction:
        failures.append(
            f"valid depth fraction {valid_fraction:.3f} is below "
            f"{args.min_valid_depth_fraction:.3f}"
        )

    cleaned_depth, accepted, depth_metrics = clean_depth(
        raw_depth, args.depth_median_window, args.depth_smooth_window
    )
    points_m = backproject(path_xy, cleaned_depth, intrinsic)
    right_in_bounds_fraction = draw_overlays(
        left,
        right,
        controls,
        path_xy,
        cleaned_depth,
        intrinsic,
        args.baseline_m,
        out_dir,
    )
    reliability = np.clip(support, 0.0, 1.0) * accepted.astype(np.float64)
    report = {
        "status": "failed" if failures else "passed",
        "algorithm": ALGORITHM,
        "fallbacks": False,
        "output_units": "millimeters",
        "inputs": {
            "left": str(Path(args.left).resolve()),
            "right": str(Path(args.right).resolve()),
            "manual_points": str(Path(args.points).resolve()),
            "depth_npy": str(Path(args.depth_npy).resolve()),
            "K": str(Path(args.K).resolve()),
            "baseline_m": float(args.baseline_m),
        },
        "metrics": {
            **depth_metrics,
            "manual_control_point_count": int(len(controls)),
            "sample_count": int(len(path_xy)),
            "manual_path_length_px": float(
                np.sum(np.linalg.norm(np.diff(controls, axis=0), axis=1))
            ),
            "valid_depth_fraction": valid_fraction,
            "right_projection_in_bounds_fraction": right_in_bounds_fraction,
            "depth_m_min_median_max": [
                float(np.min(cleaned_depth)),
                float(np.median(cleaned_depth)),
                float(np.max(cleaned_depth)),
            ],
        },
        "thresholds": {
            "depth_range_m": [float(args.min_depth_m), float(args.max_depth_m)],
            "min_valid_depth_fraction": float(args.min_valid_depth_fraction),
        },
        "failures": failures,
    }
    write_report(report_path, report)
    if failures:
        raise RuntimeError(
            "Manual thread reconstruction failed calibrated quality checks. "
            f"See {report_path}: {'; '.join(failures)}"
        )

    output = out_dir / "thread_samples.npz"
    np.savez(
        output,
        u=np.linspace(0.0, 1.0, len(points_m)),
        points=points_m * 1000.0,
        points_m=points_m,
        reliability=reliability,
        pixel_xy=path_xy,
        control_pixel_xy=controls,
        depth_m=cleaned_depth,
        algorithm=np.asarray(ALGORITHM),
    )
    print(json.dumps(report["metrics"], indent=2))
    print(f"Thread samples: {output}")
    print(f"Quality report: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        report_path = None
        if "--out" in sys.argv:
            position = sys.argv.index("--out")
            if position + 1 < len(sys.argv):
                report_path = Path(sys.argv[position + 1]) / "quality_report.json"
        if report_path is not None and not report_path.exists():
            write_report(
                report_path,
                {
                    "status": "failed",
                    "algorithm": ALGORITHM,
                    "fallbacks": False,
                    "failures": [str(error)],
                },
            )
        raise
