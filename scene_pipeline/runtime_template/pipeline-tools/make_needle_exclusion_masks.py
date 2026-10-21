#!/usr/bin/env python3
"""Create stereo needle exclusion masks from ordered left-image labels.

The left exclusion follows the manual pixel labels exactly. Metric
Fast-FoundationStereo depth and the calibrated rectified baseline project the
same ordered labels into the right view. These masks are consumed by SAM thread
selection so needle pixels cannot be selected as thread pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def load_points(path: str | Path) -> np.ndarray:
    points = np.loadtxt(path, delimiter=",", comments="#").astype(np.float64)
    if points.ndim != 2 or points.shape[0] != 2 or points.shape[1] < 5:
        raise RuntimeError(f"Expected ordered point CSV shape (2,N), got {points.shape}")
    points = points.T
    if not np.isfinite(points).all():
        raise RuntimeError("Needle labels contain non-finite coordinates")
    return points


def load_intrinsics(path: str | Path) -> np.ndarray:
    intrinsic = np.loadtxt(path, delimiter=",").astype(np.float64)
    if intrinsic.shape != (3, 3):
        raise RuntimeError(f"Expected K shape (3,3), got {intrinsic.shape}")
    return intrinsic


def sample_depths(
    depth: np.ndarray,
    points_xy: np.ndarray,
    radius: int,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    height, width = depth.shape
    sampled = np.full(len(points_xy), np.nan, dtype=np.float64)
    for index, (x_float, y_float) in enumerate(points_xy):
        x, y = int(round(x_float)), int(round(y_float))
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        patch = depth[y0:y1, x0:x1]
        rows, cols = np.ogrid[y0:y1, x0:x1]
        disk = (cols - x_float) ** 2 + (rows - y_float) ** 2 <= radius**2
        valid = (
            disk
            & np.isfinite(patch)
            & (patch >= min_depth_m)
            & (patch <= max_depth_m)
        )
        if np.any(valid):
            sampled[index] = float(np.median(patch[valid]))
    return sampled


def fill_depths(raw_depth: np.ndarray, min_valid_fraction: float) -> tuple[np.ndarray, float]:
    valid = np.isfinite(raw_depth)
    valid_fraction = float(np.mean(valid))
    if valid_fraction < min_valid_fraction or np.count_nonzero(valid) < 2:
        raise RuntimeError(
            f"Only {np.count_nonzero(valid)}/{len(raw_depth)} needle labels have valid depth; "
            f"required fraction is {min_valid_fraction:.3f}"
        )
    indices = np.arange(len(raw_depth))
    valid_indices = np.flatnonzero(valid)
    return np.interp(indices, valid_indices, raw_depth[valid]), valid_fraction


def draw_polyline_mask(shape: tuple[int, int], points_xy: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px < 0:
        raise ValueError("exclusion radius must be nonnegative")
    mask = np.zeros(shape, dtype=np.uint8)
    polyline = np.round(points_xy).astype(np.int32).reshape(-1, 1, 2)
    thickness = max(1, 2 * int(radius_px) + 1)
    cv2.polylines(mask, [polyline], False, 255, thickness, cv2.LINE_8)
    cv2.circle(mask, tuple(polyline[0, 0]), radius_px, 255, -1, cv2.LINE_8)
    cv2.circle(mask, tuple(polyline[-1, 0]), radius_px, 255, -1, cv2.LINE_8)
    return mask


def save_mask(path: str | Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask):
        raise RuntimeError(f"Failed to write mask: {path}")


def save_overlay(path: str | Path, image: np.ndarray, mask: np.ndarray) -> None:
    overlay = image.copy()
    selected = mask > 0
    color = np.asarray([20, 20, 255], dtype=np.float32)
    overlay[selected] = (
        0.45 * overlay[selected].astype(np.float32) + 0.55 * color
    ).astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), overlay):
        raise RuntimeError(f"Failed to write overlay: {path}")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--points", required=True)
    parser.add_argument("--depth-npy", required=True)
    parser.add_argument("--K", required=True)
    parser.add_argument("--baseline-m", type=float, required=True)
    parser.add_argument("--sample-radius", type=int, default=8)
    parser.add_argument("--exclusion-radius-px", type=int, default=1)
    parser.add_argument("--min-depth-m", type=float, default=0.03)
    parser.add_argument("--max-depth-m", type=float, default=0.50)
    parser.add_argument("--min-valid-depth-fraction", type=float, default=0.70)
    parser.add_argument("--left-mask-out", required=True)
    parser.add_argument("--right-mask-out", required=True)
    parser.add_argument("--left-overlay-out", required=True)
    parser.add_argument("--right-overlay-out", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    left = load_image(args.left)
    right = load_image(args.right)
    if left.shape != right.shape:
        raise RuntimeError(f"Stereo image shape mismatch: {left.shape} vs {right.shape}")
    points_left = load_points(args.points)
    height, width = left.shape[:2]
    inside = (
        (points_left[:, 0] >= 0)
        & (points_left[:, 0] < width)
        & (points_left[:, 1] >= 0)
        & (points_left[:, 1] < height)
    )
    if not np.all(inside):
        raise RuntimeError("Needle labels fall outside the selected left image")

    depth = np.squeeze(np.load(args.depth_npy)).astype(np.float64)
    if depth.shape != (height, width):
        raise RuntimeError(
            f"Depth shape {depth.shape} does not match image shape {(height, width)}"
        )
    intrinsic = load_intrinsics(args.K)
    raw_depth = sample_depths(
        depth,
        points_left,
        args.sample_radius,
        args.min_depth_m,
        args.max_depth_m,
    )
    depth_m, valid_fraction = fill_depths(
        raw_depth, args.min_valid_depth_fraction
    )
    disparity = float(intrinsic[0, 0]) * float(args.baseline_m) / depth_m
    points_right = points_left.copy()
    points_right[:, 0] -= disparity

    left_mask = draw_polyline_mask((height, width), points_left, args.exclusion_radius_px)
    right_mask = draw_polyline_mask((height, width), points_right, args.exclusion_radius_px)
    save_mask(args.left_mask_out, left_mask)
    save_mask(args.right_mask_out, right_mask)
    save_overlay(args.left_overlay_out, left, left_mask)
    save_overlay(args.right_overlay_out, right, right_mask)

    report = {
        "algorithm": "manual_left_needle_depth_projected_stereo_exclusion",
        "points_file": str(Path(args.points).resolve()),
        "points_file_sha256": file_sha256(args.points),
        "point_count": int(len(points_left)),
        "valid_depth_fraction": valid_fraction,
        "depth_m_min_median_max": [
            float(np.min(depth_m)),
            float(np.median(depth_m)),
            float(np.max(depth_m)),
        ],
        "disparity_px_min_median_max": [
            float(np.min(disparity)),
            float(np.median(disparity)),
            float(np.max(disparity)),
        ],
        "exclusion_radius_px": int(args.exclusion_radius_px),
        "left_excluded_area_px": int(np.count_nonzero(left_mask)),
        "right_excluded_area_px": int(np.count_nonzero(right_mask)),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"ERROR: needle exclusion mask generation failed: {error}") from None
