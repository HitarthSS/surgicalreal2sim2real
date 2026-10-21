#!/usr/bin/env python3
"""Backproject fine-tuned SAM 3 needle-mask fragments into metric 3D points."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from reconstruct_thread_from_files import (
    backproject,
    clean_depth,
    load_intrinsics,
    load_mask,
    load_rgb,
    longest_skeleton_path,
    resample_path,
    right_mask_distances,
    sample_depth,
)


def component_paths(
    mask: np.ndarray, min_area_px: int, min_path_length_px: float
) -> tuple[list[np.ndarray], list[dict]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    paths = []
    reports = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        component = labels == label
        try:
            path, metrics = longest_skeleton_path(component)
        except RuntimeError:
            continue
        if metrics["ordered_path_length_px"] < min_path_length_px:
            continue
        paths.append(path)
        reports.append({"label": label, "area_px": area, **metrics})
    if not paths:
        raise RuntimeError("Needle mask has no component with a usable centerline")
    return paths, reports


def allocate_samples(paths: list[np.ndarray], total: int, minimum: int = 8) -> list[int]:
    if total < minimum * len(paths):
        total = minimum * len(paths)
    lengths = np.asarray(
        [np.linalg.norm(np.diff(path, axis=0), axis=1).sum() for path in paths],
        dtype=np.float64,
    )
    raw = total * lengths / lengths.sum()
    counts = np.maximum(minimum, np.floor(raw).astype(int))
    while counts.sum() < total:
        counts[int(np.argmax(raw - counts))] += 1
    return counts.tolist()


def save_overlays(
    out_dir: Path,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    paths_rc: np.ndarray,
    right_xy: np.ndarray,
    accepted: np.ndarray,
) -> None:
    left = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR)
    right = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR)
    for (row, col), right_point, keep in zip(paths_rc, right_xy, accepted):
        color = (0, 255, 0) if keep else (0, 0, 255)
        cv2.circle(left, (int(round(col)), int(round(row))), 2, color, -1)
        cv2.circle(
            right,
            (int(round(right_point[0])), int(round(right_point[1]))),
            2,
            color,
            -1,
        )
    cv2.imwrite(str(out_dir / "left_mask_points_overlay.png"), left)
    cv2.imwrite(str(out_dir / "right_mask_validation_overlay.png"), right)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--left-mask", required=True)
    parser.add_argument("--right-mask", required=True)
    parser.add_argument("--depth-npy", required=True)
    parser.add_argument("--K", required=True)
    parser.add_argument("--baseline-m", type=float, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--min-component-area-px", type=int, default=12)
    parser.add_argument("--min-path-length-px", type=float, default=12.0)
    parser.add_argument("--depth-sample-radius", type=int, default=3)
    parser.add_argument("--min-depth-m", type=float, default=0.03)
    parser.add_argument("--max-depth-m", type=float, default=0.50)
    parser.add_argument("--min-valid-depth-fraction", type=float, default=0.55)
    parser.add_argument("--right-mask-tolerance-px", type=float, default=8.0)
    parser.add_argument("--min-right-mask-fraction", type=float, default=0.50)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "mask_backprojection_report.json"
    points_path = out_dir / "needle_points3d.csv"

    left = load_rgb(args.left)
    right = load_rgb(args.right)
    if left.shape != right.shape:
        raise RuntimeError(f"Stereo image shape mismatch: {left.shape} vs {right.shape}")
    left_mask = load_mask(args.left_mask, left.shape[:2])
    right_mask = load_mask(args.right_mask, right.shape[:2])
    depth = np.squeeze(np.load(args.depth_npy).astype(np.float64))
    if depth.shape != left.shape[:2]:
        raise RuntimeError(f"Depth shape {depth.shape} does not match image {left.shape[:2]}")
    intrinsic = load_intrinsics(args.K)

    paths, component_report = component_paths(
        left_mask, args.min_component_area_px, args.min_path_length_px
    )
    sample_counts = allocate_samples(paths, args.samples)
    sampled_parts = []
    depth_parts = []
    raw_valid_parts = []
    for path, sample_count in zip(paths, sample_counts):
        sampled = resample_path(path, sample_count)
        raw_depth, _ = sample_depth(
            depth,
            left_mask,
            sampled,
            args.depth_sample_radius,
            args.min_depth_m,
            args.max_depth_m,
        )
        cleaned, depth_inliers, _ = clean_depth(raw_depth, 7, 11)
        sampled_parts.append(sampled)
        depth_parts.append(cleaned)
        raw_valid_parts.append(np.isfinite(raw_depth) & depth_inliers)

    sampled_rc = np.concatenate(sampled_parts, axis=0)
    cleaned_depth = np.concatenate(depth_parts)
    raw_valid = np.concatenate(raw_valid_parts)
    points_m = backproject(sampled_rc, cleaned_depth, intrinsic)
    right_distance, right_xy = right_mask_distances(
        sampled_rc,
        cleaned_depth,
        right_mask,
        float(intrinsic[0, 0]),
        args.baseline_m,
    )
    stereo_valid = right_distance <= args.right_mask_tolerance_px
    accepted = raw_valid & stereo_valid
    valid_depth_fraction = float(np.mean(raw_valid))
    right_mask_fraction = float(np.mean(stereo_valid[raw_valid])) if np.any(raw_valid) else 0.0

    failures = []
    if valid_depth_fraction < args.min_valid_depth_fraction:
        failures.append(
            f"valid depth fraction {valid_depth_fraction:.3f} is below "
            f"{args.min_valid_depth_fraction:.3f}"
        )
    if right_mask_fraction < args.min_right_mask_fraction:
        failures.append(
            f"right-mask fraction {right_mask_fraction:.3f} is below "
            f"{args.min_right_mask_fraction:.3f}"
        )
    if np.count_nonzero(accepted) < 5:
        failures.append("fewer than five stereo-validated 3D needle samples")

    save_overlays(out_dir, left, right, sampled_rc, right_xy, accepted)
    report = {
        "status": "failed" if failures else "passed",
        "algorithm": "sam3_mask_fragments_foundationstereo_backprojection",
        "fallbacks": False,
        "component_count": len(paths),
        "components": component_report,
        "sample_count": int(len(sampled_rc)),
        "accepted_count": int(np.count_nonzero(accepted)),
        "valid_depth_fraction": valid_depth_fraction,
        "right_mask_fraction": right_mask_fraction,
        "failures": failures,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        points_path.unlink(missing_ok=True)
        raise RuntimeError("Needle mask reconstruction failed: " + "; ".join(failures))

    with points_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["x", "y", "z"])
        writer.writerows(points_m[accepted])
    print(json.dumps(report, indent=2))
    print(f"3D needle points: {points_path}")


if __name__ == "__main__":
    main()
