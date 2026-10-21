#!/usr/bin/env python3
"""Run the original thread backend with SAM3 masks and FFS metric depth."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import matplotlib
import numpy as np
from skimage.morphology import skeletonize

matplotlib.use("Agg")
os.environ.setdefault("THREAD_RECON_BREAK", "0")


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask(path: str, shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.shape != shape:
        raise RuntimeError(f"Mask shape {mask.shape} does not match image {shape}")
    return mask > 0


def load_intrinsics(path: str) -> np.ndarray:
    intrinsic = np.loadtxt(path, delimiter=",").astype(np.float64)
    if intrinsic.shape != (3, 3):
        raise RuntimeError(f"Expected K shape (3,3), got {intrinsic.shape}")
    return intrinsic


def selector_centerline(mask: np.ndarray) -> np.ndarray:
    """Adapt a dense segmentation mask to the original selector's pixel graph."""
    centerline = skeletonize(mask).astype(bool)
    if not np.any(centerline):
        raise RuntimeError("SAM3 thread mask produced an empty centerline")
    return centerline


def rectified_matrices(
    intrinsic: np.ndarray, baseline_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return P1/P2/Q in the original backend's millimetre convention."""
    fx = float(intrinsic[0, 0])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    baseline_mm = baseline_m * 1000.0
    p1 = np.column_stack((intrinsic, np.zeros(3, dtype=np.float64)))
    p2 = p1.copy()
    p2[0, 3] = -fx * baseline_mm
    q = np.asarray(
        [
            [1.0, 0.0, 0.0, -cx],
            [0.0, 1.0, 0.0, -cy],
            [0.0, 0.0, 0.0, fx],
            [0.0, 0.0, 1.0 / baseline_mm, 0.0],
        ],
        dtype=np.float64,
    )
    return p1, p2, q


def foundation_stereo_adapter(
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    q: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
):
    """Create the `_run_stereo` result expected by original keypt_selection."""
    inverse_q = np.linalg.inv(q)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])

    def run(segpix1, _img1, _img2, *_args):
        rows = segpix1[:, 0].astype(np.int64)
        cols = segpix1[:, 1].astype(np.int64)
        z_m = depth_m[rows, cols].astype(np.float64)
        valid = (
            np.isfinite(z_m)
            & (z_m >= min_depth_m)
            & (z_m <= max_depth_m)
        )
        # Invalid pixels are placed far outside the original selector's
        # workspace box so its existing range filter removes them.
        z_mm = np.where(valid, z_m * 1000.0, 1.0e9)
        x_mm = (cols - cx) * z_mm / fx
        y_mm = (rows - cy) * z_mm / fy
        xyz_h = np.vstack((x_mm, y_mm, z_mm, np.ones_like(z_mm)))
        q_inputs = inverse_q @ xyz_h
        q_inputs /= np.where(
            np.abs(q_inputs[3]) > 1.0e-12, q_inputs[3], 1.0
        )
        reliability = valid.astype(np.float64)
        return reliability, q_inputs

    return run


def import_original_backend(original_src: str):
    source = Path(original_src).resolve()
    package = source / "thread_reconstruction"
    if not package.is_dir():
        raise RuntimeError(
            f"Expected original package at {package}; run collect_blackdragon_dependencies.sh"
        )
    sys.path.insert(0, str(source))
    selection = importlib.import_module("thread_reconstruction.keypt_selection")
    ordering = importlib.import_module("thread_reconstruction.keypt_ordering")
    optim = importlib.import_module("thread_reconstruction.optim")
    return selection, ordering, optim


def save_overlay(path: Path, image_rgb: np.ndarray, pixel_xy: np.ndarray) -> None:
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    polyline = np.round(pixel_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [polyline], False, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-src", required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--left-mask", required=True)
    parser.add_argument("--right-mask", required=True)
    parser.add_argument("--depth-npy", required=True)
    parser.add_argument("--K", required=True)
    parser.add_argument("--baseline-m", required=True, type=float)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--segment-connection-radius-px", type=float, default=80.0)
    parser.add_argument("--min-depth-m", type=float, default=0.03)
    parser.add_argument("--max-depth-m", type=float, default=0.50)
    parser.add_argument("--min-valid-depth-fraction", type=float, default=0.70)
    args = parser.parse_args()

    if args.samples < 4:
        raise ValueError("--samples must be at least 4")
    if args.segment_connection_radius_px <= 0.0:
        raise ValueError("--segment-connection-radius-px must be positive")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    left = load_image(args.left)
    right = load_image(args.right)
    if left.shape != right.shape:
        raise RuntimeError(f"Stereo shape mismatch: {left.shape} vs {right.shape}")
    left_mask = load_mask(args.left_mask, left.shape[:2])
    right_mask = load_mask(args.right_mask, right.shape[:2])
    if not np.any(left_mask):
        raise RuntimeError("Left SAM3 thread mask is empty")
    if not np.any(right_mask):
        raise RuntimeError("Right SAM3 thread mask is empty")
    selection_mask = selector_centerline(left_mask)
    depth = np.load(args.depth_npy).astype(np.float64)
    if depth.shape != left.shape[:2]:
        raise RuntimeError(f"Depth shape {depth.shape} does not match {left.shape[:2]}")
    intrinsic = load_intrinsics(args.K)
    mask_depth = depth[selection_mask]
    valid_depth_fraction = float(
        np.mean(
            np.isfinite(mask_depth)
            & (mask_depth >= args.min_depth_m)
            & (mask_depth <= args.max_depth_m)
        )
    )
    if valid_depth_fraction < args.min_valid_depth_fraction:
        raise RuntimeError(
            f"Valid FFS depth fraction {valid_depth_fraction:.3f} is below "
            f"{args.min_valid_depth_fraction:.3f}"
        )
    p1, p2, q = rectified_matrices(intrinsic, args.baseline_m)

    selection_module, ordering_module, optim_module = import_original_backend(
        args.original_src
    )
    common_args = SimpleNamespace(time=False)
    selector = selection_module.Select(common_args)
    orderer = ordering_module.Order(common_args)
    optimizer = optim_module.Optim(common_args)
    orderer.COLD_ASSEMBLY_MAX_JOIN_PX = args.segment_connection_radius_px

    original_stereo = selection_module._run_stereo
    selection_module._run_stereo = foundation_stereo_adapter(
        depth, intrinsic, q, args.min_depth_m, args.max_depth_m
    )
    try:
        selected = selector.keypt_selection(
            left, right, selection_mask, q, speedy=True
        )
    finally:
        selection_module._run_stereo = original_stereo

    if len(selected) not in (8, 9):
        raise RuntimeError(f"Unexpected original selection result length: {len(selected)}")
    img_3d, _, cluster_map, keypoints, grow_paths, adjacents, intersections, _ = selected[:8]
    keypoint_confidence = getattr(selector, "last_keypt_conf", None)
    _, ordered_keypoints, _, order = orderer.keypt_ordering(
        left,
        img_3d,
        cluster_map,
        keypoints,
        grow_paths,
        adjacents,
        intersection_segments=intersections,
        speedy=True,
        keypt_conf=keypoint_confidence,
    )
    if order is None or len(order) < 4:
        raise RuntimeError("Original keypoint ordering returned fewer than four points")

    thread_dict, specs = optimizer.optim(
        left_mask,
        ordered_keypoints,
        order,
        intrinsic,
        p1,
        p2,
        speedy=True,
        keypt_conf=keypoint_confidence,
    )
    if not thread_dict or thread_dict.get("thread") is None:
        raise RuntimeError("Original spline optimizer returned no reconstruction")

    spline = thread_dict["thread"]
    domain = (float(spline.t[spline.k]), float(spline.t[-spline.k - 1]))
    spline_u = np.linspace(domain[0], domain[1], args.samples)
    points_mm = np.asarray(spline(spline_u), dtype=np.float64)
    if points_mm.shape != (args.samples, 3) or not np.all(np.isfinite(points_mm)):
        raise RuntimeError("Original spline produced invalid samples")
    projected = (p1 @ np.column_stack((points_mm, np.ones(args.samples))).T).T
    projected_xy = projected[:, :2] / projected[:, 2:3]

    raw_reliability = specs.get("reliability") if specs else None
    reliability = (
        np.ones(args.samples, dtype=np.float64)
        if raw_reliability is None
        else np.asarray(raw_reliability, dtype=np.float64).reshape(-1)
    )
    if reliability.size != args.samples:
        reliability = np.interp(
            np.linspace(0.0, 1.0, args.samples),
            np.linspace(0.0, 1.0, reliability.size),
            reliability,
        )
    np.savez(
        out_dir / "thread_samples.npz",
        u=np.linspace(0.0, 1.0, args.samples),
        points=points_mm,
        points_m=points_mm / 1000.0,
        reliability=np.clip(reliability, 0.0, 1.0),
        pixel_xy=projected_xy,
        algorithm=np.asarray("sam3_ffs_original_ordering_spline"),
        spline_degree=np.asarray(int(spline.k)),
        control_points=np.asarray(spline.c),
        knots=np.asarray(spline.t),
    )
    save_overlay(out_dir / "left_reconstruction_overlay.png", left, projected_xy)
    report = {
        "status": "passed",
        "algorithm": "sam3_ffs_original_ordering_spline",
        "front_end": {
            "segmentation": "fine_tuned_sam3",
            "stereo": "fast_foundationstereo_metric_depth",
        },
        "original_backend": [
            "keypoint clustering",
            "intersection detection",
            "keypoint ordering",
            "directional segment assembly",
            "QP spline optimization",
        ],
        "segment_connection_radius_px": args.segment_connection_radius_px,
        "sam3_mask_area_px": int(np.count_nonzero(left_mask)),
        "selector_centerline_px": int(np.count_nonzero(selection_mask)),
        "selector_centerline_component_count": int(
            cv2.connectedComponents(selection_mask.astype(np.uint8), 8)[0] - 1
        ),
        "ordered_keypoint_count": int(len(order)),
        "spline_degree": int(spline.k),
        "control_point_count": int(len(spline.c)),
        "sample_count": int(args.samples),
        "depth_range_m": [args.min_depth_m, args.max_depth_m],
        "valid_depth_fraction": valid_depth_fraction,
    }
    (out_dir / "quality_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Thread samples: {out_dir / 'thread_samples.npz'}")


if __name__ == "__main__":
    main()
