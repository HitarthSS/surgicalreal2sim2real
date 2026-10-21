#!/usr/bin/env python3
"""Reconstruct one thread with the pipeline's single canonical algorithm.

The algorithm is deliberately fixed:

1. Assemble nearby SAM/SAM3 mask components and extract their ordered skeleton.
2. Sample Fast-FoundationStereo metric depth along that skeleton.
3. Robustly smooth depth along curve arclength.
4. Backproject with the calibrated left-camera intrinsic matrix.
5. Reject the result unless it projects onto the selected right SAM mask.

There is no alternate matcher, optimizer, or fallback path in this module.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize


def load_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.shape != shape:
        raise RuntimeError(f"Mask {path} shape {mask.shape} does not match image {shape}")
    return mask > 0


def load_intrinsics(path: str | Path) -> np.ndarray:
    intrinsic = np.loadtxt(path, delimiter=",").astype(np.float64)
    if intrinsic.shape != (3, 3):
        raise RuntimeError(f"Expected K shape (3,3), got {intrinsic.shape}")
    if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
        raise RuntimeError("Camera focal lengths must be positive")
    return intrinsic


def connect_skeleton_components(
    graph: nx.Graph, radius_px: float
) -> list[dict[str, object]]:
    """Greedily bridge component endpoints within the original raw-distance cap."""
    joins: list[dict[str, object]] = []
    if radius_px <= 0.0:
        return joins

    while True:
        components = [set(nodes) for nodes in nx.connected_components(graph)]
        if len(components) <= 1:
            break
        best = None
        for left_index, left_nodes in enumerate(components[:-1]):
            left_endpoints = [node for node in left_nodes if graph.degree[node] == 1]
            if not left_endpoints:
                continue
            left_array = np.asarray(left_endpoints, dtype=np.float64)
            for right_nodes in components[left_index + 1 :]:
                right_endpoints = [node for node in right_nodes if graph.degree[node] == 1]
                if not right_endpoints:
                    continue
                right_array = np.asarray(right_endpoints, dtype=np.float64)
                distances = np.linalg.norm(
                    left_array[:, None, :] - right_array[None, :, :], axis=2
                )
                pair = np.unravel_index(int(np.argmin(distances)), distances.shape)
                candidate = (
                    float(distances[pair]),
                    left_endpoints[pair[0]],
                    right_endpoints[pair[1]],
                )
                if best is None or candidate[0] < best[0]:
                    best = candidate

        if best is None or best[0] > radius_px:
            break
        distance, left_node, right_node = best
        graph.add_edge(left_node, right_node, weight=distance, synthetic=True)
        joins.append(
            {
                "from_rc": [int(left_node[0]), int(left_node[1])],
                "to_rc": [int(right_node[0]), int(right_node[1])],
                "distance_px": float(distance),
            }
        )
    return joins


def skeleton_graph(
    mask: np.ndarray,
    segment_connection_radius_px: float,
    segment_min_skeleton_px: int,
) -> tuple[np.ndarray, nx.Graph, list[dict[str, object]], int, int]:
    skeleton = skeletonize(mask)
    label_count, labels = cv2.connectedComponents(
        skeleton.astype(np.uint8), connectivity=8
    )
    component_sizes = np.bincount(labels.ravel(), minlength=label_count)
    retained_labels = [
        label
        for label in range(1, label_count)
        if component_sizes[label] >= segment_min_skeleton_px
    ]
    discarded_component_count = (label_count - 1) - len(retained_labels)
    skeleton = np.isin(labels, retained_labels)
    pixels = [tuple(point) for point in np.argwhere(skeleton)]
    if len(pixels) < 2:
        raise RuntimeError("Thread mask did not produce a usable skeleton")

    graph = nx.Graph()
    graph.add_nodes_from(pixels)
    pixel_set = set(pixels)
    offsets = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
    )
    for row, col in pixels:
        for drow, dcol in offsets:
            neighbor = (row + drow, col + dcol)
            if neighbor in pixel_set and neighbor > (row, col):
                graph.add_edge(
                    (row, col),
                    neighbor,
                    weight=float(np.hypot(drow, dcol)),
                )
    component_count = nx.number_connected_components(graph)
    joins = connect_skeleton_components(graph, segment_connection_radius_px)
    return skeleton, graph, joins, component_count, discarded_component_count


def farthest_node(
    graph: nx.Graph, source: tuple[int, int], candidates
) -> tuple[tuple[int, int], float]:
    lengths = nx.single_source_dijkstra_path_length(graph, source, weight="weight")
    reachable = [(node, lengths[node]) for node in candidates if node in lengths]
    if not reachable:
        raise RuntimeError("Thread skeleton graph is disconnected")
    return max(reachable, key=lambda item: item[1])


def longest_skeleton_path(
    mask: np.ndarray,
    segment_connection_radius_px: float = 80.0,
    segment_min_skeleton_px: int = 30,
) -> tuple[np.ndarray, dict]:
    skeleton, graph, joins, component_count, discarded_component_count = skeleton_graph(
        mask, segment_connection_radius_px, segment_min_skeleton_px
    )
    remaining_component_count = nx.number_connected_components(graph)
    if remaining_component_count > 1:
        graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
    endpoints = [node for node, degree in graph.degree if degree == 1]

    if len(endpoints) >= 2:
        best_pair = None
        best_length = -1.0
        for source in endpoints:
            target, length = farthest_node(graph, source, endpoints)
            if length > best_length:
                best_pair = (source, target)
                best_length = length
    else:
        # A closed or locally cyclic skeleton has no two degree-one endpoints.
        # Two geodesic sweeps provide a deterministic diameter path without
        # changing reconstruction algorithms.
        seed = min(graph.nodes)
        first, _ = farthest_node(graph, seed, graph.nodes)
        second, best_length = farthest_node(graph, first, graph.nodes)
        best_pair = (first, second)

    if best_pair is None or best_length < 10.0:
        raise RuntimeError("Thread skeleton is too short to reconstruct")
    path = nx.shortest_path(graph, *best_pair, weight="weight")
    return np.asarray(path, dtype=np.float64), {
        "mask_area_px": int(np.count_nonzero(mask)),
        "skeleton_px": int(np.count_nonzero(skeleton)),
        "skeleton_endpoint_count": int(len(endpoints)),
        "skeleton_component_count_before_join": int(component_count),
        "skeleton_component_count_after_join": int(remaining_component_count),
        "discarded_short_component_count": int(discarded_component_count),
        "segment_min_skeleton_px": int(segment_min_skeleton_px),
        "segment_connection_radius_px": float(segment_connection_radius_px),
        "segment_join_count": int(len(joins)),
        "segment_joins": joins,
        "ordered_path_px": int(len(path)),
        "ordered_path_length_px": float(best_length),
    }


def resample_path(path_rc: np.ndarray, count: int) -> np.ndarray:
    if count < 2:
        raise ValueError("sample count must be at least two")
    segment = np.linalg.norm(np.diff(path_rc, axis=0), axis=1)
    keep = np.concatenate([[True], segment > 1.0e-9])
    path_rc = path_rc[keep]
    distance = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(path_rc, axis=0), axis=1))]
    )
    if distance[-1] <= 0.0:
        raise RuntimeError("Thread skeleton path has zero length")
    targets = np.linspace(0.0, distance[-1], count)
    return np.column_stack(
        [np.interp(targets, distance, path_rc[:, axis]) for axis in range(2)]
    )


def sample_depth(
    depth: np.ndarray,
    mask: np.ndarray,
    path_rc: np.ndarray,
    radius: int,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    values = np.full(len(path_rc), np.nan, dtype=np.float64)
    support = np.zeros(len(path_rc), dtype=np.float64)

    for index, (row_f, col_f) in enumerate(path_rc):
        row = int(round(row_f))
        col = int(round(col_f))
        row0, row1 = max(0, row - radius), min(height, row + radius + 1)
        col0, col1 = max(0, col - radius), min(width, col + radius + 1)
        rows, cols = np.ogrid[row0:row1, col0:col1]
        disk = (rows - row_f) ** 2 + (cols - col_f) ** 2 <= radius**2
        local_mask = mask[row0:row1, col0:col1] & disk
        local_depth = depth[row0:row1, col0:col1]
        valid = (
            local_mask
            & np.isfinite(local_depth)
            & (local_depth >= min_depth_m)
            & (local_depth <= max_depth_m)
        )
        support[index] = np.count_nonzero(valid) / max(np.count_nonzero(local_mask), 1)
        if np.any(valid):
            values[index] = float(np.median(local_depth[valid]))
    return values, support


def odd_window(requested: int, count: int, minimum: int) -> int:
    window = min(int(requested), count if count % 2 else count - 1)
    if window % 2 == 0:
        window -= 1
    return window if window >= minimum else 0


def clean_depth(
    raw_depth: np.ndarray, median_window: int, smooth_window: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    valid = np.isfinite(raw_depth)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) < 2:
        raise RuntimeError("Fewer than two valid thread depth samples")

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
    outlier_threshold = max(0.003, residual_median + 4.0 * 1.4826 * residual_mad)
    accepted = valid & (residual <= outlier_threshold)
    accepted_indices = np.flatnonzero(accepted)
    if len(accepted_indices) < 2:
        raise RuntimeError("Depth outlier rejection removed all thread samples")

    cleaned = np.interp(sample_index, accepted_indices, raw_depth[accepted])
    if med_window:
        cleaned = median_filter(cleaned, size=med_window, mode="nearest")
    sg_window = odd_window(smooth_window, len(cleaned), 5)
    if sg_window:
        cleaned = savgol_filter(
            cleaned, window_length=sg_window, polyorder=2, mode="interp"
        )

    return cleaned, accepted, {
        "raw_valid_count": int(np.count_nonzero(valid)),
        "depth_outlier_count": int(np.count_nonzero(valid & ~accepted)),
        "depth_outlier_threshold_m": float(outlier_threshold),
    }


def backproject(
    path_rc: np.ndarray, depth_m: np.ndarray, intrinsic: np.ndarray
) -> np.ndarray:
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    rows, cols = path_rc[:, 0], path_rc[:, 1]
    x = (cols - cx) * depth_m / fx
    y = (rows - cy) * depth_m / fy
    return np.column_stack([x, y, depth_m])


def right_mask_distances(
    path_rc: np.ndarray,
    depth_m: np.ndarray,
    right_mask: np.ndarray,
    focal_x: float,
    baseline_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    disparity = focal_x * baseline_m / depth_m
    right_cols = path_rc[:, 1] - disparity
    right_rows = path_rc[:, 0]

    distance_map = cv2.distanceTransform(
        (~right_mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    ).astype(np.float32)
    distances = cv2.remap(
        distance_map,
        right_cols.astype(np.float32).reshape(1, -1),
        right_rows.astype(np.float32).reshape(1, -1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=1.0e6,
    ).reshape(-1)
    return distances.astype(np.float64), np.column_stack([right_cols, right_rows])


def save_overlays(
    out_dir: Path,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    left_rc: np.ndarray,
    right_xy: np.ndarray,
    tolerance_px: float,
    right_distances: np.ndarray,
) -> None:
    left = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR)
    left_polyline = np.round(left_rc[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(left, [left_polyline], False, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "left_reconstruction_overlay.png"), left)

    right = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR)
    valid = right_distances <= tolerance_px
    points = np.round(right_xy).astype(np.int32)
    for point, accepted in zip(points, valid):
        color = (0, 255, 0) if accepted else (0, 0, 255)
        cv2.circle(right, tuple(point), 1, color, -1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "right_validation_overlay.png"), right)


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--left-mask", required=True)
    parser.add_argument("--right-mask", required=True)
    parser.add_argument("--depth-npy", required=True)
    parser.add_argument("--K", required=True)
    parser.add_argument("--baseline-m", type=float, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument(
        "--segment-connection-radius-px",
        type=float,
        default=80.0,
        help="Maximum raw endpoint separation to bridge between mask components; 0 disables bridging.",
    )
    parser.add_argument(
        "--segment-min-skeleton-px",
        type=int,
        default=30,
        help="Discard mask components with shorter skeletons before endpoint assembly.",
    )
    parser.add_argument("--depth-sample-radius", type=int, default=3)
    parser.add_argument("--min-depth-m", type=float, default=0.03)
    parser.add_argument("--max-depth-m", type=float, default=0.50)
    parser.add_argument("--min-valid-depth-fraction", type=float, default=0.70)
    parser.add_argument("--depth-median-window", type=int, default=9)
    parser.add_argument("--depth-smooth-window", type=int, default=21)
    parser.add_argument("--right-mask-tolerance-px", type=float, default=8.0)
    parser.add_argument("--min-right-mask-fraction", type=float, default=0.50)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "quality_report.json"

    left = load_rgb(args.left)
    right = load_rgb(args.right)
    if left.shape != right.shape:
        raise RuntimeError(f"Stereo image shape mismatch: {left.shape} vs {right.shape}")
    left_mask = load_mask(args.left_mask, left.shape[:2])
    right_mask = load_mask(args.right_mask, right.shape[:2])
    depth = np.load(args.depth_npy).astype(np.float64)
    if depth.shape != left.shape[:2]:
        raise RuntimeError(f"Depth shape {depth.shape} does not match image {left.shape[:2]}")
    intrinsic = load_intrinsics(args.K)

    if args.segment_connection_radius_px < 0.0:
        raise ValueError("--segment-connection-radius-px must be nonnegative")
    if args.segment_min_skeleton_px < 2:
        raise ValueError("--segment-min-skeleton-px must be at least 2")
    ordered_path, skeleton_metrics = longest_skeleton_path(
        left_mask,
        args.segment_connection_radius_px,
        args.segment_min_skeleton_px,
    )
    sampled_path = resample_path(ordered_path, args.samples)
    raw_depth, support = sample_depth(
        depth,
        left_mask,
        sampled_path,
        args.depth_sample_radius,
        args.min_depth_m,
        args.max_depth_m,
    )
    cleaned_depth, accepted_depth, depth_metrics = clean_depth(
        raw_depth, args.depth_median_window, args.depth_smooth_window
    )
    points_m = backproject(sampled_path, cleaned_depth, intrinsic)
    right_distance, right_xy = right_mask_distances(
        sampled_path,
        cleaned_depth,
        right_mask,
        float(intrinsic[0, 0]),
        args.baseline_m,
    )

    valid_depth_fraction = float(
        np.count_nonzero(np.isfinite(raw_depth)) / len(raw_depth)
    )
    right_mask_fraction = float(
        np.mean(right_distance <= args.right_mask_tolerance_px)
    )
    failures = []
    if valid_depth_fraction < args.min_valid_depth_fraction:
        failures.append(
            f"valid depth fraction {valid_depth_fraction:.3f} is below "
            f"{args.min_valid_depth_fraction:.3f}"
        )
    if right_mask_fraction < args.min_right_mask_fraction:
        failures.append(
            f"right-mask projection fraction {right_mask_fraction:.3f} is below "
            f"{args.min_right_mask_fraction:.3f}"
        )

    reliability = np.clip(support, 0.0, 1.0)
    reliability *= np.exp(
        -right_distance / max(args.right_mask_tolerance_px, 1.0e-6)
    )
    reliability *= accepted_depth.astype(np.float64)
    report = {
        "status": "failed" if failures else "passed",
        "algorithm": "sam_mask_foundationstereo_skeleton_backprojection",
        "fallbacks": False,
        "output_units": "millimeters",
        "inputs": {
            "left": str(Path(args.left).resolve()),
            "right": str(Path(args.right).resolve()),
            "left_mask": str(Path(args.left_mask).resolve()),
            "right_mask": str(Path(args.right_mask).resolve()),
            "depth_npy": str(Path(args.depth_npy).resolve()),
            "K": str(Path(args.K).resolve()),
            "baseline_m": float(args.baseline_m),
        },
        "metrics": {
            **skeleton_metrics,
            **depth_metrics,
            "sample_count": int(args.samples),
            "valid_depth_fraction": valid_depth_fraction,
            "median_depth_support": float(np.median(support)),
            "depth_m_min_median_max": [
                float(np.min(cleaned_depth)),
                float(np.median(cleaned_depth)),
                float(np.max(cleaned_depth)),
            ],
            "right_mask_fraction": right_mask_fraction,
            "right_mask_distance_px_median": float(np.median(right_distance)),
            "right_mask_distance_px_p95": float(np.percentile(right_distance, 95)),
        },
        "thresholds": {
            "depth_range_m": [float(args.min_depth_m), float(args.max_depth_m)],
            "min_valid_depth_fraction": float(args.min_valid_depth_fraction),
            "segment_connection_radius_px": float(
                args.segment_connection_radius_px
            ),
            "segment_min_skeleton_px": int(args.segment_min_skeleton_px),
            "right_mask_tolerance_px": float(args.right_mask_tolerance_px),
            "min_right_mask_fraction": float(args.min_right_mask_fraction),
        },
        "failures": failures,
    }
    write_report(report_path, report)
    save_overlays(
        out_dir,
        left,
        right,
        sampled_path,
        right_xy,
        args.right_mask_tolerance_px,
        right_distance,
    )
    if failures:
        raise RuntimeError(
            "Thread reconstruction failed calibrated quality checks. "
            f"See {report_path}: {'; '.join(failures)}"
        )

    output = out_dir / "thread_samples.npz"
    np.savez(
        output,
        u=np.linspace(0.0, 1.0, len(points_m)),
        points=points_m * 1000.0,
        points_m=points_m,
        reliability=reliability,
        pixel_xy=sampled_path[:, ::-1],
        depth_m=cleaned_depth,
        algorithm=np.asarray("sam_mask_foundationstereo_skeleton_backprojection"),
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
                    "algorithm": "sam_mask_foundationstereo_skeleton_backprojection",
                    "fallbacks": False,
                    "failures": [str(error)],
                },
            )
        location = f" See {report_path}." if report_path is not None else ""
        raise SystemExit(f"ERROR: thread reconstruction failed: {error}.{location}") from None
