#!/usr/bin/env python3
"""Overlay a fitted 3D needle centerline and its RANSAC labels on an image."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def load_centerline(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    data = np.load(path)
    if "needle_centerline" in data.files:
        points = np.asarray(data["needle_centerline"], dtype=np.float64)
    elif "needle_states" in data.files:
        states = np.asarray(data["needle_states"], dtype=np.float64)
        points = states[-1]
    elif "initial_world_centerline" in data.files:
        points = np.asarray(data["initial_world_centerline"], dtype=np.float64)
    else:
        raise RuntimeError(
            f"{path} does not contain needle_centerline, needle_states, or initial_world_centerline"
        )
    points = points.reshape(-1, 3)
    if len(points) < 2 or not np.isfinite(points).all():
        raise RuntimeError(f"Invalid needle centerline shape or values: {points.shape}")

    inlier_mask = None
    if "source_inlier_mask" in data.files:
        inlier_mask = np.asarray(data["source_inlier_mask"], dtype=bool).reshape(-1)
    return points, inlier_mask


def load_points2d(path: Path) -> np.ndarray:
    rows = []
    with path.open(newline="") as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            try:
                rows.append([float(value) for value in row])
            except ValueError:
                continue
    points = np.asarray(rows, dtype=np.float64)
    if points.ndim != 2:
        raise RuntimeError(f"Expected a 2D point CSV, got {points.shape}")
    if points.shape[0] == 2 and points.shape[1] > 2:
        points = points.T
    elif points.shape[1] >= 2:
        points = points[:, :2]
    else:
        raise RuntimeError(f"Expected point CSV shape (2,N) or (N,2), got {points.shape}")
    if len(points) == 0 or not np.isfinite(points).all():
        raise RuntimeError(f"Point CSV is empty or non-finite: {path}")
    return points


def project_points(points: np.ndarray, fx: float, fy: float, cx: float, cy: float):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1.0e-8)
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    uv[valid, 0] = fx * points[valid, 0] / points[valid, 2] + cx
    uv[valid, 1] = fy * points[valid, 1] / points[valid, 2] + cy
    return uv, valid


def draw_polyline(image: np.ndarray, uv: np.ndarray, valid: np.ndarray) -> None:
    for index in range(len(uv) - 1):
        if not (valid[index] and valid[index + 1]):
            continue
        p0 = tuple(np.round(uv[index]).astype(int))
        p1 = tuple(np.round(uv[index + 1]).astype(int))
        cv2.line(image, p0, p1, (20, 20, 20), 7, cv2.LINE_AA)
        cv2.line(image, p0, p1, (255, 235, 35), 3, cv2.LINE_AA)


def nearest_curve_distances(points: np.ndarray, curve: np.ndarray) -> np.ndarray:
    if len(points) == 0 or len(curve) == 0:
        return np.empty(0, dtype=np.float64)
    delta = points[:, None, :] - curve[None, :, :]
    return np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1))


def draw_legend(image: np.ndarray, inliers: int | None, outliers: int | None) -> None:
    entries = [("RANSAC fitted 3D arc", (255, 235, 35))]
    if inliers is not None:
        entries.append((f"inlier labels ({inliers})", (60, 220, 60)))
        entries.append((f"rejected labels ({outliers})", (40, 40, 255)))
    width = 245
    height = 18 + 25 * len(entries)
    overlay = image.copy()
    cv2.rectangle(overlay, (8, 8), (8 + width, 8 + height), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0.0, image)
    for index, (label, color) in enumerate(entries):
        y = 28 + index * 25
        cv2.line(image, (18, y), (44, y), color, 3, cv2.LINE_AA)
        cv2.putText(image, label, (54, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--needle-npz", required=True)
    parser.add_argument("--needle-points", default=None, help="original 24-point CSV")
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--no-legend", action="store_true")
    parser.add_argument("--fx", type=float, default=1025.8822021484375)
    parser.add_argument("--fy", type=float, default=1025.8822021484375)
    parser.add_argument("--cx", type=float, default=167.919017)
    parser.add_argument("--cy", type=float, default=234.152707)
    args = parser.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)
    centerline, inlier_mask = load_centerline(Path(args.needle_npz))
    uv, valid = project_points(centerline, args.fx, args.fy, args.cx, args.cy)
    draw_polyline(image, uv, valid)

    labels = None
    if args.needle_points:
        labels = load_points2d(Path(args.needle_points))
        if inlier_mask is None:
            inlier_mask = np.ones(len(labels), dtype=bool)
        if len(inlier_mask) != len(labels):
            raise RuntimeError(
                f"RANSAC mask has {len(inlier_mask)} entries but point CSV has {len(labels)} points"
            )
        for point, is_inlier in zip(labels, inlier_mask):
            center = tuple(np.round(point).astype(int))
            if is_inlier:
                cv2.circle(image, center, 4, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(image, center, 3, (60, 220, 60), -1, cv2.LINE_AA)
            else:
                x, y = center
                cv2.line(image, (x - 5, y - 5), (x + 5, y + 5), (40, 40, 255), 2, cv2.LINE_AA)
                cv2.line(image, (x - 5, y + 5), (x + 5, y - 5), (40, 40, 255), 2, cv2.LINE_AA)

    inlier_count = int(np.count_nonzero(inlier_mask)) if inlier_mask is not None else None
    outlier_count = int(len(inlier_mask) - inlier_count) if inlier_mask is not None else None
    if not args.no_legend:
        draw_legend(image, inlier_count, outlier_count)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), image):
        raise RuntimeError(f"Could not write {out}")

    valid_curve = uv[valid & np.isfinite(uv).all(axis=1)]
    report = {
        "image": str(Path(args.image).resolve()),
        "needle_npz": str(Path(args.needle_npz).resolve()),
        "output": str(out.resolve()),
        "projected_vertices": int(len(valid_curve)),
        "inlier_count": inlier_count,
        "outlier_count": outlier_count,
    }
    if labels is not None:
        distances = nearest_curve_distances(labels, valid_curve)
        report["label_to_arc_mean_px"] = float(np.mean(distances))
        report["label_to_arc_max_px"] = float(np.max(distances))
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
