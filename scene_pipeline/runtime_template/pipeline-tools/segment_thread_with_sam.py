#!/usr/bin/env python3
"""Automatically select stereo thread masks from SAM mask proposals."""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


@dataclass
class Candidate:
    proposal_index: int
    mask: np.ndarray
    area: int
    excluded_area: int
    bbox: tuple
    centroid: tuple
    skeleton_px: int
    width_px: float
    curve_ratio: float
    predicted_iou: float
    stability_score: float
    score: float

    def report(self):
        return {
            "proposal_index": self.proposal_index,
            "area_px": self.area,
            "needle_excluded_area_px": self.excluded_area,
            "bbox_xywh": list(self.bbox),
            "centroid_xy": [round(value, 3) for value in self.centroid],
            "skeleton_px": self.skeleton_px,
            "mean_width_px": round(self.width_px, 3),
            "curve_ratio": round(self.curve_ratio, 3),
            "predicted_iou": round(self.predicted_iou, 4),
            "stability_score": round(self.stability_score, 4),
            "shape_score": round(self.score, 4),
        }


def read_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_mask(path, shape):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.shape != shape[:2]:
        raise RuntimeError(
            f"Exclusion mask {path} shape {mask.shape} does not match image {shape[:2]}"
        )
    return mask > 0


def largest_component(mask):
    labels, count = ndimage.label(mask)
    if count <= 1:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def analyze_proposals(
    proposals,
    image_shape,
    exclusion_mask,
    min_area,
    max_area_fraction,
    min_skeleton_px,
    max_width_px,
):
    height, width = image_shape[:2]
    image_area = height * width
    candidates = []
    for index, proposal in enumerate(proposals):
        proposal_mask = np.asarray(proposal["segmentation"], dtype=bool)
        excluded_area = int(np.count_nonzero(proposal_mask & exclusion_mask))
        mask = largest_component(proposal_mask & ~exclusion_mask)
        rows, cols = np.nonzero(mask)
        area = int(len(rows))
        if area < min_area or area > max_area_fraction * image_area:
            continue

        x0, x1 = int(cols.min()), int(cols.max())
        y0, y1 = int(rows.min()), int(rows.max())
        bbox_width = x1 - x0 + 1
        bbox_height = y1 - y0 + 1
        diagonal = math.hypot(bbox_width, bbox_height)
        if diagonal < 40.0:
            continue

        skeleton_px = int(np.count_nonzero(skeletonize(mask)))
        if skeleton_px < min_skeleton_px:
            continue
        width_px = area / max(skeleton_px, 1)
        curve_ratio = skeleton_px / max(diagonal, 1.0)
        if width_px > max_width_px or curve_ratio < 0.65:
            continue

        predicted_iou = float(proposal.get("predicted_iou", 0.0))
        stability = float(proposal.get("stability_score", 0.0))
        border_pixels = int(
            np.count_nonzero(mask[:2])
            + np.count_nonzero(mask[-2:])
            + np.count_nonzero(mask[:, :2])
            + np.count_nonzero(mask[:, -2:])
        )
        border_fraction = border_pixels / max(area, 1)

        # A thread is long and curved but only a few pixels thick. The quality
        # terms break ties between similarly shaped SAM proposals.
        score = (
            2.5 * math.log1p(skeleton_px)
            + 2.0 * math.log1p(curve_ratio)
            - 1.5 * math.log1p(width_px)
            + 2.0 * predicted_iou
            + 2.0 * stability
            - 3.0 * border_fraction
        )
        candidates.append(
            Candidate(
                proposal_index=index,
                mask=mask,
                area=area,
                excluded_area=excluded_area,
                bbox=(x0, y0, bbox_width, bbox_height),
                centroid=(float(cols.mean()), float(rows.mean())),
                skeleton_px=skeleton_px,
                width_px=float(width_px),
                curve_ratio=float(curve_ratio),
                predicted_iou=predicted_iou,
                stability_score=stability,
                score=float(score),
            )
        )
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def translate_mask(mask, dx, dy):
    transform = np.float32([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]])
    return cv2.warpAffine(
        mask.astype(np.uint8),
        transform,
        (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)


def aligned_iou(left, right):
    dx = int(round(left.centroid[0] - right.centroid[0]))
    dy = int(round(left.centroid[1] - right.centroid[1]))
    shifted = translate_mask(right.mask, dx, dy)
    union = np.count_nonzero(left.mask | shifted)
    overlap = np.count_nonzero(left.mask & shifted)
    return overlap / max(union, 1), dx, dy


def select_stereo_pair(left_candidates, right_candidates, max_disparity):
    if not left_candidates or not right_candidates:
        raise RuntimeError(
            "SAM did not produce a thin, elongated candidate in both images. "
            "Inspect the saved SAM proposal metrics and configured geometric thresholds."
        )

    ranked_pairs = []
    for left in left_candidates[:30]:
        for right in right_candidates[:30]:
            overlap, disparity, vertical_shift = aligned_iou(left, right)
            area_penalty = abs(math.log(left.area / right.area))
            length_penalty = abs(math.log(left.skeleton_px / right.skeleton_px))
            disparity_penalty = 0.0
            if disparity < -8 or disparity > max_disparity:
                disparity_penalty = 5.0
            pair_score = (
                left.score
                + right.score
                + 8.0 * overlap
                - 2.5 * area_penalty
                - 2.5 * length_penalty
                - 0.12 * abs(vertical_shift)
                - disparity_penalty
            )
            ranked_pairs.append(
                (pair_score, overlap, disparity, vertical_shift, left, right)
            )
    ranked_pairs.sort(key=lambda item: item[0], reverse=True)
    return ranked_pairs[0], ranked_pairs


def save_mask(path, mask):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
        raise RuntimeError(f"Failed to write mask: {path}")


def save_overlay(path, image, candidate):
    overlay = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    color = np.array([255, 255, 0], dtype=np.uint8)
    selected = candidate.mask
    overlay[selected] = (
        0.45 * overlay[selected].astype(np.float32) + 0.55 * color
    ).astype(np.uint8)
    contours, _ = cv2.findContours(
        selected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), overlay):
        raise RuntimeError(f"Failed to write overlay: {path}")


def build_generator(args):
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    if args.model_type not in sam_model_registry:
        raise RuntimeError(f"Unsupported SAM model type: {args.model_type}")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_area,
        output_mode="binary_mask",
    )


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-type", default="vit_h")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--points-per-side", type=int, default=64)
    parser.add_argument("--points-per-batch", type=int, default=128)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.75)
    parser.add_argument("--stability-score-thresh", type=float, default=0.85)
    parser.add_argument("--min-area", type=int, default=40)
    parser.add_argument("--max-area-fraction", type=float, default=0.12)
    parser.add_argument("--min-skeleton-px", type=int, default=60)
    parser.add_argument("--max-width-px", type=float, default=20.0)
    parser.add_argument("--max-disparity", type=int, default=416)
    parser.add_argument("--left-exclude-mask", required=True)
    parser.add_argument("--right-exclude-mask", required=True)
    parser.add_argument("--left-mask-out", required=True)
    parser.add_argument("--right-mask-out", required=True)
    parser.add_argument("--left-overlay-out", required=True)
    parser.add_argument("--right-overlay-out", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    left_image = read_rgb(args.left)
    right_image = read_rgb(args.right)
    if left_image.shape != right_image.shape:
        raise RuntimeError(
            f"Stereo image shape mismatch: {left_image.shape} vs {right_image.shape}"
        )
    left_exclusion = read_mask(args.left_exclude_mask, left_image.shape)
    right_exclusion = read_mask(args.right_exclude_mask, right_image.shape)

    generator = build_generator(args)
    print("Generating SAM proposals for the left image...")
    with torch.inference_mode():
        left_proposals = generator.generate(left_image)
        print("Generating SAM proposals for the right image...")
        right_proposals = generator.generate(right_image)

    candidate_args = (
        args.min_area,
        args.max_area_fraction,
        args.min_skeleton_px,
        args.max_width_px,
    )
    left_candidates = analyze_proposals(
        left_proposals, left_image.shape, left_exclusion, *candidate_args
    )
    right_candidates = analyze_proposals(
        right_proposals, right_image.shape, right_exclusion, *candidate_args
    )
    best, ranked_pairs = select_stereo_pair(
        left_candidates, right_candidates, args.max_disparity
    )
    pair_score, overlap, disparity, vertical_shift, left, right = best

    save_mask(args.left_mask_out, left.mask)
    save_mask(args.right_mask_out, right.mask)
    save_overlay(args.left_overlay_out, left_image, left)
    save_overlay(args.right_overlay_out, right_image, right)

    report = {
        "checkpoint": str(args.checkpoint),
        "model_type": args.model_type,
        "left_needle_exclusion_mask": str(args.left_exclude_mask),
        "right_needle_exclusion_mask": str(args.right_exclude_mask),
        "left_proposal_count": len(left_proposals),
        "right_proposal_count": len(right_proposals),
        "left_candidate_count": len(left_candidates),
        "right_candidate_count": len(right_candidates),
        "selected_pair": {
            "score": round(float(pair_score), 4),
            "aligned_iou": round(float(overlap), 4),
            "centroid_disparity_px": disparity,
            "vertical_shift_px": vertical_shift,
            "left": left.report(),
            "right": right.report(),
        },
        "top_left_candidates": [item.report() for item in left_candidates[:20]],
        "top_right_candidates": [item.report() for item in right_candidates[:20]],
        "top_pair_scores": [
            {
                "score": round(float(item[0]), 4),
                "aligned_iou": round(float(item[1]), 4),
                "disparity_px": item[2],
                "vertical_shift_px": item[3],
                "left_proposal_index": item[4].proposal_index,
                "right_proposal_index": item[5].proposal_index,
            }
            for item in ranked_pairs[:20]
        ],
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["selected_pair"], indent=2))
    print(f"Left mask: {args.left_mask_out}")
    print(f"Right mask: {args.right_mask_out}")


if __name__ == "__main__":
    main()
