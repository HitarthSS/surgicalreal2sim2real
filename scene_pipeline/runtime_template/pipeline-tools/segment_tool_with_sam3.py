#!/usr/bin/env python3
"""Segment one visible surgical instrument with SAM 3 text prompts."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from segment_needle_thread_with_sam3 import (
    load_image_checkpoint_strict,
    prediction_report,
    run_prompt_sweep,
    save_mask,
    split_prompts,
)


TOOL_COLOR_BGR = np.array([0, 165, 255], dtype=np.uint8)


def validate_tool_mask(
    mask: np.ndarray,
    min_area_px: int,
    max_area_fraction: float,
) -> dict:
    mask = np.asarray(mask, dtype=bool)
    area_px = int(np.count_nonzero(mask))
    image_area_px = int(mask.size)
    area_fraction = area_px / image_area_px if image_area_px else 0.0
    failures = []
    if area_px < min_area_px:
        failures.append(
            f"tool mask has {area_px} pixels; minimum is {min_area_px}"
        )
    if area_fraction > max_area_fraction:
        failures.append(
            "tool mask covers "
            f"{area_fraction:.4f} of the image; maximum is {max_area_fraction:.4f}"
        )
    return {
        "passed": not failures,
        "failures": failures,
        "area_px": area_px,
        "image_area_px": image_area_px,
        "area_fraction": area_fraction,
        "component_count": int(
            cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1
        ),
    }


def save_tool_overlay(path: Path, image_rgb: np.ndarray, mask: np.ndarray) -> None:
    overlay = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    mask = np.asarray(mask, dtype=bool)
    overlay[mask] = (
        0.45 * overlay[mask].astype(np.float32) + 0.55 * TOOL_COLOR_BGR
    ).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, TOOL_COLOR_BGR.tolist(), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), overlay):
        raise RuntimeError(f"Failed to write overlay: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--prompts",
        default=(
            "robotic surgical instrument|surgical grasper|"
            "laparoscopic surgical instrument"
        ),
        help="Pipe-separated prompt sweep; the highest-scoring phrase is selected",
    )
    parser.add_argument("--candidate-threshold", type=float, default=0.01)
    parser.add_argument("--confidence-threshold", type=float, default=0.20)
    parser.add_argument(
        "--max-instances",
        type=int,
        default=1,
        help="Maximum accepted masks; use 1 for one tracked PSM instrument",
    )
    parser.add_argument("--min-area-px", type=int, default=300)
    parser.add_argument("--max-area-fraction", type=float, default=0.60)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.candidate_threshold <= args.confidence_threshold <= 1.0:
        raise ValueError(
            "Thresholds must satisfy 0 <= candidate <= confidence <= 1"
        )
    if args.max_instances < 1:
        raise ValueError("max-instances must be at least 1")
    if args.min_area_px < 1:
        raise ValueError("min-area-px must be at least 1")
    if not 0.0 < args.max_area_fraction <= 1.0:
        raise ValueError("max-area-fraction must be in (0, 1]")

    prompts = split_prompts(args.prompts)

    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("SAM 3 was configured for CUDA, but CUDA is unavailable")

    checkpoint = str(Path(args.checkpoint)) if args.checkpoint else None
    if checkpoint:
        model = build_sam3_image_model(
            device="cpu",
            checkpoint_path=None,
            load_from_HF=False,
            enable_segmentation=True,
        )
        checkpoint_format = load_image_checkpoint_strict(model, checkpoint)
        model = model.to(args.device).eval()
    else:
        model = build_sam3_image_model(
            device=args.device,
            checkpoint_path=None,
            load_from_HF=True,
            enable_segmentation=True,
        )
        checkpoint_format = "facebook/sam3 Hugging Face checkpoint"

    processor = Sam3Processor(
        model,
        resolution=args.resolution,
        device=args.device,
        confidence_threshold=args.candidate_threshold,
    )

    if args.device.startswith("cuda"):
        device_index = torch.device(args.device).index or 0
        if torch.cuda.get_device_properties(device_index).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        precision_context = torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        )
        precision = "bfloat16 autocast"
    else:
        precision_context = nullcontext()
        precision = "float32"

    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")
    image_rgb = np.asarray(image)
    height, width = image_rgb.shape[:2]

    with torch.inference_mode(), precision_context:
        state = processor.set_image(image)
        prediction = run_prompt_sweep(
            processor,
            state,
            prompts,
            height,
            width,
            args.confidence_threshold,
            args.max_instances,
        )

    mask = np.asarray(prediction["mask"], dtype=bool)
    validation = validate_tool_mask(
        mask,
        min_area_px=args.min_area_px,
        max_area_fraction=args.max_area_fraction,
    )
    report = {
        "algorithm": (
            "sam3_finetuned_tool_text_segmentation"
            if checkpoint
            else "sam3_zero_shot_tool_text_segmentation"
        ),
        "image": str(image_path),
        "prompts": prompts,
        "candidate_threshold": args.candidate_threshold,
        "confidence_threshold": args.confidence_threshold,
        "max_instances": args.max_instances,
        "min_area_px": args.min_area_px,
        "max_area_fraction": args.max_area_fraction,
        "resolution": args.resolution,
        "precision": precision,
        "checkpoint": checkpoint or "facebook/sam3 from Hugging Face",
        "checkpoint_format": checkpoint_format,
        "prediction": prediction_report(prediction),
        "validation": validation,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_mask(out_dir / "tool_mask.png", mask)
    save_tool_overlay(out_dir / "tool_overlay.png", image_rgb, mask)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))
    print(f"SAM 3 tool mask: {out_dir / 'tool_mask.png'}")
    print(f"SAM 3 tool overlay: {out_dir / 'tool_overlay.png'}")
    if not validation["passed"]:
        raise RuntimeError("; ".join(validation["failures"]))


if __name__ == "__main__":
    main()
