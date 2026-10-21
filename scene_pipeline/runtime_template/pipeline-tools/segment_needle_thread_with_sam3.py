#!/usr/bin/env python3
"""Segment visible suture thread and needle pixels with SAM 3 text prompts."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


COLORS_BGR = {
    "thread": np.array([255, 255, 0], dtype=np.uint8),
    "needle": np.array([255, 0, 255], dtype=np.uint8),
}


def as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
        if value.is_floating_point():
            value = value.float()
        value = value.cpu().numpy()
    return np.asarray(value)


def unpack_prediction(
    output: dict,
    height: int,
    width: int,
    acceptance_threshold: float,
    max_instances: int = 0,
) -> dict:
    candidate_scores = as_numpy(output["scores"]).reshape(-1).astype(np.float32)
    candidate_boxes = as_numpy(output["boxes"]).reshape(-1, 4).astype(np.float32)
    probabilities = as_numpy(output["masks_logits"]).astype(np.float32)
    if probabilities.size == 0:
        probabilities = np.zeros((0, height, width), dtype=np.float32)
    else:
        probabilities = probabilities.reshape(-1, height, width)

    if not (
        len(candidate_scores) == len(candidate_boxes) == len(probabilities)
    ):
        raise RuntimeError(
            "SAM 3 returned inconsistent mask, box, and score counts: "
            f"{len(probabilities)}, {len(candidate_boxes)}, "
            f"{len(candidate_scores)}"
        )

    accepted = candidate_scores >= acceptance_threshold
    scores = candidate_scores[accepted]
    boxes = candidate_boxes[accepted]
    probabilities = probabilities[accepted]
    if max_instances > 0 and len(scores) > max_instances:
        selected = np.argsort(-scores, kind="stable")[:max_instances]
        scores = scores[selected]
        boxes = boxes[selected]
        probabilities = probabilities[selected]

    if len(scores):
        confidence = np.max(
            probabilities * scores[:, np.newaxis, np.newaxis], axis=0
        )
        mask = np.any(probabilities > 0.5, axis=0)
    else:
        confidence = np.zeros((height, width), dtype=np.float32)
        mask = np.zeros((height, width), dtype=bool)

    return {
        "mask": mask,
        "confidence": confidence,
        "scores": scores,
        "boxes": boxes,
        "candidate_scores": candidate_scores,
    }


def resolve_class_overlap(thread: dict, needle: dict) -> tuple[np.ndarray, np.ndarray, int]:
    thread_mask = np.asarray(thread["mask"], dtype=bool).copy()
    needle_mask = np.asarray(needle["mask"], dtype=bool).copy()
    overlap = thread_mask & needle_mask
    overlap_count = int(np.count_nonzero(overlap))
    if overlap_count:
        thread_wins = thread["confidence"] >= needle["confidence"]
        thread_mask[overlap & ~thread_wins] = False
        needle_mask[overlap & thread_wins] = False
    return thread_mask, needle_mask, overlap_count


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
        raise RuntimeError(f"Failed to write mask: {path}")


def save_overlay(
    path: Path, image_rgb: np.ndarray, thread_mask: np.ndarray, needle_mask: np.ndarray
) -> None:
    overlay = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for name, mask in (("thread", thread_mask), ("needle", needle_mask)):
        color = COLORS_BGR[name]
        overlay[mask] = (
            0.45 * overlay[mask].astype(np.float32) + 0.55 * color
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color.tolist(), 1)
    if not cv2.imwrite(str(path), overlay):
        raise RuntimeError(f"Failed to write overlay: {path}")


def prediction_report(prediction: dict) -> dict:
    return {
        "selected_prompt": prediction["prompt"],
        "candidate_scores": [
            round(float(value), 6) for value in prediction["candidate_scores"]
        ],
        "instance_count": int(len(prediction["scores"])),
        "scores": [round(float(value), 6) for value in prediction["scores"]],
        "boxes_xyxy": [
            [round(float(value), 3) for value in box]
            for box in prediction["boxes"]
        ],
        "prompt_trials": prediction["prompt_trials"],
    }


def split_prompts(value: str) -> list[str]:
    prompts = []
    for prompt in value.split("|"):
        prompt = prompt.strip()
        if prompt and prompt not in prompts:
            prompts.append(prompt)
    if not prompts:
        raise ValueError("At least one non-empty text prompt is required")
    return prompts


def normalize_image_checkpoint_state(checkpoint: dict) -> tuple[dict, str]:
    """Extract image-model weights from trainer or official SAM 3 checkpoints."""
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict) or not state:
        raise RuntimeError("SAM 3 checkpoint does not contain a model state dict")
    state = {
        key.removeprefix("module."): value for key, value in state.items()
    }
    detector_state = {
        key.removeprefix("detector."): value
        for key, value in state.items()
        if key.startswith("detector.")
    }
    if detector_state:
        return detector_state, "official detector-prefixed checkpoint"
    return state, "trainer image-model checkpoint"


def load_image_checkpoint_strict(model, checkpoint_path: str) -> str:
    """Load every fine-tuned image-model tensor or fail before inference."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state, checkpoint_format = normalize_image_checkpoint_state(checkpoint)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "SAM 3 checkpoint did not load exactly: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    print(
        f"Strictly loaded {len(state)} tensors from {checkpoint_path} "
        f"({checkpoint_format})"
    )
    return checkpoint_format


def run_prompt_sweep(
    processor,
    state: dict,
    prompts: list[str],
    height: int,
    width: int,
    acceptance_threshold: float,
    max_instances: int,
) -> dict:
    predictions = []
    trials = []
    for prompt in prompts:
        prediction = unpack_prediction(
            processor.set_text_prompt(prompt=prompt, state=state),
            height,
            width,
            acceptance_threshold,
            max_instances,
        )
        prediction["prompt"] = prompt
        top_score = (
            float(np.max(prediction["candidate_scores"]))
            if len(prediction["candidate_scores"])
            else 0.0
        )
        trials.append(
            {
                "prompt": prompt,
                "candidate_count": int(len(prediction["candidate_scores"])),
                "candidate_scores": [
                    round(float(value), 6)
                    for value in prediction["candidate_scores"]
                ],
                "accepted_count": int(len(prediction["scores"])),
                "top_score": round(top_score, 6),
            }
        )
        predictions.append((top_score, prompt, prediction))

    _, _, best = max(predictions, key=lambda item: (item[0], item[1]))
    best["prompt_trials"] = trials
    return best


def segment_image(
    processor,
    path: Path,
    thread_prompts: list[str],
    needle_prompts: list[str],
    acceptance_threshold: float,
    thread_max_instances: int,
    needle_max_instances: int,
):
    image = Image.open(path).convert("RGB")
    image_rgb = np.asarray(image)
    height, width = image_rgb.shape[:2]
    state = processor.set_image(image)

    thread = run_prompt_sweep(
        processor,
        state,
        thread_prompts,
        height,
        width,
        acceptance_threshold,
        thread_max_instances,
    )
    needle = run_prompt_sweep(
        processor,
        state,
        needle_prompts,
        height,
        width,
        acceptance_threshold,
        needle_max_instances,
    )
    thread_mask, needle_mask, overlap_count = resolve_class_overlap(thread, needle)
    return image_rgb, thread_mask, needle_mask, {
        "thread": prediction_report(thread),
        "needle": prediction_report(needle),
        "raw_class_overlap_px": overlap_count,
        "thread_area_px": int(np.count_nonzero(thread_mask)),
        "needle_area_px": int(np.count_nonzero(needle_mask)),
        "thread_component_count": int(
            cv2.connectedComponents(thread_mask.astype(np.uint8), connectivity=8)[0] - 1
        ),
        "needle_component_count": int(
            cv2.connectedComponents(needle_mask.astype(np.uint8), connectivity=8)[0] - 1
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--thread-prompts",
        default="suture thread|surgical thread|thread|suture",
        help="Pipe-separated prompt sweep; the highest-scoring phrase is selected",
    )
    parser.add_argument(
        "--needle-prompts",
        default="curved surgical suture needle|surgical needle|suture needle|needle",
        help="Pipe-separated prompt sweep; the highest-scoring phrase is selected",
    )
    parser.add_argument("--candidate-threshold", type=float, default=0.01)
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    parser.add_argument(
        "--thread-max-instances",
        type=int,
        default=0,
        help="Maximum accepted thread masks per view; 0 preserves all components",
    )
    parser.add_argument(
        "--needle-max-instances",
        type=int,
        default=0,
        help="Maximum accepted visible needle fragments per view; 0 preserves occluded pieces",
    )
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.candidate_threshold <= args.confidence_threshold <= 1.0:
        raise ValueError(
            "Thresholds must satisfy 0 <= candidate <= confidence <= 1"
        )
    if args.thread_max_instances < 0 or args.needle_max_instances < 0:
        raise ValueError("Maximum instance counts cannot be negative")
    thread_prompts = split_prompts(args.thread_prompts)
    needle_prompts = split_prompts(args.needle_prompts)

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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "algorithm": (
            "sam3_finetuned_text_segmentation"
            if checkpoint
            else "sam3_zero_shot_text_segmentation"
        ),
        "thread_prompts": thread_prompts,
        "needle_prompts": needle_prompts,
        "candidate_threshold": args.candidate_threshold,
        "confidence_threshold": args.confidence_threshold,
        "thread_max_instances": args.thread_max_instances,
        "needle_max_instances": args.needle_max_instances,
        "resolution": args.resolution,
        "precision": precision,
        "checkpoint": checkpoint or "facebook/sam3 from Hugging Face",
        "checkpoint_format": checkpoint_format,
        "views": {},
    }

    with torch.inference_mode(), precision_context:
        for view, image_path in (
            ("left", Path(args.left)),
            ("right", Path(args.right)),
        ):
            image, thread_mask, needle_mask, view_report = segment_image(
                processor,
                image_path,
                thread_prompts,
                needle_prompts,
                args.confidence_threshold,
                args.thread_max_instances,
                args.needle_max_instances,
            )
            save_mask(out_dir / f"{view}_thread_mask.png", thread_mask)
            save_mask(out_dir / f"{view}_needle_mask.png", needle_mask)
            save_overlay(
                out_dir / f"{view}_overlay.png", image, thread_mask, needle_mask
            )
            report["views"][view] = view_report

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"SAM 3 outputs: {out_dir}")


if __name__ == "__main__":
    main()
