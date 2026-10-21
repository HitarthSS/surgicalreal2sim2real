#!/usr/bin/env python3
"""Classify visible needle-thread attachment with Gemma 4 vision-language models.

The tool presents the raw rectified stereo pair to the model and can add a
magnified needle crop plus a supplemental annotation view. Its output is a
strict JSON record that distinguishes an attached suture from incidental contact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import MethodType

import numpy as np
from PIL import Image, ImageDraw


CONTACT_PROMPT = """You are inspecting rectified stereo endoscopic images from a surgical scene.

Objects:
- needle: the dark curved metallic surgical needle
- thread: the thin flexible suture extending from the needle

Decide whether the thread is physically attached to the needle from visible image evidence.

State definitions:
- attached: the thread visibly terminates at or emerges from an endpoint of the needle, with a persistent connection supported by both stereo views
- touching: the needle and thread boundaries meet, but the images do not establish an endpoint attachment
- separate: a visible gap separates the needle and thread
- uncertain: blur, occlusion, resolution, or conflicting views prevent a defensible decision

Mere closeness, crossing, or overlap in only one camera view is not sufficient for attached. Use both stereo views when available. The annotation image, when present, only identifies the pipeline's needle points (cyan) and thread mask (yellow); verify attachment against the raw images. Do not infer attachment solely from the annotation.

Return exactly one JSON object without Markdown or extra prose. Complete the
prefilled object and end with }:
{
  "needle_thread_relation": {
    "state": "attached|touching|separate|uncertain",
    "confidence": 0.0,
    "evidence": "brief visible evidence"
  },
  "limitations": "brief limitation or none"
}
Confidence must be between 0 and 1. Do not provide hidden reasoning; keep each evidence field to one sentence."""

JSON_PREFILL = "{"


def load_points2d(path: Path) -> np.ndarray:
    points = np.loadtxt(path, delimiter=",").astype(np.float64)
    if points.ndim != 2:
        raise RuntimeError(f"Expected a 2D point CSV, got {points.shape}")
    if points.shape[0] == 2:
        points = points.T
    elif points.shape[1] != 2:
        raise RuntimeError(f"Expected point CSV shape (2,N) or (N,2), got {points.shape}")
    if len(points) == 0 or not np.isfinite(points).all():
        raise RuntimeError(f"Point CSV is empty or non-finite: {path}")
    return points


def clamp_crop_box(points: np.ndarray, image_size: tuple[int, int], padding: int) -> tuple[int, int, int, int]:
    width, height = image_size
    x1 = max(0, int(np.floor(points[:, 0].min())) - padding)
    y1 = max(0, int(np.floor(points[:, 1].min())) - padding)
    x2 = min(width, int(np.ceil(points[:, 0].max())) + padding + 1)
    y2 = min(height, int(np.ceil(points[:, 1].max())) + padding + 1)
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError("Needle points produced an empty crop")
    return x1, y1, x2, y2


def fit_with_padding(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Fit an image onto a fixed canvas without changing its aspect ratio."""
    target_width, target_height = target_size
    width, height = image.size
    scale = min(target_width / width, target_height / height)
    resized = image.resize(
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", target_size, (16, 16, 16))
    offset = ((target_width - resized.width) // 2, (target_height - resized.height) // 2)
    canvas.paste(resized, offset)
    return canvas


def make_annotation(
    image: Image.Image,
    points: np.ndarray | None,
    thread_mask_path: Path | None,
) -> Image.Image | None:
    if points is None and thread_mask_path is None:
        return None

    base = image.convert("RGBA")
    if thread_mask_path is not None:
        mask_image = Image.open(thread_mask_path).convert("L")
        if mask_image.size != image.size:
            mask_image = mask_image.resize(image.size, Image.Resampling.NEAREST)
        mask = np.asarray(mask_image)
        threshold = 0 if int(mask.max()) <= 1 else 127
        alpha = Image.fromarray(np.where(mask > threshold, 92, 0).astype(np.uint8))
        yellow = Image.new("RGBA", image.size, (255, 215, 0, 0))
        yellow.putalpha(alpha)
        base = Image.alpha_composite(base, yellow)

    if points is not None:
        draw = ImageDraw.Draw(base)
        xy = [(float(x), float(y)) for x, y in points]
        if len(xy) > 1:
            draw.line(xy, fill=(0, 255, 255, 230), width=2)
        for x, y in xy:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=(0, 255, 255, 255), width=2)
    return base.convert("RGB")


def make_contact_sheet(panels: list[tuple[str, Path]], panel_size: tuple[int, int], out_path: Path) -> None:
    columns = 2
    rows = (len(panels) + columns - 1) // columns
    width, height = panel_size
    sheet = Image.new("RGB", (columns * width, rows * height), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(panels):
        with Image.open(path) as source:
            panel = fit_with_padding(source.convert("RGB"), panel_size)
        x = (index % columns) * width
        y = (index // columns) * height
        sheet.paste(panel, (x, y))
        draw.rectangle((x, y, x + width, y + 24), fill=(0, 0, 0))
        draw.text((x + 8, y + 6), label, fill=(255, 255, 255))
    sheet.save(out_path)


def prepare_views(
    left_path: Path,
    right_path: Path | None,
    needle_points_path: Path | None,
    thread_mask_path: Path | None,
    out_dir: Path,
    crop_padding: int,
) -> list[tuple[str, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    left = Image.open(left_path).convert("RGB")
    left_full_path = out_dir / "image_1_left_raw.png"
    left.save(left_full_path)
    panels = [("RAW LEFT", left_full_path)]

    if right_path is not None:
        right = Image.open(right_path).convert("RGB")
        if right.size != left.size:
            raise RuntimeError(f"Stereo image size mismatch: left {left.size}, right {right.size}")
        right_full_path = out_dir / "image_2_right_raw.png"
        right.save(right_full_path)
        panels.append(("RAW RIGHT", right_full_path))

    points = load_points2d(needle_points_path) if needle_points_path is not None else None
    if points is not None:
        box = clamp_crop_box(points, left.size, crop_padding)
        crop = fit_with_padding(left.crop(box), left.size)
        crop_path = out_dir / "image_3_needle_thread_detail.png"
        crop.save(crop_path)
        panels.append(("NEEDLE/THREAD DETAIL", crop_path))

    annotation = make_annotation(left, points, thread_mask_path)
    if annotation is not None:
        annotation_path = out_dir / "image_4_pipeline_annotation.png"
        annotation.save(annotation_path)
        panels.append(("PIPELINE ANNOTATION", annotation_path))

    contact_sheet_path = out_dir / "input_contact_sheet.png"
    make_contact_sheet(panels, left.size, contact_sheet_path)

    descriptions = {
        "RAW LEFT": "raw left rectified stereo frame",
        "RAW RIGHT": "raw right rectified stereo frame",
        "NEEDLE/THREAD DETAIL": "magnified raw left-frame needle and thread detail",
        "PIPELINE ANNOTATION": "supplemental left-frame annotation; cyan is the selected needle and yellow is the thread mask",
    }
    return [
        (f"IMAGE {index}: {descriptions[label]}.", path)
        for index, (label, path) in enumerate(panels, start=1)
    ]


def build_messages(views: list[tuple[str, Path]]) -> list[dict]:
    content = []
    for label, path in views:
        content.append({"type": "text", "text": label})
        content.append({"type": "image", "url": str(path.resolve())})
    content.append({"type": "text", "text": CONTACT_PROMPT})
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": JSON_PREFILL}]},
    ]


def parsed_response_text(parsed, raw: str) -> str:
    """Extract Gemma's final channel from parse_response across API versions."""
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, dict):
        for key in ("content", "final", "text"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, list):
                text = "\n".join(
                    str(item.get("text", "")) if isinstance(item, dict) else str(item)
                    for item in value
                ).strip()
                if text:
                    return text
    return raw


def pipeline_response_text(output) -> str:
    value = output[0]["generated_text"]
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        value = value[-1]
    if isinstance(value, dict):
        return parsed_response_text(value, str(value))
    return str(value)


def extract_json_object(text: str) -> dict:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "needle_thread_relation" in value:
            return value
    raise RuntimeError("Gemma response did not contain the required JSON object")


def restore_json_prefill(continuation: str) -> str:
    """Restore a chat-template assistant prefill when the pipeline returns only new tokens."""
    candidates = (continuation, JSON_PREFILL + continuation.lstrip())
    for candidate in candidates:
        try:
            extract_json_object(candidate)
        except RuntimeError:
            continue
        return candidate
    return continuation


def validate_assessment(value: dict) -> dict:
    relation = value.get("needle_thread_relation", {})
    state = str(relation.get("state", "")).lower()
    if state not in {"attached", "touching", "separate", "uncertain"}:
        raise RuntimeError(f"Invalid needle_thread_relation.state: {state!r}")
    confidence = float(relation.get("confidence"))
    if not 0.0 <= confidence <= 1.0:
        raise RuntimeError(f"Invalid needle_thread_relation.confidence: {confidence}")
    relation["state"] = state
    relation["confidence"] = confidence
    relation["evidence"] = str(relation.get("evidence", ""))
    value["limitations"] = str(value.get("limitations", ""))
    return value


def normalize_gemma4_image_feature_output(model) -> None:
    """Normalize Gemma 4 image features to one 2D LM-width tensor per image group.

    Transformers releases have returned Gemma 4 vision features as tensors with
    different ranks, and some combinations expose the raw vision width instead
    of the projected language-model width. The model's own learned projection is
    the only valid conversion between those spaces.
    """
    import torch

    gemma_model = model.model
    original_get_image_features = gemma_model.get_image_features
    text_width = int(gemma_model.get_input_embeddings().weight.shape[-1])
    projection = gemma_model.embed_vision.embedding_projection
    vision_width = int(projection.in_features)
    projected_width = int(projection.out_features)
    if projected_width != text_width:
        raise RuntimeError(
            f"Gemma checkpoint is internally inconsistent: vision projection is "
            f"{vision_width}->{projected_width}, text embedding width is {text_width}"
        )
    print(f"gemma_image_widths=vision:{vision_width} projected:{projected_width} text:{text_width}")

    validation_state: dict[str, int] = {}

    def capture_placeholder_count(module, args, kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is not None:
            validation_state["placeholder_tokens"] = int(
                (input_ids == module.config.image_token_id).sum().item()
            )

    gemma_model.register_forward_pre_hook(capture_placeholder_count, with_kwargs=True)

    def normalized_get_image_features(self, pixel_values, image_position_ids=None, **kwargs):
        outputs = original_get_image_features(pixel_values, image_position_ids, **kwargs)
        features = outputs.pooler_output
        feature_groups = (features,) if isinstance(features, torch.Tensor) else tuple(features)
        normalized = []
        projected_groups = 0
        for index, feature in enumerate(feature_groups):
            if not isinstance(feature, torch.Tensor) or feature.ndim < 2:
                raise RuntimeError(
                    f"Gemma image feature group {index} has invalid shape: {getattr(feature, 'shape', None)}"
                )
            feature = feature.reshape(-1, feature.shape[-1])
            if feature.shape[-1] == vision_width and vision_width != text_width:
                feature = self.embed_vision(inputs_embeds=feature)
                projected_groups += 1
            if feature.shape[-1] != text_width:
                raise RuntimeError(
                    f"Gemma image feature group {index} has width {feature.shape[-1]}; "
                    f"expected raw vision width {vision_width} or text width {text_width}"
                )
            normalized.append(feature)
        feature_tokens = sum(group.shape[0] for group in normalized)
        placeholder_tokens = validation_state.get("placeholder_tokens")
        if placeholder_tokens is not None and feature_tokens != placeholder_tokens:
            raise RuntimeError(
                f"Gemma image placeholder/feature mismatch before merge: "
                f"placeholders={placeholder_tokens}, features={feature_tokens}"
            )
        outputs.pooler_output = tuple(normalized)
        print(
            f"gemma_image_features=groups:{len(normalized)} "
            f"placeholders:{placeholder_tokens} tokens:{feature_tokens} width:{text_width} "
            f"projected_groups:{projected_groups}"
        )
        return outputs

    gemma_model.get_image_features = MethodType(normalized_get_image_features, gemma_model)


def disable_python_native_bmm(torch_module) -> bool:
    """Avoid the experimental Triton bmm override used by recent PyTorch builds."""
    python_native = getattr(getattr(torch_module, "backends", None), "python_native", None)
    disable_operations = getattr(python_native, "disable_operations", None)
    if disable_operations is None:
        print("torch_python_native_bmm=unavailable")
        return False
    disable_operations("bmm")
    print("torch_python_native_bmm=disabled")
    return True


def run_model(model_id: str, messages: list[dict], image_tokens: int, max_new_tokens: int) -> tuple[str, str]:
    import torch

    disable_python_native_bmm(torch)

    from packaging.version import Version
    from transformers import GenerationConfig, __version__ as transformers_version, pipeline

    minimum_version = Version("5.15.0")
    if Version(transformers_version) < minimum_version:
        raise RuntimeError(
            f"Transformers {transformers_version} is too old for Gemma 4 dynamic image features; "
            f"install Transformers >= {minimum_version}"
        )
    print(
        f"torch={torch.__version__} torch_cuda={torch.version.cuda} "
        f"transformers={transformers_version} model={model_id} images={len(messages[0]['content']) // 2}"
    )

    gpu_count = torch.cuda.device_count()
    if gpu_count < 2 and model_id.lower().endswith("31b-it"):
        raise RuntimeError(f"{model_id} requires two available GPUs for BF16 inference; found {gpu_count}")
    for index in range(gpu_count):
        free, total = torch.cuda.mem_get_info(index)
        print(
            f"cuda:{index} name={torch.cuda.get_device_name(index)!r} "
            f"capability={torch.cuda.get_device_capability(index)} "
            f"free/total GiB: {free / 2**30:.1f}/{total / 2**30:.1f}"
        )

    contact_pipe = pipeline(
        task="image-text-to-text",
        model=model_id,
        device_map="auto",
        dtype="auto",
        model_kwargs={"attn_implementation": "sdpa", "low_cpu_mem_usage": True},
    )
    device_map = getattr(contact_pipe.model, "hf_device_map", None)
    print(f"gemma_device_map={device_map}")
    for index in range(gpu_count):
        free, total = torch.cuda.mem_get_info(index)
        print(f"cuda:{index} post-load free/total GiB: {free / 2**30:.1f}/{total / 2**30:.1f}")
    normalize_gemma4_image_feature_output(contact_pipe.model)
    contact_pipe.image_processor.max_soft_tokens = image_tokens
    print(f"max_soft_tokens_per_image={contact_pipe.image_processor.max_soft_tokens}")
    generation_config = GenerationConfig.from_pretrained(model_id)
    generation_config.max_new_tokens = max_new_tokens
    generation_config.do_sample = False
    generation_config.temperature = None
    generation_config.top_k = None
    generation_config.top_p = None
    output = contact_pipe(
        text=messages,
        return_full_text=False,
        enable_thinking=False,
        processor_kwargs={"truncation": False},
        generate_kwargs={"generation_config": generation_config},
    )
    raw_continuation = pipeline_response_text(output)
    final_response = restore_json_prefill(raw_continuation)
    return raw_continuation, final_response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", default=None)
    parser.add_argument("--needle-points", default=None)
    parser.add_argument("--thread-mask", default=None)
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--crop-padding", type=int, default=72)
    parser.add_argument("--image-tokens", type=int, choices=(70, 140, 280, 560, 1120), default=560)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    left_path = Path(args.left)
    right_path = Path(args.right) if args.right else None
    needle_points_path = Path(args.needle_points) if args.needle_points else None
    thread_mask_path = Path(args.thread_mask) if args.thread_mask else None
    for path in (left_path, right_path, needle_points_path, thread_mask_path):
        if path is not None and not path.exists():
            raise FileNotFoundError(path)

    out_dir = Path(args.out_dir)
    views = prepare_views(
        left_path,
        right_path,
        needle_points_path,
        thread_mask_path,
        out_dir,
        args.crop_padding,
    )
    messages = build_messages(views)
    manifest = {
        "model": args.model,
        "views": [{"label": label, "path": str(path.resolve())} for label, path in views],
        "image_tokens_per_view": args.image_tokens,
        "prompt": CONTACT_PROMPT,
    }
    (out_dir / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.prepare_only:
        print(json.dumps(manifest, indent=2))
        return

    raw, final_response = run_model(args.model, messages, args.image_tokens, args.max_new_tokens)
    raw_path = out_dir / "raw_response.txt"
    final_path = out_dir / "final_response.txt"
    raw_path.write_text(raw + "\n")
    final_path.write_text(final_response + "\n")
    try:
        assessment = validate_assessment(extract_json_object(final_response))
    except RuntimeError as error:
        raise RuntimeError(f"{error}; inspect {final_path}") from error
    result = {
        "model": args.model,
        "source_left": str(left_path.resolve()),
        "source_right": str(right_path.resolve()) if right_path else None,
        "assessment": assessment,
        "raw_response": raw,
        "final_response": final_response,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
