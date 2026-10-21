#!/usr/bin/env python3
"""Convert EndoVis palette labels into SAM 3 COCO training annotations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


CATEGORIES = (
    {"id": 1, "name": "suture thread", "rgb": (187, 155, 25)},
    {"id": 2, "name": "curved surgical suture needle", "rgb": (255, 255, 125)},
)
PRESENCE_KEYS = ("both", "thread_only", "needle_only", "neither")


def encode_uncompressed_rle(mask: np.ndarray) -> dict:
    """Return COCO's uncompressed, column-major run-length encoding."""
    pixels = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    if pixels.size == 0:
        return {"size": list(mask.shape), "counts": []}
    changes = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
    counts = np.diff(np.concatenate(([0], changes, [pixels.size]))).tolist()
    if pixels[0]:
        counts.insert(0, 0)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def component_annotations(
    mask: np.ndarray,
    image_id: int,
    category_id: int,
    annotation_id: int,
    min_component_area: int,
) -> tuple[list[dict], int]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    annotations = []
    for component_id in range(1, count):
        x, y, width, height, area = stats[component_id].tolist()
        if area < min_component_area:
            continue
        component = labels == component_id
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": encode_uncompressed_rle(component),
                "area": int(area),
                "bbox": [int(x), int(y), int(width), int(height)],
                "iscrowd": 0,
            }
        )
        annotation_id += 1
    return annotations, annotation_id


def classify_presence(has_thread: bool, has_needle: bool) -> str:
    if has_thread and has_needle:
        return "both"
    if has_thread:
        return "thread_only"
    if has_needle:
        return "needle_only"
    return "neither"


def evenly_select(items: list[dict], count: int) -> list[dict]:
    if count >= len(items):
        return list(items)
    if count <= 0:
        return []
    indices = np.linspace(0, len(items) - 1, num=count, dtype=int)
    return [items[index] for index in sorted(set(indices.tolist()))]


def subset_coco(coco: dict, image_ids: set[int]) -> dict:
    return {
        "info": coco["info"],
        "licenses": coco["licenses"],
        "categories": coco["categories"],
        "images": [image for image in coco["images"] if image["id"] in image_ids],
        "annotations": [
            annotation
            for annotation in coco["annotations"]
            if annotation["image_id"] in image_ids
        ],
    }


def find_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path, Path]]:
    image_paths = sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    label_paths = sorted(path for path in labels_dir.rglob("*.png") if path.is_file())
    images = {path.relative_to(images_dir): path for path in image_paths}
    labels = {path.relative_to(labels_dir): path for path in label_paths}
    missing_labels = sorted(set(images) - set(labels))
    missing_images = sorted(set(labels) - set(images))
    if missing_labels or missing_images:
        details = []
        if missing_labels:
            details.append(f"{len(missing_labels)} image(s) lack labels")
        if missing_images:
            details.append(f"{len(missing_images)} label(s) lack images")
        raise RuntimeError("Dataset pairing failed: " + ", ".join(details))
    if not images:
        raise RuntimeError(f"No image files found under {images_dir}")
    return [(relative, images[relative], labels[relative]) for relative in sorted(images)]


def assign_splits(
    records: list[dict], validation_fraction: float, strategy: str = "stratified"
) -> None:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if strategy not in {"stratified", "contiguous", "all_train"}:
        raise ValueError("split strategy must be stratified, contiguous, or all_train")
    for record in records:
        record["split"] = "train"
    if strategy == "all_train" or validation_fraction == 0:
        return
    if strategy == "contiguous":
        validation_count = int(round(len(records) * validation_fraction))
        if len(records) > 1:
            validation_count = max(1, min(validation_count, len(records) - 1))
        for record in records[-validation_count:]:
            record["split"] = "validation"
        return

    for presence in PRESENCE_KEYS:
        group = [record for record in records if record["presence"] == presence]
        validation_count = int(round(len(group) * validation_fraction))
        if len(group) > 1:
            validation_count = max(1, min(validation_count, len(group) - 1))
        elif len(group) == 1:
            validation_count = 0
        for record in evenly_select(group, validation_count):
            record["split"] = "validation"


def split_summary(records: list[dict]) -> dict:
    counts = Counter(record["presence"] for record in records)
    return {
        "images": len(records),
        "presence": {key: counts.get(key, 0) for key in PRESENCE_KEYS},
        "thread_pixels": sum(record["thread_pixels"] for record in records),
        "needle_pixels": sum(record["needle_pixels"] for record in records),
    }


def prepare_dataset(
    dataset_root: Path,
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    validation_fraction: float = 0.2,
    split_strategy: str = "stratified",
    max_negative_fraction: float = 0.35,
    min_component_area: int = 4,
    write_binary_masks: bool = True,
) -> dict:
    dataset_root = dataset_root.resolve()
    images_dir = images_dir.resolve()
    labels_dir = labels_dir.resolve()
    output_dir = output_dir.resolve()
    try:
        images_dir.relative_to(dataset_root)
    except ValueError as error:
        raise RuntimeError("images_dir must be inside dataset_root") from error
    if not 0.0 <= max_negative_fraction < 1.0:
        raise ValueError("max_negative_fraction must be in [0, 1)")
    if min_component_area < 1:
        raise ValueError("min_component_area must be positive")

    pairs = find_pairs(images_dir, labels_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if write_binary_masks:
        for category in ("thread", "needle"):
            (output_dir / "binary_masks" / category).mkdir(parents=True, exist_ok=True)

    coco = {
        "info": {
            "description": "EndoVis thread and needle annotations for SAM 3",
            "version": "1.0",
        },
        "licenses": [],
        "categories": [
            {"id": category["id"], "name": category["name"]}
            for category in CATEGORIES
        ],
        "images": [],
        "annotations": [],
    }
    records = []
    annotation_id = 1
    for image_id, (relative, image_path, label_path) in enumerate(pairs, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        label = np.asarray(Image.open(label_path).convert("RGB"))
        if label.shape[:2] != (height, width):
            raise RuntimeError(
                f"Dimension mismatch for {relative}: image {(width, height)}, "
                f"label {(label.shape[1], label.shape[0])}"
            )
        masks = {
            "thread": np.all(label == np.asarray(CATEGORIES[0]["rgb"]), axis=2),
            "needle": np.all(label == np.asarray(CATEGORIES[1]["rgb"]), axis=2),
        }
        instance_counts = {}
        for key, category in zip(("thread", "needle"), CATEGORIES):
            annotations, annotation_id = component_annotations(
                masks[key],
                image_id,
                category["id"],
                annotation_id,
                min_component_area,
            )
            coco["annotations"].extend(annotations)
            instance_counts[key] = len(annotations)
            if write_binary_masks:
                destination = output_dir / "binary_masks" / key / relative.with_suffix(".png")
                destination.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(masks[key].astype(np.uint8) * 255).save(destination)

        file_name = image_path.relative_to(dataset_root).as_posix()
        coco["images"].append(
            {"id": image_id, "file_name": file_name, "width": width, "height": height}
        )
        thread_pixels = int(np.count_nonzero(masks["thread"]))
        needle_pixels = int(np.count_nonzero(masks["needle"]))
        records.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "presence": classify_presence(thread_pixels > 0, needle_pixels > 0),
                "thread_pixels": thread_pixels,
                "needle_pixels": needle_pixels,
                "thread_instances": instance_counts["thread"],
                "needle_instances": instance_counts["needle"],
            }
        )

    assign_splits(records, validation_fraction, split_strategy)
    train_records = [record for record in records if record["split"] == "train"]
    validation_records = [record for record in records if record["split"] == "validation"]
    positive_train = [record for record in train_records if record["presence"] != "neither"]
    negative_train = [record for record in train_records if record["presence"] == "neither"]
    if positive_train:
        max_negatives = math.floor(
            len(positive_train) * max_negative_fraction / (1.0 - max_negative_fraction)
        )
        selected_negatives = evenly_select(negative_train, max_negatives)
    else:
        selected_negatives = negative_train
    balanced_train = sorted(
        positive_train + selected_negatives, key=lambda record: record["image_id"]
    )

    outputs = {
        "all": (output_dir / "annotations_all.json", records),
        "train": (output_dir / "annotations_train.json", train_records),
        "train_balanced": (
            output_dir / "annotations_train_balanced.json",
            balanced_train,
        ),
        "validation": (
            output_dir / "annotations_validation.json",
            validation_records,
        ),
    }
    for _, (path, selected_records) in outputs.items():
        image_ids = {record["image_id"] for record in selected_records}
        path.write_text(json.dumps(subset_coco(coco, image_ids), indent=2) + "\n")

    with (output_dir / "presence.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "dataset_root": str(dataset_root),
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "output_dir": str(output_dir),
        "category_colors": {
            category["name"]: list(category["rgb"]) for category in CATEGORIES
        },
        "min_component_area": min_component_area,
        "validation_strategy": split_strategy,
        "validation_fraction": validation_fraction,
        "max_negative_fraction_in_balanced_train": max_negative_fraction,
        "all": split_summary(records),
        "train": split_summary(train_records),
        "train_balanced": split_summary(balanced_train),
        "validation": split_summary(validation_records),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--split-strategy",
        choices=("stratified", "contiguous", "all_train"),
        default="stratified",
    )
    parser.add_argument("--max-negative-fraction", type=float, default=0.35)
    parser.add_argument("--min-component-area", type=int, default=4)
    parser.add_argument("--no-write-binary-masks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_dataset(
        dataset_root=args.dataset_root,
        images_dir=args.images,
        labels_dir=args.labels,
        output_dir=args.out,
        validation_fraction=args.validation_fraction,
        split_strategy=args.split_strategy,
        max_negative_fraction=args.max_negative_fraction,
        min_component_area=args.min_component_area,
        write_binary_masks=not args.no_write_binary_masks,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
