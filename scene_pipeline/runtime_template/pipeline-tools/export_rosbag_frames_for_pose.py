#!/usr/bin/env python3
"""Export synchronized stereo ROS bag frames for tool pose estimation.

This creates a simple image-file dataset from the same ROS2 bag used by the
thread reconstruction.  The output layout is intentionally close to the
Genesis/tool-pose scripts that expect files such as:

    episode_0000/colors/left_image_000212.jpg

It also writes a manifest mapping frame indices to ROS timestamps so later
thread samples and tool poses can be synchronized.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

DEFAULT_LEFT = "/stereo/left/rectified_downscaled_image"
DEFAULT_RIGHT = "/stereo/right/rectified_downscaled_image"


def stamp_ns(msg):
    stamp = msg.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def image_msg_to_array(msg):
    channels_by_encoding = {
        "mono8": 1,
        "8uc1": 1,
        "rgb8": 3,
        "bgr8": 3,
        "8uc3": 3,
    }
    encoding = msg.encoding.lower()
    if encoding not in channels_by_encoding:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")
    channels = channels_by_encoding[encoding]
    height, width, step = int(msg.height), int(msg.width), int(msg.step)
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if data.size < height * step:
        raise ValueError(
            f"Image data too small for {height=} {width=} {step=}: {data.size} bytes"
        )
    rows = data[: height * step].reshape(height, step)
    useful = rows[:, : width * channels]
    if channels == 1:
        return useful.reshape(height, width)
    return useful.reshape(height, width, channels)


def as_rgb(msg):
    image = image_msg_to_array(msg)
    encoding = msg.encoding.lower()
    if encoding in ("rgb8", "8uc3"):
        return image
    if encoding == "bgr8":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if encoding in ("mono8", "8uc1"):
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    raise ValueError(f"Unsupported RGB image encoding: {msg.encoding}")


def write_rgb(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Failed to write image: {path}")


def read_synced_stereo_frames(bag_dir, left_topic, right_topic, tolerance_ns):
    topics = {"left": left_topic, "right": right_topic}
    stores = {name: {} for name in topics}
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    with AnyReader([Path(bag_dir)], default_typestore=typestore) as reader:
        wanted = set(topics.values())
        connections = [connection for connection in reader.connections if connection.topic in wanted]
        found = {connection.topic for connection in connections}
        missing = sorted(wanted - found)
        if missing:
            raise RuntimeError(f"Bag is missing required image topics: {missing}")

        topic_to_name = {topic: name for name, topic in topics.items()}
        for connection, _, rawdata in reader.messages(connections=connections):
            message = reader.deserialize(rawdata, connection.msgtype)
            stores[topic_to_name[connection.topic]][stamp_ns(message)] = message

    left_stamps = np.asarray(sorted(stores["left"]), dtype=np.int64)
    right_stamps = np.asarray(sorted(stores["right"]), dtype=np.int64)
    if not len(left_stamps) or not len(right_stamps):
        raise RuntimeError(
            f"Image topics are empty: left={len(left_stamps)}, right={len(right_stamps)}"
        )

    matches = []
    used_right = set()
    for left_stamp in left_stamps:
        position = int(np.searchsorted(right_stamps, left_stamp))
        candidates = []
        if position < len(right_stamps):
            candidates.append(int(right_stamps[position]))
        if position > 0:
            candidates.append(int(right_stamps[position - 1]))
        candidates = [stamp for stamp in candidates if stamp not in used_right]
        if not candidates:
            continue
        right_stamp = min(candidates, key=lambda stamp: abs(stamp - int(left_stamp)))
        if abs(right_stamp - int(left_stamp)) <= tolerance_ns:
            matches.append((int(left_stamp), right_stamp))
            used_right.add(right_stamp)

    if not matches:
        raise RuntimeError(
            "No synchronized stereo image pairs found. "
            f"Counts: left={len(left_stamps)}, right={len(right_stamps)}. "
            f"Try increasing --sync-tolerance-ms above {tolerance_ns / 1e6:.1f}."
        )
    print(
        f"Using {len(matches)} synchronized stereo image pair(s) "
        f"(tolerance {tolerance_ns / 1e6:.1f} ms)."
    )
    return stores, matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="ROS2 bag directory")
    parser.add_argument("--out", required=True, help="Output dataset directory")
    parser.add_argument("--left-topic", default=DEFAULT_LEFT)
    parser.add_argument("--right-topic", default=DEFAULT_RIGHT)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sync-tolerance-ms", type=float, default=75.0)
    parser.add_argument("--episode", default="episode_0000")
    args = parser.parse_args()

    tolerance_ns = int(args.sync_tolerance_ms * 1e6)
    stores, stamps = read_synced_stereo_frames(
        args.bag, args.left_topic, args.right_topic, tolerance_ns=tolerance_ns
    )
    selected = stamps[args.start_index :: max(args.stride, 1)]
    if args.max_frames > 0:
        selected = selected[: args.max_frames]
    if not selected:
        raise IndexError(
            f"--start-index {args.start_index} selected no frames from "
            f"{len(stamps)} synchronized pair(s). Valid nonnegative indices are "
            f"0..{len(stamps) - 1}; use -1 for the last pair."
        )

    out = Path(args.out)
    colors = out / args.episode / "colors"
    manifest_path = out / "frame_manifest.csv"
    out.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "left_stamp_ns",
                "right_stamp_ns",
                "left_image",
                "right_image",
            ],
        )
        writer.writeheader()

        for frame_i, stamp_tuple in enumerate(selected):
            left_stamp, right_stamp = stamp_tuple
            left = as_rgb(stores["left"][left_stamp])
            right = as_rgb(stores["right"][right_stamp])

            left_name = f"left_image_{frame_i:06d}.jpg"
            right_name = f"right_image_{frame_i:06d}.jpg"

            write_rgb(colors / left_name, left)
            write_rgb(colors / right_name, right)

            writer.writerow(
                {
                    "frame": frame_i,
                    "left_stamp_ns": left_stamp,
                    "right_stamp_ns": right_stamp,
                    "left_image": str((colors / left_name).relative_to(out)),
                    "right_image": str((colors / right_name).relative_to(out)),
                }
            )
            print(f"[{frame_i}] {left_name} stamp={left_stamp}")

    print(f"Wrote {len(selected)} frame(s) under {out}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
