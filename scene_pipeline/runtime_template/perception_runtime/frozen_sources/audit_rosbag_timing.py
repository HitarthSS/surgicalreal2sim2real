#!/usr/bin/env python3
"""Read-only ROS2 SQLite timing audit; never creates training action labels."""
import argparse
import bisect
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
import struct

HEADER_TYPES = {
    "sensor_msgs/msg/Image", "sensor_msgs/msg/JointState",
    "sensor_msgs/msg/CameraInfo", "geometry_msgs/msg/PoseStamped",
    "geometry_msgs/msg/PointStamped", "geometry_msgs/msg/TwistStamped",
    "geometry_msgs/msg/WrenchStamped", "std_msgs/msg/Header",
}
LEFT = "/stereo/left/rectified_downscaled_image"
RIGHT = "/stereo/right/rectified_downscaled_image"
JOINT = "/PSM1/joint_states"
COMMANDS = ["/PSM1/setpoint_cp", "/PSM1/jaw/setpoint_js"]
LIMITATIONS = [
    "Header-only audit; message values, exposure timing and clock sources unverified.",
    "Nearest timestamps are diagnostics, not causal training action labels.",
    "Command/setpoint topic meaning and publisher lineage must be verified separately.",
    "No clock offsets are corrected or inferred.",
    "Database content hashes are not computed; metadata alone is not a cache identity.",
]


def header_stamp_ns(prefix, msgtype, serialization_format="cdr"):
    """Only plain CDR with a known leading Header layout is accepted."""
    if msgtype not in HEADER_TYPES:
        raise ValueError("unsupported_message_type")
    if serialization_format != "cdr":
        raise ValueError("unsupported_serialization")
    if len(prefix) < 12:
        raise ValueError("truncated_header")
    representation = int.from_bytes(prefix[:2], "big")
    if representation not in (0, 1):
        raise ValueError("unsupported_cdr_representation")
    sec, nsec = struct.unpack((">" if representation == 0 else "<") + "iI", prefix[4:12])
    if nsec >= 1_000_000_000:
        raise ValueError("invalid_nanoseconds")
    result = sec * 1_000_000_000 + nsec
    if result <= 0:
        raise ValueError("nonpositive_header_stamp")
    return result


def describe(values):
    if not values:
        return {"n": 0, "min": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "n": len(values), "min": ordered[0], "median": statistics.median(ordered),
        "p95": ordered[max(0, math.ceil(.95 * len(ordered)) - 1)], "max": ordered[-1],
    }


def pair_stamps(observations, candidates, limit_ms):
    """Report nearest and latest-preceding stamps; never match a future causal sample."""
    candidates = sorted(candidates)
    nearest, causal = [], []
    missing_nearest = missing_causal = 0
    for stamp in observations:
        i = bisect.bisect_left(candidates, stamp)
        choices = candidates[max(0, i - 1):min(len(candidates), i + 1)]
        if choices:
            nearest.append(min(abs(x - stamp) for x in choices) / 1e6)
        else:
            missing_nearest += 1
        j = bisect.bisect_right(candidates, stamp) - 1
        if j >= 0:
            causal.append((stamp - candidates[j]) / 1e6)
        else:
            missing_causal += 1
    return {
        "observation_count": len(observations),
        "nearest_header_delta_ms": describe(nearest),
        "latest_preceding_age_ms": describe(causal),
        "missing_nearest_count": missing_nearest,
        "missing_preceding_count": missing_causal,
        "limit_ms": limit_ms,
        "nearest_over_limit_count": sum(x > limit_ms for x in nearest),
        "preceding_over_limit_count": sum(x > limit_ms for x in causal),
        "causal_fail_count": missing_causal + sum(x > limit_ms for x in causal),
    }


def audit_bag(bag, image_topic=LEFT, stereo_topic=RIGHT, state_topics=None,
              command_topics=None, stereo_limit_ms=2., state_limit_ms=20.,
              command_limit_ms=100., receipt_warning_ms=100.):
    bag = Path(bag).resolve()
    files = [bag] if bag.is_file() and bag.suffix == ".db3" else sorted(bag.glob("*.db3"))
    if not files:
        raise ValueError("No SQLite .db3 files found; MCAP/compressed bags are not supported.")
    state_topics = [JOINT] if state_topics is None else list(state_topics)
    command_topics = COMMANDS if command_topics is None else list(command_topics)
    limits = [stereo_limit_ms, state_limit_ms, command_limit_ms, receipt_warning_ms]
    if any(not math.isfinite(x) or x <= 0 for x in limits):
        raise ValueError("All timing limits must be positive and finite.")
    selected = list(dict.fromkeys([image_topic, stereo_topic] + state_topics + command_topics))
    samples = defaultdict(list)
    invalid = defaultdict(Counter)
    total = Counter()
    types = defaultdict(set)
    provenance = []
    for path in files:
        before = path.stat()
        provenance.append({
            "path": str(path), "bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
            "content_sha256": None,
        })
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        try:
            connection.execute("PRAGMA query_only = ON")
            columns = {r[1] for r in connection.execute("PRAGMA table_info(topics)")}
            if not {"id", "name", "type", "serialization_format"} <= columns:
                raise ValueError("Unsupported ROS2 SQLite topics schema.")
            topic_rows = list(connection.execute(
                "SELECT id, name, type, serialization_format FROM topics ORDER BY id"))
            for topic_id, name, msgtype, serialization in topic_rows:
                if name not in selected:
                    continue
                types[name].add(msgtype)
                # Read only the CDR header prefix, not multi-megabyte image payloads.
                for message_id, receipt, prefix in connection.execute(
                    "SELECT id, timestamp, substr(data, 1, 16) FROM messages "
                    "WHERE topic_id = ? ORDER BY timestamp, id", (topic_id,)):
                    total[name] += 1
                    try:
                        stamp = header_stamp_ns(prefix, msgtype, serialization)
                    except ValueError as error:
                        invalid[name][str(error)] += 1
                        continue
                    samples[name].append((int(receipt), stamp))
        finally:
            connection.close()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("Input database changed during audit; retry with a closed recording.")
    topic_reports = {}
    flags = []
    for name in selected:
        data = sorted(samples[name])
        stamps = [x[1] for x in data]
        offsets = [(r - s) / 1e6 for r, s in data]
        gaps = [(b - a) / 1e6 for a, b in zip(stamps, stamps[1:])]
        regressions = sum(x < 0 for x in gaps)
        topic_reports[name] = {
            "types": sorted(types[name]), "message_count": total[name],
            "valid_header_count": len(data), "invalid_headers": dict(invalid[name]),
            "duplicate_header_count": len(stamps) - len(set(stamps)),
            "header_regression_count": regressions,
            "positive_header_interval_ms": describe([x for x in gaps if x > 0]),
            "receipt_minus_header_ms": describe(offsets),
            "receipt_header_warning_count": sum(abs(x) > receipt_warning_ms for x in offsets),
        }
        if not data:
            flags.append({"topic": name, "reason": "missing_or_no_valid_headers"})
        if invalid[name]:
            flags.append({"topic": name, "reason": "invalid_headers", "counts": dict(invalid[name])})
        if len(types[name]) > 1:
            flags.append({"topic": name, "reason": "mixed_message_types"})
        if regressions:
            flags.append({"topic": name, "reason": "header_time_regression", "count": regressions})
        if any(abs(x) > receipt_warning_ms for x in offsets):
            flags.append({"topic": name, "reason": "receipt_header_discrepancy"})
    images = [s for _, s in samples[image_topic]]
    pairings = {}
    for name in selected:
        if name == image_topic:
            continue
        is_command = name in command_topics
        limit = command_limit_ms if is_command else (
            stereo_limit_ms if name == stereo_topic else state_limit_ms)
        pairings[name] = pair_stamps(images, [s for _, s in samples[name]], limit)
        failures = (pairings[name]["causal_fail_count"] if is_command else
                    pairings[name]["missing_nearest_count"] + pairings[name]["nearest_over_limit_count"])
        if failures:
            flags.append({"topic": name, "reason": "timing_limit_exceeded", "count": failures,
                          "diagnostic": "preceding" if is_command else "nearest"})
    return {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "recording": bag.stem if bag.is_file() else bag.name, "bag": str(bag),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "databases": provenance, "image_topic": image_topic,
        "settings": {"stereo_limit_ms": stereo_limit_ms, "state_limit_ms": state_limit_ms,
                     "command_limit_ms": command_limit_ms, "receipt_warning_ms": receipt_warning_ms,
                     "p95_method": "nearest_rank"},
        "topics": topic_reports, "image_pairing": pairings, "flags": flags,
        "training_ready": False, "limitations": LIMITATIONS,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New report file; never overwritten.")
    parser.add_argument("--image-topic", default=LEFT)
    parser.add_argument("--stereo-topic", default=RIGHT)
    parser.add_argument("--state-topic", action="append", help="Repeat; replaces default joint topic.")
    parser.add_argument("--command-topic", action="append", help="Repeat; replaces default setpoint topics.")
    parser.add_argument("--stereo-limit-ms", type=float, default=2.)
    parser.add_argument("--state-limit-ms", type=float, default=20.)
    parser.add_argument("--command-limit-ms", type=float, default=100.)
    parser.add_argument("--receipt-warning-ms", type=float, default=100.)
    args = parser.parse_args()
    try:
        if args.out.exists():
            raise FileExistsError("Output already exists; choose a new report path.")
        report = audit_bag(args.bag, args.image_topic, args.stereo_topic, args.state_topic,
                           args.command_topic, args.stereo_limit_ms, args.state_limit_ms,
                           args.command_limit_ms, args.receipt_warning_ms)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (ValueError, RuntimeError, OSError, sqlite3.Error) as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps({"report": str(args.out), "recording": report["recording"],
                      "flags": len(report["flags"]), "training_ready": False}))


if __name__ == "__main__":
    main()
