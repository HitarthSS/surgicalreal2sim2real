#!/usr/bin/env python3
"""Build a one-frame Newton scene with a VLM-gated needle-thread joint.

The reconstructed thread is a Newton rod and the reconstructed needle is a
compound rigid body. A ball joint is created only when the VLM reports an
attachment and the closest reconstructed endpoints pass a metric distance gate.
The tool remains a render-only scene element.
"""

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from simulate_newton_thread import load_points, parse_vec3, simulate_newton


def load_tool_scene(path):
    data = np.load(path)
    if "cam_to_ee" in data:
        pose = np.asarray(data["cam_to_ee"], dtype=float)
    elif "R" in data and "t" in data:
        pose = np.eye(4, dtype=float)
        pose[:3, :3] = np.asarray(data["R"], dtype=float)
        pose[:3, 3] = np.asarray(data["t"], dtype=float).reshape(3)
    else:
        keys = ", ".join(data.files)
        raise ValueError(f"{path} needs cam_to_ee or R/t arrays; found: {keys}")

    if pose.shape != (4, 4):
        raise ValueError(f"tool pose must be 4x4, got {pose.shape}")
    if not np.all(np.isfinite(pose)):
        raise ValueError("tool pose contains NaN or inf")

    branch_pts = None
    if "branch_pts_3d" in data:
        branch_pts = np.asarray(data["branch_pts_3d"], dtype=float).reshape(-1, 3)
        branch_pts = branch_pts[np.all(np.isfinite(branch_pts), axis=1)]
        if len(branch_pts) < 2:
            branch_pts = None

    return {
        "pose": pose,
        "branch_pts_3d": branch_pts,
    }


def normalize_quat_xyzw(quat):
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1.0e-12:
        raise ValueError("zero-length quaternion")
    return quat / norm


def quat_xyzw_to_matrix(quat):
    qx, qy, qz, qw = normalize_quat_xyzw(quat)
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def transform_needle_points(local_points, pose_xyzw):
    pose = np.asarray(pose_xyzw, dtype=np.float64).reshape(7)
    rot = quat_xyzw_to_matrix(pose[3:7])
    return pose[:3].reshape(1, 3) + np.asarray(local_points, dtype=np.float64) @ rot.T


def load_needle_scene(path):
    if path is None:
        return None

    data = np.load(path)
    if "needle_states" in data:
        states = np.asarray(data["needle_states"], dtype=np.float64)
    elif "local_centerline" in data and "poses_xyzw" in data:
        local_points = np.asarray(data["local_centerline"], dtype=np.float64)
        poses = np.asarray(data["poses_xyzw"], dtype=np.float64).reshape(-1, 7)
        states = np.stack([transform_needle_points(local_points, pose) for pose in poses], axis=0)
    elif "initial_world_centerline" in data:
        states = np.asarray(data["initial_world_centerline"], dtype=np.float64).reshape(1, -1, 3)
    else:
        keys = ", ".join(data.files)
        raise ValueError(
            f"{path} needs needle_states, local_centerline/poses_xyzw, or "
            f"initial_world_centerline; found: {keys}"
        )

    if states.ndim != 3 or states.shape[-1] != 3:
        raise ValueError(f"needle states must have shape (F,N,3), got {states.shape}")
    if states.shape[1] < 2:
        raise ValueError("needle centerline needs at least two points")
    if not np.all(np.isfinite(states)):
        raise ValueError("needle states contain NaN or inf")

    radius = 0.00035
    if "tube_radius" in data:
        radius = float(np.asarray(data["tube_radius"]).reshape(-1)[0])
    elif "needle_tube_radius" in data:
        radius = float(np.asarray(data["needle_tube_radius"]).reshape(-1)[0])

    return {
        "states": states,
        "tube_radius": radius,
        "source": str(Path(path).resolve()),
    }


def load_needle_thread_relation(path):
    data = json.loads(Path(path).read_text())
    assessment = data.get("assessment", data)
    relation = assessment.get("needle_thread_relation")
    if not isinstance(relation, dict):
        raise ValueError(f"{path} does not contain assessment.needle_thread_relation")
    state = str(relation.get("state", "")).lower()
    if state not in {"attached", "touching", "separate", "uncertain"}:
        raise ValueError(f"invalid needle_thread_relation.state: {state!r}")
    confidence = float(relation.get("confidence"))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"invalid needle_thread_relation.confidence: {confidence}")
    return {
        "state": state,
        "confidence": confidence,
        "evidence": str(relation.get("evidence", "")),
        "source": str(Path(path).resolve()),
    }


def evaluate_needle_thread_attachment(
    relation,
    thread_points,
    needle_points,
    min_confidence,
    max_distance,
):
    thread_points = np.asarray(thread_points, dtype=np.float64)
    needle_points = np.asarray(needle_points, dtype=np.float64)
    candidates = []
    for thread_index in (0, len(thread_points) - 1):
        for needle_index in (0, len(needle_points) - 1):
            distance = float(np.linalg.norm(thread_points[thread_index] - needle_points[needle_index]))
            candidates.append((distance, thread_index, needle_index))
    distance, thread_index, needle_index = min(candidates)

    state_ok = relation["state"] == "attached"
    confidence_ok = float(relation["confidence"]) >= float(min_confidence)
    distance_ok = distance <= float(max_distance)
    failed_checks = []
    if not state_ok:
        failed_checks.append("vlm_state_is_not_attached")
    if not confidence_ok:
        failed_checks.append("vlm_confidence_below_threshold")
    if not distance_ok:
        failed_checks.append("endpoint_distance_above_threshold")

    thread_endpoint = thread_points[thread_index]
    needle_endpoint = needle_points[needle_index]
    return {
        "applied": state_ok and confidence_ok and distance_ok,
        "failed_checks": failed_checks,
        "vlm_state": relation["state"],
        "vlm_confidence": float(relation["confidence"]),
        "vlm_evidence": relation.get("evidence", ""),
        "vlm_source": relation.get("source", ""),
        "min_confidence": float(min_confidence),
        "max_endpoint_distance_m": float(max_distance),
        "endpoint_distance_m": distance,
        "thread_endpoint_index": int(thread_index),
        "needle_endpoint_index": int(needle_index),
        "thread_endpoint_m": thread_endpoint.tolist(),
        "needle_endpoint_m": needle_endpoint.tolist(),
        "attachment_midpoint_m": (0.5 * (thread_endpoint + needle_endpoint)).tolist(),
    }


def axis_from_points(points):
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    axis = axis / max(np.linalg.norm(axis), 1.0e-12)
    return axis


def orthogonal_unit(axis, preferred):
    vec = preferred - axis * np.dot(preferred, axis)
    norm = np.linalg.norm(vec)
    if norm < 1.0e-9:
        reference = np.asarray([1.0, 0.0, 0.0])
        if abs(np.dot(reference, axis)) > 0.9:
            reference = np.asarray([0.0, 1.0, 0.0])
        vec = reference - axis * np.dot(reference, axis)
        norm = np.linalg.norm(vec)
    return vec / norm


def make_tool_proxy(tool_scene, shaft_length, jaw_length, jaw_opening):
    pose = tool_scene["pose"]
    branch_pts = tool_scene["branch_pts_3d"]

    if branch_pts is not None:
        axis = axis_from_points(branch_pts)
        proj = branch_pts @ axis
        p0 = branch_pts[np.argmin(proj)]
        p1 = branch_pts[np.argmax(proj)]
        if np.linalg.norm(p1 - p0) < 1.0e-6:
            center = branch_pts.mean(axis=0)
            p0 = center - 0.5 * shaft_length * axis
            p1 = center + 0.5 * shaft_length * axis
    else:
        axis = -pose[:3, 2]
        axis = axis / max(np.linalg.norm(axis), 1.0e-12)
        p1 = pose[:3, 3]
        p0 = p1 - shaft_length * axis

    # Prefer the pose y-axis as the jaw opening direction, but keep it
    # orthogonal to the estimated shaft.
    jaw_dir = orthogonal_unit(axis, pose[:3, 1])
    tip = p0
    jaw_base = tip + axis * (0.15 * jaw_length)
    jaw_forward = axis * jaw_length
    jaw_side = jaw_dir * (0.5 * jaw_opening)
    jaw_left = np.vstack([jaw_base, jaw_base + jaw_forward + jaw_side])
    jaw_right = np.vstack([jaw_base, jaw_base + jaw_forward - jaw_side])
    shaft = np.vstack([p1, tip])

    segments = np.stack([shaft, jaw_left, jaw_right], axis=0)
    return {
        "segments": segments,
        "points": np.vstack([segments.reshape(-1, 3), branch_pts if branch_pts is not None else np.empty((0, 3))]),
    }


def resolve_jaw_opening(jaw_path, jaw_length, configured_opening):
    """Convert the recorded total jaw angle to proxy tip separation."""
    if jaw_path:
        jaw = np.asarray(np.load(jaw_path), dtype=float).reshape(-1)
        if jaw.size != 1 or not np.isfinite(jaw[0]):
            raise RuntimeError(f"Expected one finite jaw value in {jaw_path}")
        jaw_angle = float(jaw[0])
        separation = 2.0 * jaw_length * np.sin(0.5 * abs(jaw_angle))
        return jaw_angle, float(separation)
    separation = 0.010 if configured_opening is None else float(configured_opening)
    return None, separation


def make_sim_args(args):
    return SimpleNamespace(
        device=args.device,
        gravity=args.gravity,
        contact_stiffness=args.contact_stiffness,
        contact_damping=args.contact_damping,
        friction=args.friction,
        radius=args.radius,
        stretch_stiffness=args.stretch_stiffness,
        stretch_damping=args.stretch_damping,
        bend_stiffness=args.bend_stiffness,
        bend_damping=args.bend_damping,
        closed=False,
        pin="none",
        drive_end="none",
        drive_amplitude=0.0,
        drive_waypoints=2,
        drive_seed=0,
        ground=args.ground,
        contacts=args.ground,
        iterations=args.iterations,
        contact_alpha=args.contact_alpha,
        steps=args.steps,
        dt=args.dt,
        save_every=max(1, args.save_every),
        contact_interval=args.contact_interval,
        needle_density=args.needle_density,
        needle_collision_segments=args.needle_collision_segments,
    )


def pad_from_scene(thread_points, tool_points, z_offset, margin):
    pts = np.vstack([thread_points, tool_points])
    low = pts.min(axis=0)
    high = pts.max(axis=0)
    z = low[2] - z_offset
    return np.asarray(
        [
            [low[0] - margin, low[1] - margin, z],
            [high[0] + margin, low[1] - margin, z],
            [high[0] + margin, high[1] + margin, z],
            [low[0] - margin, high[1] + margin, z],
        ],
        dtype=float,
    )


def set_equal_axes(ax, arrays):
    pts = np.vstack([a.reshape(-1, 3) for a in arrays if a.size])
    center = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
    radius = 0.5 * np.max(pts.max(axis=0) - pts.min(axis=0))
    radius = max(float(radius), 1.0e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def write_preview(out_dir, sim_result, tool_scene, tool_proxy, pad_corners, axis_len, needle_scene=None):
    import os

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "thread_reconstruction_matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    initial = sim_result["initial_centerline"]
    frame = sim_result["states"][-1]
    needle = None if needle_scene is None else needle_scene["states"][-1]
    tool_pose = tool_scene["pose"]
    branch_pts = tool_scene["branch_pts_3d"]
    tool_t = tool_pose[:3, 3]
    tool_axes = tool_pose[:3, :3] * axis_len
    closed_pad = np.vstack([pad_corners, pad_corners[0]])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(initial[:, 0], initial[:, 1], initial[:, 2], color="0.65", lw=2, label="input thread")
    ax.plot(frame[:, 0], frame[:, 1], frame[:, 2], color="tab:red", lw=2.5, label="Newton thread")
    if needle is not None:
        ax.plot(needle[:, 0], needle[:, 1], needle[:, 2], color="tab:cyan", lw=3, label="Newton needle")
    ax.plot(closed_pad[:, 0], closed_pad[:, 1], closed_pad[:, 2], color="tab:green", lw=1.5, label="background pad")
    ax.plot_trisurf(
        pad_corners[:, 0],
        pad_corners[:, 1],
        pad_corners[:, 2],
        triangles=np.asarray([[0, 1, 2], [0, 2, 3]]),
        color="lightgreen",
        alpha=0.18,
        shade=False,
    )
    for vec, color, label in zip(tool_axes.T, ("tab:blue", "tab:orange", "tab:purple"), ("tool x", "tool y", "tool z")):
        end = tool_t + vec
        ax.plot([tool_t[0], end[0]], [tool_t[1], end[1]], [tool_t[2], end[2]], color=color, lw=2, label=label)
    for i, seg in enumerate(tool_proxy["segments"]):
        label = "tool proxy" if i == 0 else None
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="black", lw=4 if i == 0 else 3, label=label)
    if branch_pts is not None:
        ax.scatter(branch_pts[:, 0], branch_pts[:, 1], branch_pts[:, 2], color="magenta", s=24, label="pose 3d points")
    ax.scatter(tool_t[0], tool_t[1], tool_t[2], color="black", s=35, label="tool pose")
    ax.scatter(frame[0, 0], frame[0, 1], frame[0, 2], color="cyan", s=35, label="thread start")
    ax.scatter(frame[-1, 0], frame[-1, 1], frame[-1, 2], color="gold", s=35, label="thread end")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    arrays = [initial, frame, pad_corners, tool_proxy["points"], tool_t.reshape(1, 3), tool_t.reshape(1, 3) + tool_axes.T]
    if needle is not None:
        arrays.append(needle)
    set_equal_axes(ax, arrays)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "scene_preview_3d.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(initial[:, 0], initial[:, 1], color="0.65", lw=2, label="input thread")
    ax.plot(frame[:, 0], frame[:, 1], color="tab:red", lw=2.5, label="Newton thread")
    if needle is not None:
        ax.plot(needle[:, 0], needle[:, 1], color="tab:cyan", lw=3, label="Newton needle")
    ax.plot(closed_pad[:, 0], closed_pad[:, 1], color="tab:green", lw=1.5, label="background pad")
    for i, seg in enumerate(tool_proxy["segments"]):
        label = "tool proxy" if i == 0 else None
        ax.plot(seg[:, 0], seg[:, 1], color="black", lw=4 if i == 0 else 3, label=label)
    if branch_pts is not None:
        ax.scatter(branch_pts[:, 0], branch_pts[:, 1], color="magenta", s=24, label="pose 3d points")
    ax.scatter(tool_t[0], tool_t[1], color="black", s=35, label="tool pose")
    ax.scatter(frame[0, 0], frame[0, 1], color="cyan", s=35, label="thread start")
    ax.scatter(frame[-1, 0], frame[-1, 1], color="gold", s=35, label="thread end")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "scene_preview_xy.png", dpi=170)
    plt.close(fig)


def load_rgb_image(path, width, height):
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[1] != width or image.shape[0] != height:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def make_synthetic_pad(width, height):
    yy, xx = np.mgrid[0:height, 0:width]
    x = xx.astype(np.float32) / max(width - 1, 1)
    y = yy.astype(np.float32) / max(height - 1, 1)
    base = np.zeros((height, width, 3), dtype=np.float32)
    base[..., 0] = 184.0 + 28.0 * (1.0 - y) + 8.0 * x
    base[..., 1] = 111.0 + 18.0 * (1.0 - y)
    base[..., 2] = 104.0 + 13.0 * x

    highlight = np.exp(-(((x - 0.72) / 0.22) ** 2 + ((y - 0.76) / 0.18) ** 2))
    vignette = 0.88 + 0.12 * np.exp(-(((x - 0.5) / 0.65) ** 2 + ((y - 0.52) / 0.65) ** 2))
    base = base * vignette[..., None] + highlight[..., None] * np.asarray([35.0, 28.0, 24.0])

    rng = np.random.default_rng(12)
    noise = rng.normal(0.0, 2.2, size=base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def project_points(points, fx, fy, cx, cy):
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    z = pts[:, 2]
    valid = np.isfinite(pts).all(axis=1) & (z > 1.0e-6)
    uv = np.full((len(pts), 2), np.nan, dtype=float)
    uv[valid, 0] = fx * pts[valid, 0] / z[valid] + cx
    uv[valid, 1] = fy * pts[valid, 1] / z[valid] + cy
    return uv, valid


def blend_line(image, p0, p1, color, thickness, alpha=1.0):
    import cv2

    overlay = image.copy()
    cv2.line(
        overlay,
        tuple(np.round(p0).astype(int)),
        tuple(np.round(p1).astype(int)),
        tuple(int(c) for c in color),
        int(max(1, thickness)),
        lineType=cv2.LINE_AA,
    )
    if alpha >= 1.0:
        image[:] = overlay
    else:
        cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)


def draw_polyline_camera(image, points, fx, fy, cx, cy, color, radius_m, min_px, max_px, shadow=True):
    pts = np.asarray(points, dtype=float)
    uv, valid = project_points(pts, fx, fy, cx, cy)
    h, w = image.shape[:2]
    for i in range(len(pts) - 1):
        if not (valid[i] and valid[i + 1]):
            continue
        p0 = uv[i]
        p1 = uv[i + 1]
        if (
            max(p0[0], p1[0]) < -50
            or min(p0[0], p1[0]) > w + 50
            or max(p0[1], p1[1]) < -50
            or min(p0[1], p1[1]) > h + 50
        ):
            continue
        z = max(0.5 * (pts[i, 2] + pts[i + 1, 2]), 1.0e-6)
        thickness = int(np.clip(fx * radius_m / z * 2.0, min_px, max_px))
        if shadow:
            blend_line(image, p0 + np.asarray([2.0, 2.0]), p1 + np.asarray([2.0, 2.0]), (35, 28, 28), thickness + 3, alpha=0.26)
        blend_line(image, p0, p1, color, thickness, alpha=1.0)
        blend_line(image, p0 - np.asarray([1.0, 1.0]), p1 - np.asarray([1.0, 1.0]), (255, 245, 232), max(1, thickness // 3), alpha=0.45)


def draw_tool_camera(image, tool_proxy, fx, fy, cx, cy):
    for i, seg in enumerate(tool_proxy["segments"]):
        radius = 0.0025 if i == 0 else 0.0016
        draw_polyline_camera(
            image,
            seg,
            fx,
            fy,
            cx,
            cy,
            color=(62, 69, 74) if i == 0 else (44, 47, 51),
            radius_m=radius,
            min_px=4 if i == 0 else 3,
            max_px=18 if i == 0 else 12,
            shadow=True,
        )


def write_camera_render(
    out_dir,
    sim_result,
    tool_proxy,
    needle_scene,
    image_path,
    width,
    height,
    fx,
    fy,
    cx,
    cy,
    thread_radius,
):
    import cv2

    synthetic = make_synthetic_pad(width, height)
    thread = sim_result["states"][-1]
    needle = None if needle_scene is None else needle_scene["states"][-1]
    draw_polyline_camera(
        synthetic,
        thread,
        fx,
        fy,
        cx,
        cy,
        color=(232, 226, 216),
        radius_m=thread_radius,
        min_px=3,
        max_px=12,
        shadow=True,
    )
    if needle is not None:
        draw_polyline_camera(
            synthetic,
            needle,
            fx,
            fy,
            cx,
            cy,
            color=(52, 214, 233),
            radius_m=needle_scene["tube_radius"],
            min_px=2,
            max_px=10,
            shadow=True,
        )
    draw_tool_camera(synthetic, tool_proxy, fx, fy, cx, cy)
    cv2.imwrite(str(out_dir / "camera_scene_render.png"), cv2.cvtColor(synthetic, cv2.COLOR_RGB2BGR))

    if image_path:
        real = load_rgb_image(image_path, width, height)
        # Keep the real frame as context, but render the simulated objects with
        # enough opacity that misalignment is obvious.
        canvas = (0.70 * real + 0.30 * make_synthetic_pad(width, height)).astype(np.uint8)
        draw_polyline_camera(
            canvas,
            thread,
            fx,
            fy,
            cx,
            cy,
            color=(238, 231, 219),
            radius_m=thread_radius,
            min_px=3,
            max_px=12,
            shadow=True,
        )
        if needle is not None:
            draw_polyline_camera(
                canvas,
                needle,
                fx,
                fy,
                cx,
                cy,
                color=(52, 214, 233),
                radius_m=needle_scene["tube_radius"],
                min_px=2,
                max_px=10,
                shadow=True,
            )
        draw_tool_camera(canvas, tool_proxy, fx, fy, cx, cy)
        cv2.imwrite(str(out_dir / "camera_scene_on_real_frame.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread-samples", required=True, help="frame_00000_*_samples.npz")
    parser.add_argument("--tool-pose", required=True, help="pose estimator .npz for the same frame")
    parser.add_argument("--needle-npz", required=True, help="reconstructed rigid needle centerline NPZ")
    parser.add_argument("--contact-assessment", required=True, help="Gemma needle-thread assessment JSON")
    parser.add_argument("--out-dir", default="newton_frame_scene_000000")
    parser.add_argument("--input-units", choices=("m", "cm", "mm"), default="mm")
    parser.add_argument("--num-nodes", type=int, default=65)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--dt", type=float, default=1.0 / 1200.0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--device", default=None)

    parser.add_argument("--radius", type=float, default=0.001)
    parser.add_argument("--stretch-stiffness", type=float, default=1.0e6)
    parser.add_argument("--stretch-damping", type=float, default=1.0e3)
    parser.add_argument("--bend-stiffness", type=float, default=1.0e4)
    parser.add_argument("--bend-damping", type=float, default=1.0e2)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--gravity", type=parse_vec3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--ground", action="store_true")
    parser.add_argument("--contact-interval", type=int, default=10)
    parser.add_argument("--contact-alpha", type=float, default=0.0)
    parser.add_argument("--contact-stiffness", type=float, default=1.0e4)
    parser.add_argument("--contact-damping", type=float, default=0.0)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--needle-density", type=float, default=7800.0)
    parser.add_argument("--needle-collision-segments", type=int, default=24)
    parser.add_argument("--needle-thread-min-confidence", type=float, default=0.75)
    parser.add_argument("--needle-thread-max-distance-m", type=float, default=0.004)

    parser.add_argument("--pad-z-offset", type=float, default=0.01)
    parser.add_argument("--pad-margin", type=float, default=0.02)
    parser.add_argument("--tool-axis-len", type=float, default=0.025)
    parser.add_argument("--tool-shaft-length", type=float, default=0.06)
    parser.add_argument("--tool-jaw-length", type=float, default=0.018)
    parser.add_argument("--jaw", default=None, help="Recorded one-value jaw angle .npy")
    parser.add_argument(
        "--tool-jaw-opening",
        type=float,
        default=None,
        help="Proxy tip separation used only when --jaw is omitted.",
    )
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--camera-render", action="store_true")
    parser.add_argument("--image", default=None, help="optional left frame image for a camera-aligned render")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fx", type=float, default=1025.8822021484375)
    parser.add_argument("--fy", type=float, default=1025.8822021484375)
    parser.add_argument("--cx", type=float, default=167.919017)
    parser.add_argument("--cy", type=float, default=234.152707)
    args = parser.parse_args()

    if args.num_nodes < 3:
        raise ValueError("--num-nodes must be at least 3")
    if args.steps < 0:
        raise ValueError("--steps must be non-negative")
    if not 0.0 <= args.needle_thread_min_confidence <= 1.0:
        raise ValueError("--needle-thread-min-confidence must be between 0 and 1")
    if args.needle_thread_max_distance_m <= 0.0:
        raise ValueError("--needle-thread-max-distance-m must be positive")
    if args.needle_density <= 0.0:
        raise ValueError("--needle-density must be positive")
    if args.needle_collision_segments < 2:
        raise ValueError("--needle-collision-segments must be at least 2")

    thread_points = load_points(args.thread_samples, args.num_nodes, args.input_units)
    tool_scene = load_tool_scene(args.tool_pose)
    needle_scene = load_needle_scene(args.needle_npz)
    jaw_angle, jaw_opening = resolve_jaw_opening(
        args.jaw, args.tool_jaw_length, args.tool_jaw_opening
    )
    relation = load_needle_thread_relation(args.contact_assessment)
    attachment = evaluate_needle_thread_attachment(
        relation,
        thread_points,
        needle_scene["states"][-1],
        args.needle_thread_min_confidence,
        args.needle_thread_max_distance_m,
    )
    tool_proxy = make_tool_proxy(
        tool_scene,
        shaft_length=args.tool_shaft_length,
        jaw_length=args.tool_jaw_length,
        jaw_opening=jaw_opening,
    )
    pad_corners = pad_from_scene(thread_points, tool_proxy["points"], args.pad_z_offset, args.pad_margin)

    sim_result = simulate_newton(
        thread_points,
        make_sim_args(args),
        needle_points=needle_scene["states"][-1],
        needle_radius=needle_scene["tube_radius"],
        needle_thread_attachment=attachment,
    )
    simulated_needle_scene = dict(needle_scene)
    simulated_needle_scene["states"] = sim_result["needle_states"]
    attachment["post_sim_endpoint_distance_m"] = float(
        np.linalg.norm(
            sim_result["states"][-1, attachment["thread_endpoint_index"]]
            - sim_result["needle_states"][-1, attachment["needle_endpoint_index"]]
        )
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "needle_thread_attachment_report.json").write_text(
        json.dumps(attachment, indent=2) + "\n"
    )
    np.savez(
        out_dir / "newton_frame_scene.npz",
        **sim_result,
        thread_samples=str(Path(args.thread_samples).resolve()),
        tool_pose_file=str(Path(args.tool_pose).resolve()),
        tool_cam_to_ee=tool_scene["pose"],
        tool_position=tool_scene["pose"][:3, 3],
        tool_jaw_angle_rad=np.asarray(
            np.nan if jaw_angle is None else jaw_angle, dtype=float
        ),
        tool_jaw_tip_separation_m=np.asarray(jaw_opening, dtype=float),
        tool_branch_pts_3d=(
            tool_scene["branch_pts_3d"]
            if tool_scene["branch_pts_3d"] is not None
            else np.empty((0, 3), dtype=float)
        ),
        tool_proxy_segments=tool_proxy["segments"],
        needle_centerline=sim_result["needle_states"][-1],
        needle_tube_radius=np.asarray(needle_scene["tube_radius"], dtype=float),
        needle_source=needle_scene["source"],
        needle_thread_attachment_applied=np.asarray(attachment["applied"]),
        needle_thread_endpoint_distance_m=np.asarray(attachment["endpoint_distance_m"]),
        needle_thread_vlm_state=np.asarray(attachment["vlm_state"]),
        needle_thread_vlm_confidence=np.asarray(attachment["vlm_confidence"]),
        needle_thread_contact_assessment=str(Path(args.contact_assessment).resolve()),
        pad_corners=pad_corners,
        input_units=args.input_units,
        radius=args.radius,
    )
    if args.preview:
        write_preview(
            out_dir,
            sim_result,
            tool_scene,
            tool_proxy,
            pad_corners,
            args.tool_axis_len,
            simulated_needle_scene,
        )
    if args.camera_render:
        write_camera_render(
            out_dir=out_dir,
            sim_result=sim_result,
            tool_proxy=tool_proxy,
            needle_scene=simulated_needle_scene,
            image_path=args.image,
            width=args.width,
            height=args.height,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy,
            thread_radius=args.radius,
        )

    print(f"saved one-frame scene bundle to {out_dir / 'newton_frame_scene.npz'}")
    print(
        "needle-thread attachment: "
        f"applied={attachment['applied']} state={attachment['vlm_state']} "
        f"confidence={attachment['vlm_confidence']:.3f} "
        f"endpoint_distance_mm={1000.0 * attachment['endpoint_distance_m']:.3f} "
        f"post_sim_distance_mm={1000.0 * attachment['post_sim_endpoint_distance_m']:.3f}"
    )
    if args.preview:
        print(f"preview: {out_dir / 'scene_preview_3d.png'}")
    if args.camera_render:
        print(f"camera render: {out_dir / 'camera_scene_render.png'}")
        if args.image:
            print(f"camera render on real frame: {out_dir / 'camera_scene_on_real_frame.png'}")


if __name__ == "__main__":
    main()
