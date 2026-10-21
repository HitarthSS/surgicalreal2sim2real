#!/usr/bin/env python3
"""Simulate a reconstructed thread spline with NVIDIA Newton.

The reconstructed spline is converted to a Newton rod:
``ModelBuilder.add_rod(...)`` builds capsule bodies connected by cable joints,
and ``SolverVBD`` advances the cable/rod dynamics.

Inputs can be either:
  - an offline reconstruction sample file: ``*_samples.npz`` with ``points``
  - a saved reconstruction pickle: ``*_spline.pkl`` containing a BSpline or
    ``{"thread": spline}``
"""

import argparse
import copy
import inspect
import math
import pickle
from pathlib import Path

import numpy as np


def parse_vec3(text):
    vals = [float(v.strip()) for v in text.split(",")]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    return tuple(vals)


def load_points(path, num_nodes, input_units):
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path)
        if "points" not in data:
            raise ValueError(f"{path} does not contain a 'points' array")
        points = np.asarray(data["points"], dtype=float)
    elif path.suffix == ".pkl":
        with path.open("rb") as f:
            obj = pickle.load(f)
        spline = obj.get("thread") if isinstance(obj, dict) else obj
        u = np.linspace(0.0, 1.0, max(200, num_nodes * 3))
        points = np.asarray(spline(u), dtype=float)
    else:
        raise ValueError("input must be a .npz or .pkl file")

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected points with shape (N, 3), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("input points contain NaN or inf")

    scale_by_unit = {
        "m": 1.0,
        "cm": 0.01,
        "mm": 0.001,
    }
    points = points * scale_by_unit[input_units]
    return resample_arclength(points, num_nodes)


def resample_arclength(points, n):
    diffs = np.diff(points, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    keep = np.concatenate([[True], seg > 1e-10])
    points = points[keep]
    if len(points) < 3:
        raise ValueError("need at least three distinct centerline points")

    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    target = np.linspace(0.0, s[-1], n)
    out = np.empty((n, 3), dtype=float)
    for dim in range(3):
        out[:, dim] = np.interp(target, s, points[:, dim])
    return out


def import_newton():
    try:
        import newton
        import warp as wp
    except ModuleNotFoundError as exc:
        missing = exc.name or "newton"
        raise SystemExit(
            f"Missing dependency '{missing}'. Install NVIDIA Newton first:\n"
            '  pip install "newton[examples]"\n'
            "Newton requires Python 3.10+; GPU runs need a supported NVIDIA "
            "driver, while macOS is CPU-only."
        ) from exc
    return newton, wp


def set_body_kinematic(builder, wp, body_idx):
    builder.body_mass[body_idx] = 0.0
    builder.body_inv_mass[body_idx] = 0.0
    builder.body_inertia[body_idx] = wp.mat33(0.0)
    builder.body_inv_inertia[body_idx] = wp.mat33(0.0)


def pin_rod_bodies(builder, wp, rod_bodies, pin_mode):
    if pin_mode in ("start", "both"):
        set_body_kinematic(builder, wp, int(rod_bodies[0]))
    if pin_mode in ("end", "both"):
        set_body_kinematic(builder, wp, int(rod_bodies[-1]))


def driven_body_index(rod_bodies, drive_end):
    if drive_end == "start":
        return int(rod_bodies[0])
    if drive_end == "end":
        return int(rod_bodies[-1])
    return None


def set_body_translation(state, body_idx, xyz):
    body_q = state.body_q.numpy()
    body_q[int(body_idx), :3] = np.asarray(xyz, dtype=body_q.dtype)
    state.body_q.assign(body_q)
    if getattr(state, "body_qd", None) is not None:
        body_qd = state.body_qd.numpy()
        body_qd[int(body_idx), :] = 0.0
        state.body_qd.assign(body_qd)


def body_positions(state, body_indices):
    body_q = state.body_q.numpy()
    return np.asarray([body_q[int(i)][:3] for i in body_indices], dtype=np.float64)


def normalize_vector(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-12:
        raise ValueError("cannot normalize a zero-length vector")
    return vector / norm


def axis_frame_from_z(direction):
    z_axis = normalize_vector(direction)
    reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(reference, z_axis))) > 0.9:
        reference = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = normalize_vector(np.cross(reference, z_axis))
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def matrix_to_quat_xyzw(matrix):
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    return quat / max(float(np.linalg.norm(quat)), 1.0e-12)


def quat_xyzw_to_matrix(quat):
    qx, qy, qz, qw = normalize_vector(np.asarray(quat, dtype=np.float64).reshape(4))
    return np.asarray(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
            [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
            [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def add_capsule_between(builder, wp, body, start, end, radius, cfg, label):
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    length = float(np.linalg.norm(end - start))
    if length < 1.0e-9:
        return None
    center = 0.5 * (start + end)
    quat = matrix_to_quat_xyzw(axis_frame_from_z(end - start))
    return builder.add_shape_capsule(
        body=body,
        xform=wp.transform(wp.vec3(*center), wp.quat(*quat)),
        radius=float(radius),
        half_height=0.5 * length,
        cfg=cfg,
        label=label,
    )


def add_needle_body(builder, wp, points, radius, density, collision_segments, as_link):
    points = np.asarray(points, dtype=np.float64)
    segment_count = max(2, min(int(collision_segments), len(points) - 1))
    collision_points = resample_arclength(points, segment_count + 1)
    body_origin = points.mean(axis=0)
    local_points = collision_points - body_origin
    identity = wp.transform(wp.vec3(*body_origin), wp.quat_identity())
    if as_link:
        body = builder.add_link(xform=identity, label="reconstructed_needle")
    else:
        body = builder.add_body(xform=identity, label="reconstructed_needle")
    cfg = copy.copy(builder.default_shape_cfg)
    cfg.density = float(density)
    for index, (start, end) in enumerate(zip(local_points[:-1], local_points[1:])):
        add_capsule_between(
            builder,
            wp,
            body,
            start,
            end,
            radius,
            cfg,
            f"reconstructed_needle_segment_{index}",
        )
    return int(body)


def transform_body_points(state, body_idx, local_points):
    pose = np.asarray(state.body_q.numpy()[int(body_idx)], dtype=np.float64)
    rotation = quat_xyzw_to_matrix(pose[3:7])
    return pose[:3] + np.asarray(local_points, dtype=np.float64) @ rotation.T


def rod_centerline(state, rod_bodies, rest_points):
    body_q = state.body_q.numpy()
    starts = np.asarray([body_q[int(index)][:3] for index in rod_bodies], dtype=np.float64)
    last_pose = np.asarray(body_q[int(rod_bodies[-1])], dtype=np.float64)
    last_length = float(np.linalg.norm(rest_points[-1] - rest_points[-2]))
    last_rotation = quat_xyzw_to_matrix(last_pose[3:7])
    end = last_pose[:3] + last_rotation @ np.asarray([0.0, 0.0, last_length])
    return np.vstack([starts, end])


def make_random_drive_trajectory(anchor, steps, dt, amplitude, waypoints, seed):
    rng = np.random.default_rng(seed)
    n_waypoints = max(int(waypoints), 2)
    key_t = np.linspace(0.0, steps * dt, n_waypoints)
    total_time = max(key_t[-1], dt)

    offsets = rng.normal(size=(n_waypoints, 3))
    offsets[:, 2] *= 0.6
    norms = np.linalg.norm(offsets, axis=1, keepdims=True)
    offsets = offsets / np.maximum(norms, 1e-12)
    scales = rng.uniform(0.35, 1.0, size=(n_waypoints, 1))
    offsets = offsets * scales * float(amplitude)
    offsets[0] = 0.0

    t = np.arange(steps + 1, dtype=float) * dt
    out = np.empty((steps + 1, 3), dtype=float)
    for i, ti in enumerate(t):
        seg = min(np.searchsorted(key_t, ti, side="right") - 1, n_waypoints - 2)
        seg = max(seg, 0)
        u = (ti - key_t[seg]) / max(key_t[seg + 1] - key_t[seg], 1e-12)
        u = np.clip(u, 0.0, 1.0)
        smooth = u * u * (3.0 - 2.0 * u)
        out[i] = anchor + (1.0 - smooth) * offsets[seg] + smooth * offsets[seg + 1]
    out[-1] = anchor + offsets[-1]
    return out


def simulate_newton(points, args, needle_points=None, needle_radius=0.00035, needle_thread_attachment=None, scene_extension=None, state_observer=None, step_controller=None):
    newton, wp = import_newton()
    if args.device:
        wp.set_device(args.device)

    # Newton's ModelBuilder constructor takes a signed scalar along up_axis,
    # not a full vector. With the default Z-up builder, (0, 0, -9.81) -> -9.81.
    builder = newton.ModelBuilder(gravity=float(args.gravity[2]))
    builder.default_shape_cfg.ke = args.contact_stiffness
    builder.default_shape_cfg.kd = args.contact_damping
    builder.default_shape_cfg.mu = args.friction
    builder.default_shape_cfg.gap = getattr(args, "contact_gap_m", None)

    points = np.asarray(points, dtype=np.float64)
    needle_points = None if needle_points is None else np.asarray(needle_points, dtype=np.float64)
    attachment_applied = bool(needle_thread_attachment and needle_thread_attachment.get("applied"))
    if needle_points is not None:
        if needle_points.ndim != 2 or needle_points.shape[1] != 3 or len(needle_points) < 2:
            raise ValueError(f"needle_points must have shape (N,3) with N >= 2, got {needle_points.shape}")
        if not np.isfinite(needle_points).all():
            raise ValueError("needle_points contain NaN or inf")
    if attachment_applied:
        if needle_points is None:
            raise ValueError("an applied needle-thread attachment requires needle_points")
        if args.closed:
            raise ValueError("needle-thread endpoint attachment requires an open thread rod")
        if int(needle_thread_attachment["thread_endpoint_index"]) not in {0, len(points) - 1}:
            raise ValueError("thread attachment index must select a material endpoint")
        if int(needle_thread_attachment["needle_endpoint_index"]) not in {0, len(needle_points) - 1}:
            raise ValueError("needle attachment index must select a material endpoint")
    thread_reversed = bool(
        getattr(args, "force_thread_reversal", False) or
        (attachment_applied and int(needle_thread_attachment["thread_endpoint_index"]) == 0)
    )
    model_points = points[::-1].copy() if thread_reversed else points.copy()

    wp_points = [wp.vec3(float(x), float(y), float(z)) for x, y, z in model_points]
    quats = newton.utils.create_parallel_transport_cable_quaternions(wp_points)

    rod_kwargs = dict(
        positions=wp_points,
        quaternions=quats,
        radius=args.radius,
        stretch_stiffness=args.stretch_stiffness,
        stretch_damping=args.stretch_damping,
        bend_stiffness=args.bend_stiffness,
        bend_damping=args.bend_damping,
        closed=args.closed,
        label="reconstructed_thread",
        wrap_in_articulation=not attachment_applied,
    )
    if "body_frame_origin" in inspect.signature(builder.add_rod).parameters:
        rod_kwargs["body_frame_origin"] = "start"
    rod_bodies, rod_joints = builder.add_rod(**rod_kwargs)

    needle_body = None
    attachment_joint = None
    if needle_points is not None:
        needle_body = add_needle_body(
            builder,
            wp,
            needle_points,
            needle_radius,
            getattr(args, "needle_density", 7800.0),
            getattr(args, "needle_collision_segments", 24),
            as_link=attachment_applied,
        )
        if attachment_applied:
            needle_endpoint = int(needle_thread_attachment["needle_endpoint_index"])
            segment_length = float(np.linalg.norm(model_points[-1] - model_points[-2]))
            parent_xform = wp.transform(
                wp.vec3(0.0, 0.0, segment_length),
                wp.quat_identity(),
            )
            child_point = needle_points[needle_endpoint] - needle_points.mean(axis=0)
            child_xform = wp.transform(wp.vec3(*child_point), wp.quat_identity())
            attachment_joint = builder.add_joint_ball(
                parent=int(rod_bodies[-1]),
                child=needle_body,
                parent_xform=parent_xform,
                child_xform=child_xform,
                label="needle_thread_attachment",
                collision_filter_parent=True,
            )
            builder.add_articulation(
                [*map(int, rod_joints), int(attachment_joint)],
                label="needle_thread_articulation",
            )
    pin_rod_bodies(builder, wp, rod_bodies, args.pin)
    drive_body = driven_body_index(rod_bodies, args.drive_end)
    if drive_body is not None:
        set_body_kinematic(builder, wp, drive_body)

    if args.ground:
        builder.add_ground_plane()

    if scene_extension is not None:
        scene_extension(builder, newton, wp, rod_bodies, needle_body)
    builder.color()
    model = builder.finalize()
    collision_pipeline = (
        newton.CollisionPipeline(model, contact_matching="latest")
        if (args.ground or args.contacts or needle_body is not None) and not getattr(args, 'diagnostic_disable_contacts', False)
        else None
    )
    contacts = collision_pipeline.contacts() if collision_pipeline is not None else None
    solver = newton.solvers.SolverVBD(
        model,
        iterations=args.iterations,
        rigid_avbd_contact_alpha=args.contact_alpha,
        rigid_avbd_joint_alpha=getattr(args, "joint_alpha", 0.95),
        rigid_joint_linear_ke=getattr(args, "joint_linear_ke", 1.0e5),
        rigid_body_contact_buffer_size=getattr(args, "body_contact_buffer_size", 64),
    )

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    drive_traj = None
    if drive_body is not None:
        anchor = points[0] if args.drive_end == "start" else points[-1]
        drive_traj = make_random_drive_trajectory(
            anchor=anchor,
            steps=args.steps,
            dt=args.dt,
            amplitude=args.drive_amplitude,
            waypoints=args.drive_waypoints,
            seed=args.drive_seed,
        )
        set_body_translation(state_0, drive_body, drive_traj[0])
        set_body_translation(state_1, drive_body, drive_traj[0])

    def thread_state(state):
        centerline = rod_centerline(state, rod_bodies, model_points)
        return centerline[::-1] if thread_reversed else centerline

    states = [thread_state(state_0)]
    needle_local_points = needle_points - needle_points.mean(axis=0) if needle_body is not None else None
    needle_states = [transform_body_points(state_0, needle_body, needle_local_points)] if needle_body is not None else []
    times = [0.0]
    saved_drive = [drive_traj[0].copy()] if drive_traj is not None else []

    if state_observer is not None:
        state_observer(state_0, 0, None)

    for step in range(1, args.steps + 1):
        state_0.clear_forces()
        if step_controller is not None:
            step_controller(state_0, state_1, step, args.dt)
        if drive_body is not None:
            set_body_translation(state_0, drive_body, drive_traj[step])
            set_body_translation(state_1, drive_body, drive_traj[step])
        refresh_contacts = (
            collision_pipeline is not None
            and (step == 1 or step % args.contact_interval == 0)
        )
        if refresh_contacts:
            collision_pipeline.collide(state_0, contacts)
        if hasattr(solver, "set_rigid_history_update"):
            solver.set_rigid_history_update(bool(refresh_contacts))
        solver.step(state_0, state_1, control, contacts, args.dt)
        state_0, state_1 = state_1, state_0
        if state_observer is not None:
            state_observer(state_0, step, contacts)

        if step % args.save_every == 0 or step == args.steps:
            states.append(thread_state(state_0))
            if needle_body is not None:
                needle_states.append(transform_body_points(state_0, needle_body, needle_local_points))
            times.append(step * args.dt)
            if drive_traj is not None:
                saved_drive.append(drive_traj[step].copy())

    result = {
        "times": np.asarray(times, dtype=np.float64),
        "states": np.asarray(states, dtype=np.float64),
        "initial_centerline": points,
        "rod_bodies": np.asarray(rod_bodies, dtype=np.int32),
        "rod_joints": np.asarray(rod_joints, dtype=np.int32),
        "thread_reversed_for_attachment": np.asarray(thread_reversed),
    }
    if needle_body is not None:
        result["needle_states"] = np.asarray(needle_states, dtype=np.float64)
        result["needle_body"] = np.asarray(needle_body, dtype=np.int32)
        result["needle_attachment_joint"] = np.asarray(
            -1 if attachment_joint is None else int(attachment_joint), dtype=np.int32
        )
    if drive_traj is not None:
        result["drive_trajectory"] = np.asarray(saved_drive, dtype=np.float64)
        result["drive_end"] = np.asarray(args.drive_end)
    return result


def write_preview(out_dir, initial, states):
    try:
        import os
        import tempfile

        os.environ.setdefault(
            "MPLCONFIGDIR",
            str(Path(tempfile.gettempdir()) / "thread_reconstruction_matplotlib"),
        )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"preview skipped: matplotlib unavailable ({exc})")
        return

    final = states[-1]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(initial[:, 0], initial[:, 1], initial[:, 2], color="0.55", lw=2, label="input centerline")
    ax.plot(final[:, 0], final[:, 1], final[:, 2], color="tab:red", lw=2, label="Newton final")
    ax.scatter(final[0, 0], final[0, 1], final[0, 2], color="cyan", s=35)
    ax.scatter(final[-1, 0], final[-1, 1], final[-1, 2], color="gold", s=35)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "newton_thread_preview_3d.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(initial[:, 0], initial[:, 1], color="0.55", lw=2, label="input centerline")
    ax.plot(final[:, 0], final[:, 1], color="tab:red", lw=2, label="Newton final")
    ax.scatter(final[0, 0], final[0, 1], color="cyan", s=35)
    ax.scatter(final[-1, 0], final[-1, 1], color="gold", s=35)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "newton_thread_preview_xy.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="*_samples.npz or *_spline.pkl")
    parser.add_argument("--out-dir", default="newton_thread_sim")
    parser.add_argument("--input-units", choices=("m", "cm", "mm"), default="mm")
    parser.add_argument("--num-nodes", type=int, default=65, help="rod centerline nodes; segments = nodes - 1")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--dt", type=float, default=1.0 / 1200.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--device", default=None, help="Newton/Warp device, e.g. cpu or cuda:0")

    parser.add_argument("--radius", type=float, default=0.001, help="rod/capsule radius in meters")
    parser.add_argument("--stretch-stiffness", type=float, default=1.0e6)
    parser.add_argument("--stretch-damping", type=float, default=1.0e3)
    parser.add_argument("--bend-stiffness", type=float, default=1.0e4)
    parser.add_argument("--bend-damping", type=float, default=1.0e2)
    parser.add_argument(
        "--twist-stiffness",
        type=float,
        default=None,
        help="accepted for compatibility; Newton add_rod uses bend-stiffness for bend/twist",
    )
    parser.add_argument(
        "--twist-damping",
        type=float,
        default=None,
        help="accepted for compatibility; Newton add_rod uses bend-damping for bend/twist",
    )
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--gravity", type=parse_vec3, default=(0.0, 0.0, -9.81))
    parser.add_argument("--pin", choices=("none", "start", "end", "both"), default="both")
    parser.add_argument("--closed", action="store_true")
    parser.add_argument("--drive-end", choices=("none", "start", "end"), default="none")
    parser.add_argument("--drive-amplitude", type=float, default=0.018, help="random endpoint motion amplitude in meters")
    parser.add_argument("--drive-waypoints", type=int, default=7, help="number of smooth random trajectory waypoints")
    parser.add_argument("--drive-seed", type=int, default=7)

    parser.add_argument("--ground", action="store_true")
    parser.add_argument("--contacts", action="store_true", help="enable collision pipeline without adding a ground plane")
    parser.add_argument("--contact-interval", type=int, default=10)
    parser.add_argument("--contact-alpha", type=float, default=0.0)
    parser.add_argument("--contact-stiffness", type=float, default=1.0e4)
    parser.add_argument("--contact-damping", type=float, default=0.0)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="write diagnostic Matplotlib PNG previews of the Newton trajectory",
    )
    args = parser.parse_args()

    if args.num_nodes < 3:
        raise ValueError("--num-nodes must be at least 3")
    if args.save_every < 1:
        raise ValueError("--save-every must be at least 1")
    if args.contact_interval < 1:
        raise ValueError("--contact-interval must be at least 1")

    points = load_points(args.input, args.num_nodes, args.input_units)
    result = simulate_newton(points, args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "newton_thread_simulation.npz",
        **result,
        input=str(Path(args.input).resolve()),
        input_units=args.input_units,
        radius=args.radius,
        stretch_stiffness=args.stretch_stiffness,
        bend_stiffness=args.bend_stiffness,
        pin=args.pin,
    )
    if args.preview:
        write_preview(out_dir, result["initial_centerline"], result["states"])
    print(f"saved {len(result['states'])} Newton states to {out_dir / 'newton_thread_simulation.npz'}")


if __name__ == "__main__":
    main()
