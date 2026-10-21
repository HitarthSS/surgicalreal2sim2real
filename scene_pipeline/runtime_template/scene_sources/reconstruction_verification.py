"""Shared geometric checks; diagnostic consistency is never independent accuracy."""
import csv
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np


def load_base(tools):
    spec = importlib.util.spec_from_file_location("ismr_thread_base", Path(tools) / "reconstruct_thread_from_files.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inspect_case(inputs, tools):
    base = load_base(tools)
    left, right = base.load_rgb(inputs["left"]), base.load_rgb(inputs["right"])
    if left.shape != right.shape:
        raise ValueError("Stereo dimensions differ")
    shape = left.shape[:2]
    K = base.load_intrinsics(inputs["K"])
    if not np.isfinite(K).all() or abs(K[2, 2] - 1) > 1e-6:
        raise ValueError("Invalid intrinsic matrix")
    depth = np.squeeze(np.load(inputs["depth_npy"], allow_pickle=False))
    if depth.shape != shape:
        raise ValueError("Depth dimensions differ")
    observation = {"image_shape": list(shape), "depth_finite_fraction": float(np.isfinite(depth).mean()),
                   "images": {"left": inputs["left"], "right": inputs["right"]},
                   "needle_visibility": "not_estimated", "attachment_reliability": "not_estimated"}
    for prefix in ("thread", "needle"):
        if prefix + "_left_mask" not in inputs:
            continue
        mask = base.load_mask(inputs[prefix + "_left_mask"], shape)
        base.load_mask(inputs[prefix + "_right_mask"], shape)
        observation[prefix + "_mask_pixels"] = int(mask.sum())
        if prefix == "thread":
            try:
                _, graph, _, count, discarded = base.skeleton_graph(mask, 0., 30)
                observation.update(thread_component_count=count, discarded_short_components=discarded,
                                   thread_endpoint_count=sum(graph.degree[n] == 1 for n in graph))
            except RuntimeError as error:
                observation["thread_inspection_error"] = str(error)
    observation["left_occluder_pixels"] = (
        int(base.load_mask(inputs["left_occluder"], shape).sum()) if "left_occluder" in inputs else 0)
    if "right_occluder" in inputs:
        base.load_mask(inputs["right_occluder"], shape)
    return observation


def write_curve_obj(path, points, inferred):
    """Centerline only, in meters; preserves observation/inference groups."""
    lines = ["# Estimated centerline in meters; inferred geometry is not ground truth."]
    lines.extend("v " + " ".join(f"{v:.12g}" for v in point) for point in points)
    flags = inferred[:-1] | inferred[1:]
    for flag, name in ((False, "visible_curve_support"), (True, "inferred_bridge")):
        lines.append("o " + name)
        for i in np.flatnonzero(flags == flag):
            lines.append(f"l {i + 1} {i + 2}")
    Path(path).write_text("\n".join(lines) + "\n")


def verify_native_qp(inputs, output, tools, backend_report, limits):
    """Inspect an unchanged production spline; never invent a native join ledger."""
    return verify_thread(inputs, output, tools, backend_report, limits, native_qp=True)


def verify_thread(inputs, output, tools, backend_report, limits, native_qp=False):
    base = load_base(tools)
    output = Path(output)
    failures = []
    artifact = output / "thread_samples.npz"
    if not artifact.exists():
        return {"accepted": False, "failures": ["missing_thread_samples"], "metrics": {},
                "independent_accuracy_measured": False}
    with np.load(artifact, allow_pickle=False) as data:
        points = data["points_m"].copy()
        xy = data["pixel_xy"].copy()
        depths = points[:, 2].copy() if native_qp else data["depth_m"].copy()
        reliability = data["reliability"].copy()
        inferred = data["inferred"].copy() if "inferred" in data else None
        native_observed = data["observed"].copy() if "observed" in data else None
    n = len(points)
    if (points.shape != (n, 3) or xy.shape != (n, 2) or depths.shape != (n,)
            or reliability.shape != (n,) or n != limits["thread_samples"]):
        raise ValueError("Malformed thread arrays or inconsistent sample count")
    if not all(np.isfinite(a).all() for a in (points, xy, depths, reliability)):
        return {"accepted": False, "failures": ["nonfinite_geometry_or_support"], "metrics": {},
                "independent_accuracy_measured": False}
    left, right = base.load_rgb(inputs["left"]), base.load_rgb(inputs["right"])
    shape = left.shape[:2]
    lm = base.load_mask(inputs["thread_left_mask"], shape)
    rm = base.load_mask(inputs["thread_right_mask"], shape)
    depth = np.squeeze(np.load(inputs["depth_npy"], allow_pickle=False))
    K = base.load_intrinsics(inputs["K"])
    skel, graph, _, before, discarded = base.skeleton_graph(lm, 0., limits["segment_min_skeleton_px"])
    qp_support_metrics = {}
    if native_qp:
        # Label only strict stereo/depth input support. This is not ground truth,
        # and unsupported QP sections have unknown rather than verified join provenance.
        left_dist = cv2.distanceTransform((~lm).astype("uint8"), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        ld = cv2.remap(left_dist, xy[:, 0].astype("float32").reshape(1, -1),
                       xy[:, 1].astype("float32").reshape(1, -1), cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=1e6).reshape(-1)
        raw_depth, _ = base.sample_depth(depth, lm, xy[:, ::-1], 3,
                                        limits["min_depth_m"], limits["max_depth_m"])
        rd, _ = base.right_mask_distances(xy[:, ::-1], depths, rm, K[0, 0], inputs["baseline_m"])
        depth_agrees = (np.isfinite(raw_depth) & (raw_depth > 0) &
                        (np.abs(raw_depth - depths) <= limits["max_depth_relative_difference"] * raw_depth))
        supported = (ld <= 1.5) & (rd <= limits["right_tolerance_px"]) & depth_agrees
        inferred = ~supported
        native_observed = supported
        # Native QP may export default ones. Do not present them as confidence.
        reliability = np.zeros(n, dtype=float)
        if inferred.any():
            failures.append("unverified_qp_unsupported_geometry")
        if before > 1:
            failures.append("missing_native_join_provenance")
        qp_support_metrics = {"qp_input_supported_sample_count": int(supported.sum()),
                              "qp_unsupported_sample_count": int(inferred.sum()),
                              "qp_left_support_label_tolerance_px": 1.5}
    metrics = backend_report.get("metrics", {})
    after = before if native_qp else backend_report.get("component_count_after", metrics.get("skeleton_component_count_after_join"))
    if after != 1:
        failures.append("unresolved_fragments")
    joins = [] if native_qp else backend_report.get("joins", metrics.get("segment_joins", []))
    for join in joins:
        a, b = tuple(join["from_rc"]), tuple(join["to_rc"])
        if a not in graph or b not in graph or graph.degree[a] != 1 or graph.degree[b] != 1:
            failures.append("invalid_join_endpoints")
        else:
            graph.add_edge(a, b, weight=float(np.linalg.norm(np.asarray(a) - b)))
    terminal_count = sum(graph.degree[node] == 1 for node in graph)
    actual_components = base.nx.number_connected_components(graph)
    if terminal_count != 2:
        failures.append("ambiguous_thread_topology")
    if actual_components != after:
        failures.append("inconsistent_component_report")
    if inferred is None:
        inferred = np.zeros(n, bool)
        # Legacy bridges are straight edges; conservative projection labels do not change native output.
        for join in joins:
            a, b = np.asarray(join["from_rc"], float)[::-1], np.asarray(join["to_rc"], float)[::-1]
            delta = b - a
            if np.dot(delta, delta) <= 0:
                failures.append("degenerate_join")
                continue
            t = ((xy - a) @ delta) / (delta @ delta)
            distance = np.linalg.norm(xy - (a + t[:, None] * delta), axis=1)
            inferred |= (t > 1e-6) & (t < 1 - 1e-6) & (distance <= .75)
        provenance_method = "conservative_legacy_straight_bridge_projection"
    else:
        if inferred.shape != (n,) or inferred.dtype.kind != "b":
            raise ValueError("Invalid inferred provenance")
        if native_observed is None or native_observed.shape != (n,) or not np.array_equal(native_observed, ~inferred):
            failures.append("inconsistent_observed_inferred_labels")
        if np.any(reliability[inferred] != 0):
            failures.append("nonzero_inferred_reliability")
        provenance_method = ("qp_stereo_depth_input_support_unknown_gap_provenance"
                             if native_qp else "native_inferred_edge_labels")
    for join in joins:
        endpoints = np.asarray([join["from_rc"], join["to_rc"]], float)
        endpoint_depth, _ = base.sample_depth(depth, lm, endpoints, 5,
                                             limits["min_depth_m"], limits["max_depth_m"])
        if not np.isfinite(endpoint_depth).all():
            failures.append("missing_join_endpoint_depth")
            continue
        xyz = base.backproject(endpoints, endpoint_depth, K)
        if np.linalg.norm(xyz[1] - xyz[0]) > limits["max_join_gap_m"]:
            failures.append("metric_join_gap_exceeded")
        if abs(endpoint_depth[1] - endpoint_depth[0]) / min(endpoint_depth) > limits["max_depth_relative_difference"]:
            failures.append("incompatible_join_endpoint_depth")
    inside = ((xy[:, 0] >= 0) & (xy[:, 0] <= shape[1] - 1) &
              (xy[:, 1] >= 0) & (xy[:, 1] <= shape[0] - 1))
    if not inside.all():
        failures.append("curve_outside_image")
    if np.any((depths < limits["min_depth_m"]) | (depths > limits["max_depth_m"])):
        failures.append("depth_outside_physical_limits")
    if not np.allclose(base.backproject(xy[:, ::-1], depths, K), points, atol=1e-8, rtol=1e-5):
        failures.append("inconsistent_camera_frame_geometry")
    observed = ~inferred
    raw, _ = base.sample_depth(depth, lm, xy[:, ::-1], 3,
                               limits["min_depth_m"], limits["max_depth_m"])
    rd, right_xy = base.right_mask_distances(xy[:, ::-1], depths, rm, K[0, 0], inputs["baseline_m"])
    valid_fraction = float(np.isfinite(raw[observed]).mean()) if observed.any() else 0.
    right_fraction = float((rd[observed] <= limits["right_tolerance_px"]).mean()) if observed.any() else 0.
    raster = np.zeros(shape, "uint8")
    cv2.polylines(raster, [np.rint(xy).astype("int32")], False, 1, 1)
    distances = cv2.distanceTransform(1 - raster, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    coverage = float((distances[skel] <= limits["right_tolerance_px"]).mean())
    if coverage < limits["min_visible_coverage"]:
        failures.append("visible_fragments_not_fully_represented")
    if valid_fraction < limits["min_valid_visible_depth"]:
        failures.append("insufficient_visible_depth")
    if right_fraction < limits["min_visible_right_fraction"]:
        failures.append("visible_stereo_projection_failed")
    shared_reliability = reliability.copy()
    shared_reliability[inferred] = 0.
    np.savez(output / "shared_thread_samples.npz", points_m=points, points=points * 1000.,
             pixel_xy=xy, depth_m=depths, inferred=inferred, observed=observed,
             reliability=shared_reliability, provenance_method=np.array(provenance_method))
    write_curve_obj(output / "thread_centerline.obj", points, inferred)
    overlay = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
    for i in range(n - 1):
        color = (0, 165, 255) if inferred[i] or inferred[i + 1] else (0, 200, 0)
        cv2.line(overlay, tuple(np.rint(xy[i]).astype(int)), tuple(np.rint(xy[i + 1]).astype(int)), color, 2)
    cv2.imwrite(str(output / "shared_left_overlay.png"), overlay)
    return {
        "accepted": not failures, "failures": list(dict.fromkeys(failures)),
        "metrics": {"visible_skeleton_coverage": coverage, "valid_visible_depth_fraction": valid_fraction,
                    "visible_right_mask_fraction": right_fraction, "inferred_sample_count": int(inferred.sum()),
                    "sample_count": n, "component_count_before": before, "component_count_after": after,
                    "terminal_endpoint_count": terminal_count, "discarded_short_components": discarded,
                    **qp_support_metrics},
        "provenance_method": provenance_method, "independent_accuracy_measured": False,
        "geometry_is_estimated": True, "physics_validated": False, "obj_units": "meters",
        "obj_kind": "centerline_only", "observed_label_means": "visible curve support, not measured 3D ground truth",
        "artifacts_include_rejected_geometry": bool(failures),
        "native_join_provenance_available": not native_qp,
        "native_qp_reliability_used_as_confidence": False if native_qp else None,
    }


def verify_needle(inputs, output, tools, backend_report, limits):
    base = load_base(tools)
    path = Path(output) / "needle_points3d.csv"
    if not path.exists():
        return {"accepted": False, "failures": ["missing_needle_points"], "metrics": {},
                "independent_accuracy_measured": False}
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    points = np.asarray([[float(row[k]) for k in ("x", "y", "z")] for row in rows], float)
    failures = []
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 5 or not np.isfinite(points).all():
        return {"accepted": False, "failures": ["invalid_needle_points"], "metrics": {},
                "independent_accuracy_measured": False}
    z = points[:, 2]
    if np.any((z < limits["min_depth_m"]) | (z > limits["max_depth_m"])):
        failures.append("depth_outside_physical_limits")
    K = base.load_intrinsics(inputs["K"])
    left = base.load_rgb(inputs["left"])
    lm = base.load_mask(inputs["needle_left_mask"], left.shape[:2])
    rm = base.load_mask(inputs["needle_right_mask"], left.shape[:2])
    xy = np.column_stack([points[:, 0] * K[0, 0] / z + K[0, 2],
                          points[:, 1] * K[1, 1] / z + K[1, 2]])
    ld = cv2.distanceTransform((~lm).astype("uint8"), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    left_distance = cv2.remap(ld, xy[:, 0].astype("float32").reshape(1, -1),
                             xy[:, 1].astype("float32").reshape(1, -1),
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1e6).reshape(-1)
    rd, _ = base.right_mask_distances(xy[:, ::-1], z, rm, K[0, 0], inputs["baseline_m"])
    left_fraction = float((left_distance <= limits["right_tolerance_px"]).mean())
    right_fraction = float((rd <= limits["right_tolerance_px"]).mean())
    if left_fraction < limits["min_visible_right_fraction"] or right_fraction < limits["min_visible_right_fraction"]:
        failures.append("needle_stereo_projection_failed")
    return {"accepted": not failures, "failures": failures,
            "metrics": {"point_count": len(points), "left_mask_fraction": left_fraction,
                        "right_mask_fraction": right_fraction},
            "independent_accuracy_measured": False, "geometry_is_estimated": True,
            "needle_pose_estimated": False, "hidden_geometry_completed": False, "physics_validated": False}
