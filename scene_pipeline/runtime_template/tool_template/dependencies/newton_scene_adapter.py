"""Strict reconstruction-to-Newton handoff. No accuracy or training approval implied."""
import hashlib
import json
from pathlib import Path

import numpy as np


CAMERA_TO_NEWTON = np.array([[1., 0, 0, 0], [0, 0, 1., 0],
                             [0, -1., 0, 0], [0, 0, 0, 1.]])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rigid_transform(value):
    t = np.asarray(value, dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-5)):
        raise ValueError("Expected an SE(3) transform; scaling/reflection forbidden")
    return t


def transform(points, matrix):
    t = rigid_transform(matrix)
    return np.asarray(points) @ t[:3, :3].T + t[:3, 3]


def centerline(value):
    p = np.asarray(value, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3 or not np.isfinite(p).all():
        raise ValueError("Expected at least three finite 3D centerline points")
    if np.any(np.linalg.norm(np.diff(p, axis=0), axis=1) <= 1e-9):
        raise ValueError("Duplicate/zero-length centerline segments")
    return p


def project(points_camera, K, baseline_m=0.):
    p = np.asarray(points_camera).copy()
    p[:, 0] -= baseline_m
    if np.any(p[:, 2] <= 0):
        raise ValueError("Points behind camera")
    h = p @ np.asarray(K).T
    return h[:, :2] / h[:, 2:3]


def read_obj_object(path, name):
    """Extract only a named native OBJ object; never import old scene states."""
    vertices, faces, active = [], [], False
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == 'o':
            active = ' '.join(fields[1:]) == name
        elif fields[0] == 'v':
            vertices.append([float(x) for x in fields[1:4]])
        elif fields[0] == 'f' and active:
            if len(fields) != 4:
                raise ValueError("Native tool OBJ must contain triangles")
            indices = [int(x.split('/')[0]) for x in fields[1:]]
            faces.append([i - 1 if i > 0 else len(vertices) + i for i in indices])
    if not faces:
        raise ValueError("Requested tool object missing/empty")
    faces = np.array(faces, dtype=np.int32)
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Invalid OBJ indices")
    used, remapped = np.unique(faces, return_inverse=True)
    verts = np.array(vertices, dtype=float)[used]
    if not np.isfinite(verts).all():
        raise ValueError("Nonfinite tool geometry")
    return verts, remapped.reshape(-1, 3).astype(np.int32)


def assemble(contract, base_dir):
    """Inputs: explicit SI contract and accepted shared-tool artifacts.

    Outputs: camera/Newton geometry, labels, calibrated projections, provenance.
    Acceptance: units, frame, SE(3), scale, finite geometry and source hashes.
    These are consistency checks; they do not measure independent accuracy.
    """
    base = Path(base_dir)
    if contract.get('units') != 'm' or contract.get('frame') != 'rectified_left_camera':
        raise ValueError("Require explicit meters and rectified_left_camera frame")
    if contract.get('split') not in ('development', 'train', 'validation'):
        raise ValueError("Test scenes cannot be used for development")
    inputs = contract['inputs']
    paths = {k: (base / v).resolve() for k, v in inputs.items()}
    expected = contract['input_sha256']
    if set(expected) != set(paths) or any(sha256(p) != expected[k] for k, p in paths.items()):
        raise ValueError("Input fingerprint mismatch")
    verdict = json.loads(paths['thread_result'].read_text())
    if verdict.get('accepted') is not True or verdict.get('verification', {}).get('accepted') is not True:
        raise ValueError("Rejected reconstruction cannot enter adapter")
    if verdict.get('tool') not in {'thread_qp', 'thread_distance', 'thread_occlusion'}:
        raise ValueError('Require a shared-verified thread operation')
    artifact_key = str(verdict.get('call_id', '')) + '/backend/shared_thread_samples.npz'
    if verdict.get('artifact_sha256', {}).get(artifact_key) != sha256(paths['thread_samples']):
        raise ValueError('Thread geometry is not the exact shared-verified artifact')
    with np.load(paths['thread_samples'], allow_pickle=False) as a:
        p = centerline(a['points_m'])
        inferred = a['inferred'].copy()
        observed = a['observed'].copy()
        reliability = a['reliability'].copy()
    if (inferred.dtype != bool or observed.dtype != bool or inferred.shape != (len(p),)
            or observed.shape != inferred.shape or reliability.shape != inferred.shape
            or not np.array_equal(observed, ~inferred) or not np.isfinite(reliability).all()
            or np.any((reliability < 0) | (reliability > 1)) or np.any(reliability[inferred] != 0)):
        raise ValueError("Invalid observation/inference labels")
    if np.any((p[:, 2] < .03) | (p[:, 2] > .5)):
        raise ValueError("Thread outside declared 0.03–0.5 m working depth")
    K = np.loadtxt(paths['K'], delimiter=',')
    if (K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0
            or not np.allclose(K[2], [0, 0, 1]) or K[1, 0] != 0):
        raise ValueError("Invalid intrinsics")
    baseline = float(contract['baseline_m'])
    if not np.isfinite(baseline) or not 0 < baseline < .1:
        raise ValueError("Invalid stereo baseline")
    tool_camera, tool_faces = read_obj_object(paths['tool_obj'], 'tracked_tool')
    # The shaft may extend to/behind the image plane; only the distal frame
    # must be within working depth. Do not silently crop native shaft geometry.
    if (np.abs(tool_camera).max() > 1. or not .001 <= np.ptp(tool_camera, axis=0).max() <= 1.):
        raise ValueError("Tool mesh scale/frame inconsistent with meter camera geometry")
    with np.load(paths['tracker_result'], allow_pickle=False) as a:
        tool_pose = rigid_transform(a['frame4_to_camera'])
        distal = a['distal_joints_rad'].copy()
    if not .03 <= tool_pose[2, 3] <= .5:
        raise ValueError("Tool distal pose outside working depth")
    if distal.shape != (4,) or not np.isfinite(distal).all():
        raise ValueError("Invalid fitted distal joints")
    world = transform(p, CAMERA_TO_NEWTON)
    recovered = transform(world, np.linalg.inv(CAMERA_TO_NEWTON))
    needle = contract.get('needle')
    needle_camera = np.empty((0, 3))
    needle_inferred = np.empty(0, dtype=bool)
    if needle is not None:
        evidence = needle.get('evidence')
        if evidence == 'shared_verified_free_radius_arc':
            if not {'needle_result', 'needle_samples'} <= set(paths):
                raise ValueError('Estimated needle requires fingerprinted shared result and geometry')
            result = json.loads(paths['needle_result'].read_text())
            check = result.get('verification', {})
            if (result.get('accepted') is not True or result.get('tool') != 'needle_arc' or
                    check.get('accepted') is not True or check.get('fixed_radius_prior') is not False or
                    check.get('independent_accuracy_measured') is not False):
                raise ValueError('Rejected/unverified needle arc cannot enter adapter')
            artifact_key = result['call_id'] + '/backend/shared_needle_arc.npz'
            if result.get('artifact_sha256', {}).get(artifact_key) != sha256(paths['needle_samples']):
                raise ValueError('Needle geometry is not the exact shared-verified artifact')
            with np.load(paths['needle_samples'], allow_pickle=False) as a:
                needle_camera = centerline(a['points_m'])
                needle_inferred = a['inferred'].copy()
                reliability_needle = a['reliability'].copy()
                radius = float(a['tube_radius_m'])
            if (needle_inferred.dtype != bool or needle_inferred.shape != (len(needle_camera),) or
                    reliability_needle.shape != needle_inferred.shape or
                    not np.isfinite(reliability_needle).all() or
                    np.any((reliability_needle < 0) | (reliability_needle > 1)) or
                    np.any(reliability_needle[needle_inferred] != 0)):
                raise ValueError('Invalid estimated needle support labels')
        elif evidence in ('synthetic_fixture', 'measured_fixed_template'):
            local = centerline(needle['local_points_m'])
            if not .001 <= np.ptp(local, axis=0).max() <= .1:
                raise ValueError("Invalid fixed needle template scale")
            radius = float(needle['tube_radius_m'])
            needle_camera = transform(local, needle['T_camera_from_needle'])
            needle_inferred = np.ones(len(needle_camera), dtype=bool)
        else:
            raise ValueError('Require explicit fixture evidence or a shared-verified estimated arc')
        if not np.isfinite(radius) or not 1e-5 <= radius <= .005:
            raise ValueError("Invalid explicit needle tube radius")
        if np.any((needle_camera[:, 2] < .03) | (needle_camera[:, 2] > .5)):
            raise ValueError("Needle outside working depth")
    arrays = dict(points_camera_m=p, points_newton_m=world, inferred=inferred,
                  observed=observed, reliability=reliability, K=K,
                  left_pixel_xy=project(p, K), right_pixel_xy=project(p, K, baseline),
                  tool_vertices_camera_m=tool_camera, tool_vertices_newton_m=transform(tool_camera, CAMERA_TO_NEWTON),
                  tool_faces=tool_faces, T_camera_from_tracker_frame4=tool_pose,
                  T_newton_from_tracker_frame4=CAMERA_TO_NEWTON @ tool_pose,
                  fitted_distal_joints_rad=distal,
                  needle_points_camera_m=needle_camera,
                  needle_points_newton_m=transform(needle_camera, CAMERA_TO_NEWTON),
                  needle_inferred=needle_inferred,
                  T_newton_from_camera=CAMERA_TO_NEWTON)
    if needle is not None:
        arrays['needle_tube_radius_m'] = np.asarray(radius)
    report = dict(schema='ismr_newton_handoff_v1', scene_id=contract['scene_id'],
                  units='m', frame='newton_z_up', input_sha256=expected,
                  sample_count=len(p), inferred_count=int(inferred.sum()),
                  thread_arclength_m=float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()),
                  tool_triangles=len(tool_faces), needle_present=needle is not None,
                  tool_mode='static_newest_native_tracked_mesh',
                  transform_roundtrip_max_m=float(np.abs(p - recovered).max()),
                  reprojection_roundtrip_max_px=float(np.abs(project(p, K) - project(recovered, K)).max()),
                  independent_accuracy_measured=False, physics_validated=False,
                  training_eligible=False,
                  limitations=['Tool articulation/action mapping unverified', 'Physical parameters uncalibrated',
                               'No support surface imported', 'No synchronized action trajectory',
                               'Visible support and inferred geometry are estimates'],
                  needle_evidence=None if needle is None else needle['evidence'])
    if needle is not None and needle['evidence'] == 'shared_verified_free_radius_arc':
        report.update(needle_geometry_estimated=True, needle_inferred_count=int(needle_inferred.sum()),
                      needle_tube_radius_is_unmeasured_assumption=True)
        report['limitations'].append('Free-radius needle fit has no independent accuracy or size verification')
    if needle is None:
        report['limitations'].append('No independently sized/posed needle imported')
    return arrays, report


def physics_verdict(states, log, max_displacement_m=.02):
    """Fail even when a native simulator silently recovers or exits zero."""
    a = np.asarray(states)
    failures = []
    if a.ndim != 3 or a.shape[2] != 3 or not len(a) or not np.isfinite(a).all():
        failures.append('nonfinite_or_missing_states')
        drift = None
    else:
        drift = float(np.linalg.norm(a - a[0], axis=2).max())
        if drift > max_displacement_m:
            failures.append('static_probe_displacement_limit')
    lower = log.lower()
    for phrase, failure in [('overflow', 'contact_overflow'), ('non-finite', 'native_nonfinite_recovery'),
                            ('nonfinite', 'native_nonfinite_recovery')]:
        if phrase in lower and failure not in failures:
            failures.append(failure)
    return dict(passed=not failures, failures=failures, max_displacement_m=drift,
                displacement_limit_m=max_displacement_m, training_eligible=False,
                independent_accuracy_measured=False)
