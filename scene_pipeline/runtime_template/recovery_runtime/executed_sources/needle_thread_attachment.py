"""Evidence-gated topology proposal using the existing attachment selector.

The VLM relation is a hypothesis, not measured truth. This module neither edits
source geometry nor sets joint/material parameters. The caller must inspect the
applied joint's actual endpoint error and retain physics failures.
"""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def existing_selector(source):
    tree = ast.parse(Path(source).read_text())
    names = {'load_needle_thread_relation', 'evaluate_needle_thread_attachment'}
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in selected} != names:
        raise ValueError('Existing attachment selector unavailable')
    namespace = {'np': np, 'json': json, 'Path': Path}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['load_needle_thread_relation'], namespace['evaluate_needle_thread_attachment']


def propose_attachment(geometry, relation_path, selector_source, expected_case='frame_000787'):
    load, select = existing_selector(selector_source)
    relation = load(relation_path)
    raw = json.loads(Path(relation_path).read_text())
    images = {side: raw.get('source_' + side, '') for side in ('left', 'right')}
    if not all(isinstance(p, str) and expected_case in p for p in images.values()):
        raise ValueError('VLM report does not identify both images of the selected case')
    t, n = geometry['points_newton_m'], geometry['needle_points_newton_m']
    if any(p.ndim != 2 or p.shape[1] != 3 or len(p) < 3 or not np.isfinite(p).all() for p in (t, n)):
        raise ValueError('Require finite open thread and needle centerlines')
    candidates = sorted([{'thread_endpoint_index': ti, 'needle_endpoint_index': ni,
                          'distance_m': float(np.linalg.norm(t[ti]-n[ni]))}
                         for ti in (0, len(t)-1) for ni in (0, len(n)-1)], key=lambda c: c['distance_m'])
    # Preserve the existing 4 mm / .75 selector settings. Model confidence is
    # uncalibrated; endpoint uniqueness and stereo-support labels add safeguards.
    report = select(relation, t, n, .75, .004)
    if sum(c['distance_m'] <= .004 for c in candidates) != 1:
        report['failed_checks'].append('missing_or_ambiguous_endpoint_pair')
    ti, ni = report['thread_endpoint_index'], report['needle_endpoint_index']
    for key, i in (('inferred', ti), ('needle_inferred', ni)):
        flags = geometry.get(key)
        count = len(t) if key == 'inferred' else len(n)
        if flags is None or flags.shape != (count,) or flags.dtype != bool or flags[i]:
            report['failed_checks'].append('attachment_endpoint_lacks_visible_input_support')
    report['applied'] = bool(report['applied'] and not report['failed_checks'])
    report.update(kind='inferred_needle_thread_topology', joint_type='ball', candidates=candidates,
                  geometry_modified=False, attachment_ground_truth_measured=False,
                  vlm_confidence_calibrated=False, source_images_cryptographically_bound=False,
                  input_relation_sha256=sha256(relation_path), selector_source_sha256=sha256(selector_source),
                  input_image_names=images,
                  limitations=['Historical VLM image names match case; no recorded image SHA binding',
                               'Endpoint support is input consistency, not independent attachment evidence',
                               'Point joint approximates swage; no calibrated angular compliance',
                               'Prototype wire radii/material parameters remain uncalibrated'])
    return report


def world_endpoint_error(native, poses, rods, needle_body, thread_points, needle_points, proposal,
                         thread_reversed=None, needle_local_origin=None):
    """Measure material endpoints in the same world frame, including reversal."""
    if thread_reversed is None:
        thread_reversed = proposal['thread_endpoint_index'] == 0
    model_points = thread_points[::-1] if thread_reversed else thread_points
    selected = len(model_points)-1-proposal['thread_endpoint_index'] if thread_reversed else proposal['thread_endpoint_index']
    if selected == 0:
        parent = poses[int(rods[0])]
        local_anchor = np.zeros(3)
    elif selected == len(model_points)-1:
        parent = poses[int(rods[-1])]
        local_anchor = np.array([0., 0., float(np.linalg.norm(model_points[-1]-model_points[-2]))])
    else:
        raise ValueError('Attachment measurement requires a terminal endpoint')
    child = poses[int(needle_body)]
    parent_anchor = parent[:3] + native.quat_xyzw_to_matrix(parent[3:7]) @ local_anchor
    origin = needle_points.mean(axis=0) if needle_local_origin is None else np.asarray(needle_local_origin)
    needle_local = needle_points[proposal['needle_endpoint_index']] - origin
    child_anchor = child[:3] + native.quat_xyzw_to_matrix(child[3:7]) @ needle_local
    return float(np.linalg.norm(parent_anchor-child_anchor))
