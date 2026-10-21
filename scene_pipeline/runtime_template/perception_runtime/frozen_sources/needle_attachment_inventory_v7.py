"""Read-only, stereo-bound attachment evidence for a specialist controller.

A visible arc endpoint is not automatically the physical needle swage. This
inventory preserves that distinction and never extends a needle, creates a
connection, changes source masks, or claims independent accuracy. An attachment
may be assessed on one reconstructed thread fragment before the rest is joined.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from compact_agent_protocol import digest
from complex_case_diagnostics import line_coverage
from newton_scene_adapter import centerline, project
from reconstruction_tools import fingerprints
from thread_visibility_repairs_v5 import (RepairLimits, at, distances,
    endpoint_table, in_view, load_inputs, load_parts)


MAX_CONNECTOR_LENGTH_M = .004  # Existing attachment tool's protected limit.
NEEDLE_VISIBLE_TOLERANCE_PX = 1.5  # Existing arc support-label tolerance.


def _flags(part, key, count):
    value = part.get(key)
    if value is None:
        return None
    value = np.asarray(value)
    if value.shape != (count,) or value.dtype != bool:
        raise ValueError('Invalid endpoint support labels: ' + key)
    return value


def _view_support(point, mask, occluder, K, baseline, shape, tolerance,
                  occluder_supplied):
    xy = project(np.asarray(point)[None], K, baseline)
    inside = bool(in_view(xy, shape)[0])
    distance = float(distances(mask, xy)[0]) if inside else None
    covered = bool(at(occluder, xy)[0]) if inside and occluder_supplied else None
    h, w = shape
    boundary = bool(inside and min(xy[0, 0], xy[0, 1], w-1-xy[0, 0],
                                  h-1-xy[0, 1]) <= 6.)
    supported = bool(inside and distance <= tolerance)
    return dict(xy=xy[0].tolist(), in_image=inside, mask_distance_px=distance,
                mask_supported=supported, near_image_boundary=boundary,
                occluder_supplied=occluder_supplied,
                marked_occluded=covered,
                clear_visible_support=bool(supported and covered is not True),
                occlusion_is_mask_evidence_not_ground_truth=True)


def _mask_terminals(base, mask):
    """Display all retained mask skeleton terminals, without assigning identity."""
    if not mask.any():
        return dict(endpoints_xy=[], retained_components=0,
                    discarded_short_components=0, junction_pixels=0)
    try:
        _, graph, _, count, discarded = base.skeleton_graph(mask, 0., 5)
    except RuntimeError as error:
        return dict(endpoints_xy=[], retained_components=0,
                    discarded_short_components=None, junction_pixels=0,
                    skeleton_inspection_failure=str(error),
                    physical_tip_or_swage_identity_known=False)
    ends = sorted(node for node in graph if graph.degree[node] == 1)
    return dict(endpoints_xy=[[int(x), int(y)] for y, x in ends],
                retained_components=int(count), discarded_short_components=int(discarded),
                junction_pixels=sum(graph.degree[node] > 2 for node in graph),
                physical_tip_or_swage_identity_known=False,
                skeleton_features_are_not_physical_endpoints=True)


def _endpoint_support(row, point, mask_kind, masks, occ, K, baseline, shape,
                      inputs, inferred, depth_inferred, source_support_known):
    tolerance = (NEEDLE_VISIBLE_TOLERANCE_PX if mask_kind == 'needle'
                 else RepairLimits().stereo_match_px)
    support = {side: _view_support(point, masks[side], occ[side], K,
               baseline if side == 'right' else 0., shape, tolerance,
               side+'_occluder' in inputs) for side in ('left', 'right')}
    blockers, warnings = [], []
    if not source_support_known:
        blockers.append('endpoint_source_support_labels_missing')
    if inferred is True:
        warnings.append('needle_endpoint_inferred' if mask_kind == 'needle'
                        else 'thread_endpoint_geometry_inferred')
    if depth_inferred is True:
        warnings.append('endpoint_depth_inferred')
    if not .03 <= float(point[2]) <= .5:
        blockers.append('endpoint_depth_outside_working_range')
    clear = [side for side in support if support[side]['clear_visible_support']]
    if not clear:
        blockers.append('endpoint_not_visible_in_either_view')
    if all(not s['in_image'] or s['near_image_boundary'] for s in support.values()):
        blockers.append('endpoint_identity_lost_at_frame_boundary')
    row.update(points_camera_m=np.asarray(point).tolist(),
               left_xy=support['left']['xy'], right_xy=support['right']['xy'],
               view_support=support, clear_support_views=clear,
               geometry_inferred=inferred, depth_inferred=depth_inferred,
               blockers=blockers, uncertainty_warnings=warnings,
               independent_accuracy_measured=False)
    return row


def build_attachment_inventory(case, pipeline, candidate, needle_path, out=None):
    """Return JSON-safe endpoints, all endpoint pairs, gates and full stereo.

    ``candidate`` is a shared reconstruction proposal with ``fragment_path``;
    it need not contain one completely connected thread. ``needle_path`` is the
    unchanged common free-radius needle arc. Gates reuse the 4 mm attachment
    bound, existing observation labels and camera/mask checks. The specialist
    still has to identify a *physical* swage in the images; numeric proximity is
    insufficient. ``out`` optionally creates labeled full-frame evidence files.
    """
    base, images, thread_masks, occ, depth, K, baseline = load_inputs(case, pipeline)
    inputs = case['inputs']
    shape = images['left'].shape[:2]
    if any(key not in inputs for key in ('needle_left_mask', 'needle_right_mask')):
        raise ValueError('Both unchanged needle masks required')
    needle_masks = {side: base.load_mask(inputs['needle_'+side+'_mask'], shape)
                    for side in ('left', 'right')}
    sources = dict(inputs)
    sources.update(inventory_source=str(Path(__file__).resolve()),
                   base_thread_source=str((Path(pipeline)/'reconstruct_thread_from_files.py').resolve()),
                   visibility_source=str(Path(__file__).with_name('thread_visibility_repairs_v5.py').resolve()))
    parts, needle = [], None
    blockers = []
    if candidate and candidate.get('fragment_path'):
        sources['thread_fragments'] = str(Path(candidate['fragment_path']).resolve())
        parts = load_parts(sources['thread_fragments'])
        for part in parts:
            centerline(part['points_m'])
    else:
        blockers.append('thread_reconstruction_unavailable')
    if needle_path is not None and Path(needle_path).is_file():
        sources['needle_arc'] = str(Path(needle_path).resolve())
        with np.load(needle_path, allow_pickle=False) as data:
            needle = {key: data[key].copy() for key in data.files}
        centerline(needle['points_m'])
    else:
        blockers.append('needle_reconstruction_unavailable')
    before = fingerprints(sources)
    thread_rows, needle_rows = [], []
    if parts:
        for old in endpoint_table(parts, K, baseline, shape):
            fragment, index = old['fragment'], (0 if old['end'] == 0 else len(parts[old['fragment']]['points_m'])-1)
            part = parts[fragment]
            geometry_flag = _flags(part, 'geometry_inferred', len(part['points_m']))
            depth_flag = _flags(part, 'depth_inferred', len(part['points_m']))
            row = dict(endpoint_id='thread.'+old['endpoint_id'], fragment=fragment,
                       end=old['end'], index=index,
                       outward_tangent=old['outward_tangent'],
                       physical_material_endpoint_confirmed=False)
            _endpoint_support(row, part['points_m'][index], 'thread', thread_masks,
                occ, K, baseline, shape, inputs,
                None if geometry_flag is None else bool(geometry_flag[index]),
                None if depth_flag is None else bool(depth_flag[index]),
                geometry_flag is not None and depth_flag is not None)
            thread_rows.append(row)
    if needle is not None:
        points = needle['points_m']
        flags = _flags(needle, 'inferred', len(points))
        visible = _flags(needle, 'visible_stereo_support', len(points))
        if flags is not None and visible is not None and not np.array_equal(visible, ~flags):
            raise ValueError('Inconsistent unchanged needle support labels')
        for end, index in ((0, 0), (1, len(points)-1)):
            row = dict(endpoint_id=f'needle.e{end}', end=end, index=index,
                       observed_arc_endpoint=True,
                       physical_attachment_endpoint_confirmed=False,
                       physical_role='unknown_tip_swage_or_occlusion_cut')
            _endpoint_support(row, points[index], 'needle', needle_masks, occ,
                K, baseline, shape, inputs,
                None if flags is None else bool(flags[index]), None,
                flags is not None and visible is not None)
            needle_rows.append(row)
    rows = []
    for t in thread_rows:
        for n in needle_rows:
            delta = np.asarray(t['points_camera_m'])-np.asarray(n['points_camera_m'])
            gap = float(np.linalg.norm(delta))
            failures = list(dict.fromkeys(t['blockers']+n['blockers']))
            if gap > MAX_CONNECTOR_LENGTH_M:
                failures.append('connector_exceeds_existing_4mm_limit')
            depth_difference = float(abs(delta[2]))
            relative_difference = depth_difference/min(t['points_camera_m'][2], n['points_camera_m'][2])
            if relative_difference > RepairLimits().max_depth_relative_difference:
                failures.append('endpoint_depth_disagreement')
            views = {}
            connector = np.linspace(t['points_camera_m'], n['points_camera_m'], 33)
            for side in ('left', 'right'):
                a, b = t[side+'_xy'], n[side+'_xy']
                xy = project(connector, K, baseline if side == 'right' else 0.)
                # This is input consistency, not an independent accuracy test.
                supported = distances(thread_masks[side] | needle_masks[side], xy) <= RepairLimits().mask_check_px
                explained = supported | at(occ[side], xy)
                views[side] = dict(projected_gap_px=float(np.linalg.norm(np.asarray(a)-b)),
                    connector_explained_fraction=float(explained.mean()),
                    endpoints_mask_supported=bool(t['view_support'][side]['mask_supported']
                                                 and n['view_support'][side]['mask_supported']),
                    endpoints_clear=bool(t['view_support'][side]['clear_visible_support']
                                         and n['view_support'][side]['clear_visible_support']),
                    **line_coverage(occ[side], a, b),
                    occluder_supplied=side+'_occluder' in inputs)
            if all(v['connector_explained_fraction'] < RepairLimits().min_explained_fraction for v in views.values()):
                failures.append('connector_crosses_unsupported_background_in_both_views')
            tangent = np.asarray(t['outward_tangent'])
            needle_points = needle['points_m']
            ni = n['index']
            neighbor = min(4, len(needle_points)-1) if ni == 0 else max(0, ni-4)
            needle_outward = needle_points[ni]-needle_points[neighbor]
            direction = delta / max(gap, 1e-12)  # Needle -> thread.
            warnings = list(dict.fromkeys(t['uncertainty_warnings']+n['uncertainty_warnings']))
            rows.append(dict(candidate_id=f"attach.f{t['fragment']}.e{t['end']}.n{n['end']}",
                thread_endpoint_id=t['endpoint_id'], needle_endpoint_id=n['endpoint_id'],
                fragment=t['fragment'], thread_endpoint_index=t['index'], needle_endpoint_index=n['index'],
                distance_m=gap, delta_camera_m=delta.tolist(), depth_gap_m=depth_difference,
                relative_depth_gap=relative_difference, views=views,
                max_connector_length_m=MAX_CONNECTOR_LENGTH_M, blockers=failures,
                uncertainty_warnings=warnings, inferred_endpoint_present=bool(warnings),
                direction_diagnostics=dict(thread_outward_toward_needle_cosine=float(np.dot(tangent, -direction)),
                    needle_outward_toward_thread_cosine=float(np.dot(needle_outward, direction)/max(np.linalg.norm(needle_outward),1e-12)),
                    diagnostic_only=True, short_gap_direction_may_be_noisy=True),
                physical_attachment_identity_requires_visual_assessment=True,
                semantic_confirmation_pending=True, hypothesis_not_ground_truth=True))
    # Preserve the existing tool's uniqueness gate across *all* near pairs,
    # including unsupported contenders. A VLM must not silently resolve numeric
    # ambiguity by simply choosing the nearest displayed endpoint.
    near = [row for row in rows if row['distance_m'] <= MAX_CONNECTOR_LENGTH_M]
    for row in rows:
        if len(near) != 1 and row in near:
            row['blockers'].append('multiple_endpoint_pairs_within_existing_bound')
        row['numerical_checks_passed'] = not row['blockers']
        row['eligible_for_specialist_selection'] = row['numerical_checks_passed']
    if rows and not near:
        blockers.append('no_endpoint_pair_within_existing_4mm_limit')
    elif len(near) > 1:
        blockers.append('ambiguous_near_endpoint_pairs')
    elif near and not near[0]['numerical_checks_passed']:
        blockers.extend(near[0]['blockers'])
    eligible = [r['candidate_id'] for r in rows if r['eligible_for_specialist_selection']]
    images_info = {side: {'path': before[side]['path'], 'sha256': before[side]['sha256']}
                   for side in ('left', 'right')}
    report = dict(schema='needle_attachment_inventory_v9_inferred', case_id=case['case_id'],
        baseline_m=float(baseline), K=K.tolist(),
        images=images_info, image_shape=list(shape), endpoints={'thread': thread_rows, 'needle': needle_rows},
        candidates=rows,
        tasks=[dict(task_id='needle_attachment', candidate_ids=[r['candidate_id'] for r in rows],
                    eligible_candidate_ids=eligible, status='requires_visual_identity_assessment' if eligible else 'blocked',
                    blockers=list(dict.fromkeys(blockers)),
                    whole_thread_connectivity_required=False)],
        needle_mask_boundaries={side: _mask_terminals(base, needle_masks[side]) for side in ('left', 'right')},
        source_fingerprints=before, evidence_ids=['attachment.inventory', 'image.left', 'image.right'],
        attachment_applied=False, geometry_modified=False, hidden_needle_extended=False,
        independent_accuracy_measured=False, physics_validated=False, training_eligible=False,
        calibration_status=case.get('calibration_id', 'declared_scale_not_independently_verified'),
        limitations=['Endpoints and depths are estimates, not ground truth.',
                     'Visible arc endpoints can be a tip, swage, or occlusion cut.',
                     'A nearby endpoint does not establish the physical attachment.',
                     'A tool mask supplies occlusion evidence; it is not measured visibility.',
                     'Inferred endpoints may support an explicitly inferred attachment after visual assessment; no needle extension is allowed.',
                     'Missing support, conflicting images or ambiguous endpoint pairs must defer.'])
    report['scene_version'] = digest(report)
    if out is not None:
        target = Path(out)
        target.mkdir(parents=True, exist_ok=False)
        report['labeled_images'] = {}
        for side in ('left', 'right'):
            image = cv2.cvtColor(images[side], cv2.COLOR_RGB2BGR)
            for row in thread_rows+needle_rows:
                xy = np.rint(row[side+'_xy']).astype(int)
                if not row['view_support'][side]['in_image']:
                    continue
                needle_endpoint = row['endpoint_id'].startswith('needle')
                color = (0, 220, 255) if needle_endpoint else (70, 220, 70)
                cv2.circle(image, tuple(xy), 6, color, 2)
                label = row['endpoint_id'].replace('thread.', 't.').replace('needle.', 'n.')
                text_xy = (int(np.clip(xy[0]+7, 0, shape[1]-120)),
                           int(np.clip(xy[1]+(-13 if needle_endpoint else 21), 16, shape[0]-4)))
                cv2.putText(image, label, text_xy, cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3)
                cv2.putText(image, label, text_xy, cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
            path = target/(side+'_attachment_endpoints.png')
            if not cv2.imwrite(str(path), image):
                raise RuntimeError('Could not write attachment endpoint evidence')
            report['labeled_images'][side] = str(path.resolve())
        (target/'inventory.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if fingerprints(sources) != before:
        raise ValueError('Attachment evidence inputs changed during inspection')
    return report


def select_attachment_candidate(inventory, candidate_id, *,
                                physical_attachment_endpoint_confirmed=False,
                                evidence_ids=()):
    """Bound specialist selection before the unchanged attachment constructor.

    ``physical_attachment_endpoint_confirmed`` records the specialist's image
    assertion, not measured truth. Both stereo evidence IDs are required, even
    when only one view clearly exposes the endpoint. Numeric and source-support
    gates remain authoritative; proximity alone cannot authorize an attachment.
    """
    bound = {key: value for key, value in inventory.items()
             if key not in ('scene_version', 'labeled_images')}
    if inventory.get('scene_version') != digest(bound):
        raise ValueError('Attachment inventory changed; inspect current evidence again')
    expected = inventory['source_fingerprints']
    if fingerprints({key: row['path'] for key, row in expected.items()}) != expected:
        raise ValueError('Attachment source inputs changed; inspect current evidence again')
    if physical_attachment_endpoint_confirmed is not True:
        raise ValueError('Physical swage identity requires explicit visual assessment')
    if not {'image.left', 'image.right'} <= set(evidence_ids):
        raise ValueError('Attachment identity assessment requires both stereo views')
    selected = [r for r in inventory['candidates'] if r['candidate_id'] == candidate_id]
    if (len(selected) != 1 or not selected[0]['numerical_checks_passed']
            or selected[0]['blockers']
            or selected[0]['max_connector_length_m'] != MAX_CONNECTOR_LENGTH_M
            or not np.isfinite(selected[0]['distance_m'])
            or not 0. <= selected[0]['distance_m'] <= MAX_CONNECTOR_LENGTH_M):
        raise ValueError('Unknown, ambiguous, unsupported, or out-of-bound attachment candidate')
    return dict(selected[0], specialist_asserted_physical_swage=True,
                inferred_connection_authorized=bool(selected[0].get('inferred_endpoint_present')),
                visual_evidence_ids=list(evidence_ids),
                semantic_confirmation_pending=False, hypothesis_not_ground_truth=True)
