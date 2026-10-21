"""Read-only endpoint and gap inventory for the shared reconstruction tools.

An image endpoint is an assessment unit, not proof of a physical disconnect:
it may be a true terminal, a tool occlusion, a border exit, or a mask error.
Every visible component is retained, including short/isolated mask features.
Numerical pairing checks reuse the protected v5 tools without changing limits.
"""
from dataclasses import asdict
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from complex_case_diagnostics import project, write_json
from reconstruction_tools import fingerprints
from thread_visibility_repairs_v5 import (RepairLimits, at, distances, endpoint_table,
    in_view, join_proposals, load_inputs, load_parts)


def _id(prefix, value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return prefix + '.' + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _graph(mask):
    """Retain all mask skeleton pixels, but remove redundant diagonal shortcuts.

    Only graph edges change. No source pixels or short terminal spurs disappear.
    The usual 8-neighbor graph can turn a right-angle pixel path into a triangle;
    removing its diagonal restores path adjacency without inventing a branch.
    """
    sk = skeletonize(mask)
    nodes = [tuple(map(int, rc)) for rc in np.argwhere(sk)]
    graph = nx.Graph(); graph.add_nodes_from(nodes)
    present = set(nodes)
    for a in nodes:
        for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
            b = (a[0] + dy, a[1] + dx)
            if b in present:
                graph.add_edge(a, b, weight=float(np.hypot(dy, dx)))
    removed = 0
    for a, b in list(graph.edges):
        if abs(a[0] - b[0]) == abs(a[1] - b[1]) == 1:
            if any(n in graph and graph.has_edge(a, n) and graph.has_edge(n, b)
                   for n in ((a[0], b[1]), (b[0], a[1]))):
                graph.remove_edge(a, b); removed += 1
    return sk, graph, removed


def _boundary(xy, shape, margin=6.):
    h, w = shape
    return bool(min(xy[0], xy[1], w - 1 - xy[0], h - 1 - xy[1]) <= margin)


def _tangent(graph, start, distance=15.):
    if graph.degree[start] != 1:
        return None
    path = [start]; length = 0.
    while length < distance:
        nxt = [n for n in graph.neighbors(path[-1]) if len(path) == 1 or n != path[-2]]
        if len(nxt) != 1: break
        node = nxt[0]; length += graph[path[-1]][node]['weight']; path.append(node)
        if graph.degree[node] != 2: break
    vector = np.asarray(start, float) - np.asarray(path[-1], float)
    norm = np.linalg.norm(vector)
    return (vector[::-1] / norm).tolist() if norm > 0 else None


def _view_inventory(mask, occluder, needle, side):
    skeleton, graph, diagonals = _graph(mask)
    rows = []; components = []; node_component = {}
    for nodes in sorted(nx.connected_components(graph), key=lambda n: min(n)):
        component_id = _id(side + '.component', sorted(nodes))
        node_component.update({n: component_id for n in nodes})
        sub = graph.subgraph(nodes)
        ends = sorted(n for n in sub if sub.degree[n] <= 1)
        branches = sum(sub.degree[n] > 2 for n in sub)
        components.append(dict(component_id=component_id, skeleton_pixels=len(nodes),
            endpoints=len(ends), junction_pixels=branches,
            topology='isolated' if len(nodes) == 1 else ('branched' if branches else ('loop' if not ends else 'path')),
            below_existing_reconstruction_minimum=len(nodes) < 30))
        for node in ends:
            xy = np.asarray(node[::-1], float)
            tool_distance = float(distances(occluder, xy[None])[0]) if occluder.any() else None
            needle_distance = float(distances(needle, xy[None])[0]) if needle.any() else None
            tags = []
            if _boundary(xy, mask.shape): tags.append('border_exit_candidate')
            if tool_distance is not None and tool_distance <= 6: tags.append('tool_occlusion_candidate')
            if needle_distance is not None and needle_distance <= 8: tags.append('needle_junction_candidate')
            if len(nodes) < 30: tags.append('short_mask_feature_uncertain')
            if not tags: tags.append('visible_termination_unclassified')
            endpoint_id = f'{side}.endpoint.r{node[0]}.c{node[1]}'
            rows.append(dict(endpoint_id=endpoint_id, view=side, component_id=component_id,
                xy=xy.tolist(), outward_tangent_xy=_tangent(sub, node),
                category=tags[0], category_candidates=tags,
                near_boundary=_boundary(xy, mask.shape), tool_mask_distance_px=tool_distance,
                needle_mask_distance_px=needle_distance, is_physical_terminal_known=False,
                stereo_evidence=None))
    return rows, components, graph, node_component, dict(
        source_skeleton_pixels=int(skeleton.sum()), inventoried_skeleton_pixels=len(graph),
        discarded_short_components=0, redundant_diagonal_edges_removed=diagonals,
        source_pixels_removed=0)


def _opposite_path(a, b, graphs, components, limits):
    """Opposite-view mask connectivity is image evidence, not recovered depth."""
    side = a['view']; other = 'right' if side == 'left' else 'left'
    ea, eb = a['stereo_evidence'], b['stereo_evidence']
    result = dict(opposite_view=other, available=False, continuous_visible_mask_path=False,
        hidden_depth_recovered=False)
    if not ea or not eb or ea.get('opposite_nearest_rc') is None or eb.get('opposite_nearest_rc') is None:
        result['reason'] = 'stereo_endpoint_projection_unavailable'; return result
    ra, rb = tuple(ea['opposite_nearest_rc']), tuple(eb['opposite_nearest_rc'])
    result['anchor_distances_px'] = [ea['opposite_skeleton_distance_px'], eb['opposite_skeleton_distance_px']]
    if max(result['anchor_distances_px']) > limits.stereo_match_px:
        result['reason'] = 'projected_endpoint_lacks_opposite_mask_support'; return result
    result['available'] = True
    if components[other].get(ra) != components[other].get(rb):
        result['reason'] = 'opposite_endpoints_in_different_mask_components'; return result
    path = nx.shortest_path(graphs[other], ra, rb, weight='weight')
    result.update(continuous_visible_mask_path=True, opposite_component_id=components[other][ra],
        path_length_px=float(sum(graphs[other][x][y]['weight'] for x, y in zip(path, path[1:]))),
        path_junction_pixels=sum(graphs[other].degree[n] > 2 for n in path),
        continuous_path_is_not_unique_topology=any(graphs[other].degree[n] > 2 for n in path))
    return result


def _stereo_endpoint_evidence(endpoints, graphs, masks, depth, K, baseline, base, limits):
    left_rc = np.asarray(sorted(graphs['left']), float).reshape(-1, 2)
    if len(left_rc):
        z, _ = base.sample_depth(depth, masks['left'], left_rc, 3, limits.min_depth_m, limits.max_depth_m)
        good = np.isfinite(z); left_rc, z = left_rc[good], z[good]
    else: z = np.zeros(0, float)
    points = base.backproject(left_rc, z, K) if len(z) else np.empty((0, 3))
    right_anchors = project(points, K, baseline) if len(points) else np.empty((0, 2))
    anchor_tree = cKDTree(right_anchors) if len(right_anchors) else None
    trees = {}; rc_nodes = {}
    for side in ('left', 'right'):
        rc_nodes[side] = np.asarray(sorted(graphs[side]), int).reshape(-1, 2)
        trees[side] = cKDTree(rc_nodes[side][:, ::-1]) if len(rc_nodes[side]) else None
    for endpoint in endpoints:
        side = endpoint['view']; other = 'right' if side == 'left' else 'left'
        xy = np.asarray(endpoint['xy'], float); point = None
        row = dict(estimated_depth_m=None, depth_is_independent_measurement=False,
            depth_source=None, depth_anchor_ambiguous=False, opposite_xy=None,
            opposite_skeleton_distance_px=None, opposite_nearest_rc=None,
            opposite_image_supported=False, rectified_epipolar_rows=True)
        if side == 'left':
            zz, _ = base.sample_depth(depth, masks['left'], xy[::-1][None], 3,
                limits.min_depth_m, limits.max_depth_m)
            if np.isfinite(zz[0]):
                point = base.backproject(xy[::-1][None], zz, K)[0]
                row['depth_source'] = 'existing_left_depth_near_visible_endpoint'
        elif anchor_tree:
            indexes = anchor_tree.query_ball_point(xy, 4.)
            if indexes:
                values = points[indexes, 2]
                relative = float((values.max() - values.min()) / np.median(values))
                row['depth_anchor_ambiguous'] = relative > limits.max_depth_relative_difference
                row['left_depth_anchor_count'] = len(indexes)
                row['left_depth_anchor_relative_range'] = relative
                if not row['depth_anchor_ambiguous']:
                    zz = float(np.median(values)); left_xy = xy.copy(); left_xy[0] += K[0, 0] * baseline / zz
                    point = base.backproject(left_xy[::-1][None], np.asarray([zz]), K)[0]
                    row['depth_source'] = 'left_depth_projected_to_right_endpoint'
        if point is not None:
            row['estimated_depth_m'] = float(point[2])
            opposite = project(point[None], K, baseline if other == 'right' else 0.)[0]
            row['opposite_xy'] = opposite.tolist()
            if trees[other]:
                distance, index = trees[other].query(opposite)
                row.update(opposite_skeleton_distance_px=float(distance),
                    opposite_nearest_rc=rc_nodes[other][index].tolist(),
                    opposite_image_supported=bool(distance <= limits.stereo_match_px))
        endpoint['stereo_evidence'] = row


def _view_pairs(endpoints, graphs, components, masks, occluders, limits):
    rows = []
    for a, b in itertools.combinations(endpoints, 2):
        if a['view'] != b['view'] or a['component_id'] == b['component_id']: continue
        side = a['view']; pa, pb = np.asarray(a['xy']), np.asarray(b['xy']); delta = pb - pa
        length = float(np.linalg.norm(delta)); direction = delta / max(length, 1e-12)
        tangents = [a['outward_tangent_xy'], b['outward_tangent_xy']]
        alignment = ([float(np.dot(tangents[0], direction)), float(np.dot(tangents[1], -direction))]
                     if all(t is not None for t in tangents) else None)
        chord = np.linspace(pa, pb, max(2, int(length) + 1))
        stereo = _opposite_path(a, b, graphs, components, limits)
        occ = float(at(occluders[side], chord).mean())
        categories = ('cross_view_visible_continuity_candidate' if stereo['continuous_visible_mask_path'] else
            'tool_occlusion_candidate' if occ > .1 else
            'boundary_reentry_candidate' if a['near_boundary'] or b['near_boundary'] else
            'unexplained_visible_gap')
        rows.append(dict(pair_id=_id('viewpair', sorted([a['endpoint_id'], b['endpoint_id']])),
            endpoint_ids=[a['endpoint_id'], b['endpoint_id']], view=side, classification=categories,
            gap_px=length, outward_direction_alignment=alignment,
            chord_occluder_fraction=occ, chord_is_diagnostic_not_hidden_shape=True,
            stereo_cues=stereo, tool_execution_eligible=False,
            rejection_feedback=['requires_bounded_3d_endpoint_pair_before_repair']))
    return rows


def _fragment_id(part):
    values = np.round(part['points_m'], 9).tolist()
    return _id('fragment', min(values, list(reversed(values))))


def _candidate_hidden_runs(parts, K, baseline, shape):
    """Recheck older checked bridges against the current both-view boundary rule."""
    rows = []
    for part in parts:
        inferred = np.asarray(part['geometry_inferred'], bool)
        changes = np.diff(np.r_[False, inferred, False].astype(int))
        starts, stops = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
        both_outside = np.ones(len(part['points_m']), bool)
        for shift in (0., baseline):
            both_outside &= ~in_view(project(part['points_m'], K, shift), shape)
        for start, stop in zip(starts, stops):
            # Stored connectors omit their two original curve endpoints, so
            # four extra samples match the curve[5:-5] tool verification.
            core = both_outside[start + 4:stop - 4] if stop - start > 8 else both_outside[start:stop]
            failed = bool(core.any())
            rows.append(dict(fragment_id=_fragment_id(part), start_index=int(start), stop_index=int(stop),
                unobserved_out_of_frame_fraction=float(core.mean()) if len(core) else 0.,
                current_boundary_guard_passed=not failed,
                failures=['unobserved_out_of_frame_connection'] if failed else []))
    return rows


def _represented_endpoint(endpoint, parts, candidate, K, baseline, limits, hidden_runs):
    """A checked path may already span a view-specific mask break.

    This records coverage by an existing proposal, including inferred samples;
    it never relabels hidden geometry as observed or resolves ambiguous strands.
    """
    if not candidate or candidate.get('report', {}).get('numerical_checks_passed') is not True:
        return None
    side = endpoint['view']; matches = []
    estimated_depth = endpoint['stereo_evidence'].get('estimated_depth_m')
    for part in parts:
        points = part['points_m']; xy = project(points, K, baseline if side == 'right' else 0.)
        distances_px = np.linalg.norm(xy - endpoint['xy'], axis=1)
        index = int(np.argmin(distances_px))
        if distances_px[index] > limits.stereo_match_px: continue
        arclength = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        if min(arclength[index], arclength[-1] - arclength[index]) < 6.: continue
        if estimated_depth is not None and abs(points[index, 2] - estimated_depth) / min(points[index, 2], estimated_depth) > limits.max_depth_relative_difference:
            continue
        fragment = _fragment_id(part)
        if any(run['fragment_id'] == fragment and not run['current_boundary_guard_passed'] and
               run['start_index'] - 2 <= index <= run['stop_index'] + 2 for run in hidden_runs):
            continue
        vicinity = slice(max(0, index - 2), min(len(points), index + 3))
        matches.append(dict(fragment_id=fragment, candidate_index=index,
            projected_distance_px=float(distances_px[index]),
            contains_inferred_shape=bool(np.any(part['geometry_inferred'][vicinity])),
            contains_inferred_depth=bool(np.any(part['depth_inferred'][vicinity])),
            representation_is_not_independent_accuracy=True))
    return matches[0] if len(matches) == 1 else None


def _geometry_pairs(case, pipeline, candidate, masks, graphs, components, endpoints_2d, limits):
    if not candidate or not candidate.get('fragment_path'):
        return [], [], ['ordered_3d_fragments_unavailable']
    parts = load_parts(candidate['fragment_path'])
    _, _, _, _, _, K, baseline = load_inputs(case, pipeline)
    try: tool_rows = endpoint_table(parts, K, baseline, masks['left'].shape, limits)
    except ValueError as error: return [], [], ['3d_endpoint_estimation_failed: ' + str(error)]
    endpoints = []; by_tool = {}
    for tool in tool_rows:
        fragment = _fragment_id(parts[tool['fragment']])
        stable = _id('endpoint3d', [fragment, np.round(tool['points_m'], 9).tolist()])
        associated = [r['endpoint_id'] for r in endpoints_2d
            if np.linalg.norm(np.asarray(r['xy']) - tool[r['view'] + '_xy']) <= 4.]
        row = dict(tool); row.update(endpoint_id=stable, tool_endpoint_id=tool['endpoint_id'],
            fragment_id=fragment, observed_endpoint_ids=associated,
            category_candidates=sorted({tag for r in endpoints_2d if r['endpoint_id'] in associated
                for tag in r['category_candidates']}) or ['visible_termination_unclassified'])
        endpoints.append(row); by_tool[tool['endpoint_id']] = row
    error = None; proposals = []
    if len(parts) > 1:
        if case.get('scene_priors', {}).get('single_thread') is not True:
            error = 'single_thread_prior_not_explicitly_declared'
        else:
            try: proposals = join_proposals(case, pipeline, parts, limits)
            except (ValueError, RuntimeError) as problem:
                error = 'protected_pair_diagnostics_failed: ' + str(problem)
    for proposal in proposals:
        curve = proposal['_curve']
        outside_both = np.ones(len(curve), bool)
        for side in ('left', 'right'):
            projected = project(curve, K, baseline if side == 'right' else 0.)
            outside_both &= ~in_view(projected, masks[side].shape)
        # Missing in both images is not occlusion evidence and never licenses
        # an arbitrary out-of-frame reconnection, even for a single thread.
        interior = outside_both[5:-5]
        proposal['unobserved_out_of_frame_fraction'] = float(interior.mean())
        if interior.any():
            proposal['failures'] = sorted(set(proposal['failures'] + ['unobserved_out_of_frame_connection']))
            proposal['numerical_checks_passed'] = False
    grouped = {}
    for proposal in proposals:
        key = tuple(sorted(proposal['endpoints'])); grouped.setdefault(key, []).append(proposal)
    rows = []
    for a, b in itertools.combinations(endpoints, 2):
        if a['fragment'] == b['fragment']: continue
        key = tuple(sorted([a['tool_endpoint_id'], b['tool_endpoint_id']]))
        public = [{k: v for k, v in p.items() if not k.startswith('_')} for p in grouped.get(key, [])]
        common = sorted(set(f for p in public for f in p['failures']))
        if error: common.append(error)
        universal = (sorted(set.intersection(*(set(p['failures']) for p in public))) if public else [])
        if error: universal.append(error)
        stereo = {}
        for side in ('left', 'right'):
            observed_a = next((e for e in endpoints_2d if e['view'] == side and e['endpoint_id'] in a['observed_endpoint_ids']), None)
            observed_b = next((e for e in endpoints_2d if e['view'] == side and e['endpoint_id'] in b['observed_endpoint_ids']), None)
            stereo[side] = (_opposite_path(observed_a, observed_b, graphs, components, limits)
                if observed_a and observed_b else dict(available=False, continuous_visible_mask_path=False,
                    reason='reconstructed_endpoint_not_a_terminal_in_this_view', hidden_depth_recovered=False))
        all_views = [v for p in public for v in p.get('views', {}).values()]
        classification = ('cross_view_visible_continuity_candidate' if any(v['continuous_visible_mask_path'] for v in stereo.values()) else
            'tool_occlusion_candidate' if any(v.get('occluder_fraction', 0) > .1 for v in all_views) else
            'boundary_reentry_candidate' if a['near_boundary'] or b['near_boundary'] else 'unexplained_visible_gap')
        accepted = [p['proposal_id'] for p in public if p['numerical_checks_passed']]
        competitors = [p for p in proposals if p['numerical_checks_passed'] and
            tuple(sorted(p['endpoints'])) != key and set(p['endpoints']) & set(key)]
        ambiguity = bool(accepted and competitors and min(p['score'] for p in competitors) -
            min(p['score'] for p in public if p['numerical_checks_passed']) < limits.ambiguity_score_margin)
        rows.append(dict(pair_id=_id('pair3d', sorted([a['endpoint_id'], b['endpoint_id']])),
            endpoint_ids=[a['endpoint_id'], b['endpoint_id']], tool_endpoint_ids=list(key),
            fragment_ids=[a['fragment_id'], b['fragment_id']], classification=classification,
            stereo_cues=stereo, scale_proposals=public, feasible_proposal_ids=accepted,
            rejection_feedback=common, universal_rejection_feedback=universal,
            all_scales_rejected=not bool(accepted), endpoint_pair_ambiguous=ambiguity,
            tool_execution_eligible=bool(accepted and not ambiguity),
            pair_diagnostics_are_not_independent_accuracy=True,
            inferred_hidden_geometry_is_not_ground_truth=True))
    return endpoints, rows, [error] if error else []


def build_gap_inventory(case, pipeline, candidate=None, out=None):
    """Return JSON-safe per-endpoint assessment tasks and all pairing feedback.

    ``case`` is the shared case dictionary (or its JSON path); ``candidate`` is
    the existing v6 ``{fragment_path, report, ...}`` dictionary. ``out`` optionally
    saves the inventory and labeled *full-frame* stereo observations. No repair
    is executed, no original changes, and one difficult exit never hides others.
    """
    if isinstance(case, (str, Path)): case = json.loads(Path(case).read_text())
    pipeline = Path(pipeline); limits = RepairLimits()
    paths = dict(case['inputs'])
    if candidate and candidate.get('fragment_path'): paths['candidate_fragments'] = candidate['fragment_path']
    before = fingerprints(paths)
    source_paths = dict(inventory=Path(__file__),
        protected_pair_tools=Path(__file__).with_name('thread_visibility_repairs_v5.py'),
        existing_pipeline_base=pipeline / 'reconstruct_thread_from_files.py')
    source_provenance = fingerprints(source_paths)
    base, images, masks, occluders, depth, K, baseline = load_inputs(case, pipeline)
    endpoints = []; graphs = {}; component_nodes = {}; views = {}
    for side in ('left', 'right'):
        needle = (base.load_mask(case['inputs']['needle_' + side + '_mask'], masks[side].shape)
            if 'needle_' + side + '_mask' in case['inputs'] else np.zeros_like(masks[side]))
        rows, components, graphs[side], component_nodes[side], audit = _view_inventory(
            masks[side], occluders[side], needle, side)
        endpoints.extend(rows); views[side] = dict(components=components, audit=audit,
            endpoint_count=len(rows), full_frame=True, image_crop=None,
            tool_occluder_supplied=side + '_occluder' in case['inputs'])
    _stereo_endpoint_evidence(endpoints, graphs, masks, depth, K, baseline, base, limits)
    view_pairs = _view_pairs(endpoints, graphs, component_nodes, masks, occluders, limits)
    endpoints_3d, pairs, limitations = _geometry_pairs(case, pipeline, candidate, masks,
        graphs, component_nodes, endpoints, limits)
    parts = load_parts(candidate['fragment_path']) if candidate and candidate.get('fragment_path') else []
    hidden_runs = _candidate_hidden_runs(parts, K, baseline, masks['left'].shape)
    gaps = []; represented = set()
    for endpoint in endpoints_3d:
        represented.update(endpoint['observed_endpoint_ids'])
        choices = [p for p in pairs if endpoint['endpoint_id'] in p['endpoint_ids']]
        sources = sorted({r['view'] for r in endpoints if r['endpoint_id'] in endpoint['observed_endpoint_ids']})
        reasons = [] if any(p['tool_execution_eligible'] for p in choices) else ['no_verified_unique_pair_yet']
        gaps.append(dict(gap_id='gap.' + endpoint['endpoint_id'],
            kind='boundary_exit' if endpoint['near_boundary'] else 'interior_endpoint',
            endpoint_ids=[endpoint['endpoint_id']], observed_endpoint_ids=endpoint['observed_endpoint_ids'],
            source_views=sources, candidate_pair_ids=[p['pair_id'] for p in choices],
            status='needs_assessment', unresolved_reasons=reasons,
            is_confirmed_physical_disconnect=False))
    for endpoint in endpoints:
        if endpoint['endpoint_id'] in represented: continue
        choices = [p for p in view_pairs if endpoint['endpoint_id'] in p['endpoint_ids']]
        existing = _represented_endpoint(endpoint, parts, candidate, K, baseline, limits, hidden_runs)
        gaps.append(dict(gap_id='gap.' + endpoint['endpoint_id'],
            kind='boundary_exit' if endpoint['near_boundary'] else 'interior_endpoint',
            endpoint_ids=[endpoint['endpoint_id']], observed_endpoint_ids=[endpoint['endpoint_id']],
            source_views=[endpoint['view']], candidate_pair_ids=[p['pair_id'] for p in choices],
            status='represented_by_checked_candidate' if existing else 'needs_assessment',
            candidate_representation=existing,
            unresolved_reasons=[] if existing else ['bounded_3d_endpoint_not_available'],
            is_confirmed_physical_disconnect=False))
    for side, view in views.items():
        for component in view['components']:
            if component['topology'] in ('loop', 'branched'):
                gaps.append(dict(gap_id='topology.' + component['component_id'], kind='topology_ambiguity',
                    endpoint_ids=[e['endpoint_id'] for e in endpoints if e['component_id'] == component['component_id']],
                    source_views=[side], candidate_pair_ids=[], status='needs_assessment',
                    unresolved_reasons=['projected_path_order_not_unique'], is_confirmed_physical_disconnect=False))
    public_limits = asdict(limits); public_limits['tangent_scales'] = list(limits.tangent_scales)
    uncertainty_groups = []
    for side in ('left', 'right'):
        ids = [e['endpoint_id'] for e in endpoints if e['view'] == side and
            'short_mask_feature_uncertain' in e['category_candidates']]
        if ids: uncertainty_groups.append(dict(group_id=side + '.short_mask_features', view=side,
            endpoint_ids=ids, assessment='jointly_classify_mask_noise_or_visible_short_thread',
            group_is_not_a_physical_disconnect=True))
    result = dict(schema='thread_gap_inventory_v7', case_id=case['case_id'],
        inventory_version=_id('inventory', dict(inputs=before, sources=source_provenance)), views=views,
        endpoints_2d=endpoints, endpoints_3d=endpoints_3d, gaps=gaps,
        pairing_candidates=pairs, view_pairing_candidates=view_pairs,
        uncertainty_groups=uncertainty_groups,
        candidate_hidden_run_revalidation=hidden_runs,
        limits=public_limits, limitations=limitations, provenance=before,
        source_provenance=source_provenance,
        inventory_is_exhaustive_for_visible_mask_endpoints=True,
        hidden_endpoints_discovered=False, gap_is_endpoint_assessment_unit=True,
        per_gap_deferral_only=True, global_deferral_required=False,
        observed_geometry_modified=False, independent_accuracy_measured=False,
        inferred_hidden_geometry_is_ground_truth=False, training_eligible=False)
    # Validate serializability before writing any deliverable.
    json.dumps(result, allow_nan=False)
    if out is not None:
        out = Path(out); out.mkdir(parents=True, exist_ok=False)
        result['artifacts'] = {}
        for side in ('left', 'right'):
            image = cv2.cvtColor(images[side], cv2.COLOR_RGB2BGR)
            for index, endpoint in enumerate(r for r in endpoints if r['view'] == side):
                xy = tuple(np.rint(endpoint['xy']).astype(int))
                color = (35, 200, 255) if endpoint['near_boundary'] else (90, 225, 90)
                cv2.circle(image, xy, 4, color, -1)
                cv2.putText(image, str(index), (max(0, xy[0] + 6), max(13, xy[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1, cv2.LINE_AA)
                endpoint['overlay_label'] = str(index)
            target = out / (side + '_gap_inventory.png')
            if not cv2.imwrite(str(target), image): raise RuntimeError('Could not save inventory overlay')
            result['artifacts'][side] = str(target.resolve())
        write_json(out / 'report.json', result)
    if fingerprints(paths) != before: raise ValueError('Inventory inputs changed during inspection')
    if fingerprints(source_paths) != source_provenance: raise ValueError('Inventory sources changed during inspection')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True); parser.add_argument('--pipeline', required=True)
    parser.add_argument('--candidate-summary'); parser.add_argument('--out', required=True)
    args = parser.parse_args(); candidate = None
    if args.candidate_summary:
        document = json.loads(Path(args.candidate_summary).read_text()); candidate = document.get('candidate', document)
    result = build_gap_inventory(args.case, args.pipeline, candidate, args.out)
    print(json.dumps(dict(case_id=result['case_id'], endpoints=len(result['endpoints_2d']),
        assessment_units=len(result['gaps']), pairs=len(result['pairing_candidates']))))


if __name__ == '__main__': main()
