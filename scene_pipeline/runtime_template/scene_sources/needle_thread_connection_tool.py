"""Bounded, inspectable scene-topology tool shared by agent and heuristic.

Only the controller's choice differs. The tool preserves reconstructed samples,
adds at most one explicitly inferred connector, and never promotes physics or
accuracy merely because the model requested an attachment.
"""
import json
from pathlib import Path

import numpy as np

from compact_agent_protocol import digest
from newton_scene_adapter import centerline, project, sha256, transform
from reconstruction_tools import strict_json


def decode_attachment_response(raw):
    """Remove transport wrappers only; never extract or repair a JSON payload."""
    from run_model_tool_checkpoint import decode_transport
    text, transport = decode_transport(raw)
    text = text.strip()
    fenced = text.startswith('```json\n') and text.endswith('\n```')
    if fenced:
        text = text[len('```json\n'):-len('\n```')]
    # Validate the whole remaining document. Prefix/suffix prose and multiple
    # fenced blocks are rejected; semantic validation remains unchanged.
    strict_json(text)
    return text, transport | {'removed_whole_json_fence': fenced}


def attachment_state(geometry, bundle_path, stereo_paths, baseline_m):
    t = centerline(geometry['points_newton_m'])
    n = centerline(geometry['needle_points_newton_m'])
    candidates = []
    for ti in (0, len(t)-1):
        for ni in (0, len(n)-1):
            candidates.append(dict(id=f't{ti}_n{ni}', thread_endpoint_index=ti,
                needle_endpoint_index=ni, distance_m=float(np.linalg.norm(t[ti]-n[ni])),
                endpoints_supported=bool(not geometry['inferred'][ti] and not geometry['needle_inferred'][ni])))
    images = {k: dict(path=str(Path(v).resolve()), sha256=sha256(v)) for k, v in stereo_paths.items()}
    if set(images) != {'left', 'right'}:
        raise ValueError('Both stereo images required')
    state = dict(protocol='needle_thread_attachment_v1', target='attachment',
        bundle_sha256=sha256(bundle_path), stereo_images=images, baseline_m=baseline_m,
        candidates=candidates, max_connector_length_m=.004,
        confidence_is_not_ground_truth=True, geometry_accuracy_measured=False,
        operations=['attach_needle_thread', 'defer'], evidence_ids=['stereo', 'endpoints'])
    state['scene_version'] = digest(state)
    return state


def validate_attachment_decision(decision, state):
    d = strict_json(decision) if isinstance(decision, str) else strict_json(json.dumps(decision, allow_nan=False))
    if set(d) != {'scene_version', 'target', 'operation', 'parameters', 'evidence_ids', 'reason'}:
        raise ValueError('Exactly six attachment decision fields required')
    if d['scene_version'] != state['scene_version'] or d['target'] != 'attachment':
        raise ValueError('Stale attachment decision or wrong target')
    if not isinstance(d['reason'], str) or not 1 <= len(d['reason'].strip()) <= 500:
        raise ValueError('Concise evidence-based reason required')
    ids = d['evidence_ids']
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids) or not set(ids) <= set(state['evidence_ids']):
        raise ValueError('Invalid attachment evidence IDs')
    if d['operation'] == 'defer':
        if d['parameters'] != {}:
            raise ValueError('Defer parameters must be empty')
        return d, None
    if d['operation'] != 'attach_needle_thread' or set(d['parameters']) != {'candidate_id'}:
        raise ValueError('Unknown operation or protected parameters')
    eligible = [c for c in state['candidates'] if c['distance_m'] <= state['max_connector_length_m']]
    if (not 0 < state['max_connector_length_m'] <= .004 or len(eligible) != 1
            or not eligible[0]['endpoints_supported'] or d['parameters']['candidate_id'] != eligible[0]['id']):
        raise ValueError('Unsupported, ambiguous or out-of-bound connection')
    if not {'stereo', 'endpoints'} <= set(ids):
        raise ValueError('Connection requires both stereo and endpoint evidence')
    return d, eligible[0]


def prepare_connected_geometry(geometry, candidate, baseline_m, max_length_m=.004):
    """Create a derived physics bundle; original centerlines remain bit-identical."""
    t, n = centerline(geometry['points_newton_m']), centerline(geometry['needle_points_newton_m'])
    ti, ni = candidate['thread_endpoint_index'], candidate['needle_endpoint_index']
    if ti not in (0, len(t)-1) or ni not in (0, len(n)-1):
        raise ValueError('Only terminal material endpoints can be connected')
    gap = float(np.linalg.norm(t[ti]-n[ni]))
    if not np.isfinite(gap) or gap > max_length_m or max_length_m > .004 or max_length_m <= 0:
        raise ValueError('Connector exceeds protected bound')
    inferred_endpoint = bool(geometry['inferred'][ti] or geometry['needle_inferred'][ni])
    authorized = (candidate.get('inferred_connection_authorized') is True
        and candidate.get('specialist_asserted_physical_swage') is True
        and candidate.get('numerical_checks_passed') is True
        and not candidate.get('blockers', ['missing_checks'])
        and {'image.left', 'image.right'} <= set(candidate.get('visual_evidence_ids', [])))
    if inferred_endpoint and not authorized:
        raise ValueError('Inferred endpoint requires verified stereo attachment selection')
    out = {k: np.asarray(v).copy() for k, v in geometry.items()}
    out['source_points_newton_m'] = t.copy()
    out['source_needle_points_newton_m'] = n.copy()
    out['source_inferred'] = geometry['inferred'].copy()
    bridge = gap > 1e-9
    if bridge:
        prepend = ti == 0
        out['points_newton_m'] = np.vstack((n[ni], t)) if prepend else np.vstack((t, n[ni]))
        for key, value in [('inferred', True), ('observed', False), ('reliability', 0.)]:
            if key in out:
                out[key] = np.r_[value, out[key]] if prepend else np.r_[out[key], value]
        out['points_camera_m'] = transform(out['points_newton_m'], np.linalg.inv(out['T_newton_from_camera']))
        out['left_pixel_xy'] = project(out['points_camera_m'], out['K'])
        out['right_pixel_xy'] = project(out['points_camera_m'], out['K'], baseline_m)
    connected_ti = 0 if ti == 0 else len(out['points_newton_m'])-1
    attachment = dict(applied=True, thread_endpoint_index=connected_ti, needle_endpoint_index=ni,
        endpoint_distance_m=0. if bridge else gap, kind='inferred_needle_thread_topology',
        joint_type='ball', source_endpoint_gap_m=gap,
        connector_added=bridge, connector_length_m=gap if bridge else 0.,
        inferred_connector_point_index=connected_ti if bridge else None,
        source_geometry_preserved=True, attachment_ground_truth_measured=False,
        source_thread_endpoint_inferred=bool(geometry['inferred'][ti]),
        source_needle_endpoint_inferred=bool(geometry['needle_inferred'][ni]),
        endpoint_support_policy='stereo_assessed_inferred_allowed_v9',
        uncertainty_warnings=candidate.get('uncertainty_warnings', []))
    out['attachment_connector_added'] = np.asarray(bridge)
    from anchor_contract import canonicalize
    out, attachment = canonicalize(out, attachment)
    out['left_pixel_xy'] = project(out['points_camera_m'], out['K'])
    out['right_pixel_xy'] = project(out['points_camera_m'], out['K'], baseline_m)
    return out, attachment


def local_contact_pairs(points, radius_m, needle_radius_m, contact_gap_m, connector_length_m):
    """Filter cap/margin overlaps by topology, retaining nonlocal collisions.

    Points are in actual rod construction order, with attached endpoint last.
    A short inferred continuation belongs to the junction neighborhood; the
    remaining exclusion extent is determined by the assumed wire radii/margin.
    """
    p = centerline(points)
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    local_extent = 2*radius_m + 2*contact_gap_m
    rod_pairs = [(i, j) for i in range(len(p)-1) for j in range(i+1, len(p)-1)
                 if s[j]-s[i+1] <= local_extent]
    junction_extent = connector_length_m + radius_m + needle_radius_m + 2*contact_gap_m
    junction_bodies = [i for i in range(len(p)-1) if s[-1]-s[i+1] <= junction_extent]
    return rod_pairs, junction_bodies, dict(rod_local_extent_m=local_extent,
        junction_extent_m=junction_extent, junction_rod_indices=junction_bodies,
        distant_contacts_preserved=True)


def attachment_messages(state, labeled_images):
    rules = ('Inspect both stereo views to determine whether the thread is attached to the needle, '
        'and which displayed endpoint pair is the junction. t/N markers identify reconstructed endpoints; '
        'do not mistake the free sharp needle end for the thread attachment. Numbers are estimated geometry, '
        'not ground truth. Return only six-field JSON: scene_version, target=attachment, operation, parameters, '
        'evidence_ids, reason<=500 characters. Choose attach_needle_thread with parameters={candidate_id: one '
        'listed ID}, evidence_ids=[stereo,endpoints], only if the image relation and unique bounded pair agree. '
        'Otherwise defer with parameters={}, citing evidence. Never alter geometry, physical parameters or gates. '
        'A requested attachment is a hypothesis; tool and physics validation follow.\nSTATE:'+json.dumps(state))
    content = [dict(type='text', text=rules)]
    for side in ('left', 'right'):
        content.extend([dict(type='text', text=side+' stereo view with endpoint labels'),
                        dict(type='image', url=str(labeled_images[side]))])
    return [dict(role='user', content=content)]
