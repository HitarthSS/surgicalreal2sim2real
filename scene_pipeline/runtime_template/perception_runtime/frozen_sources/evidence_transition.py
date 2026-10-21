"""Keep bounded working occlusion evidence without promoting scene geometry."""
from pathlib import Path
import json
import cv2
import numpy as np


def retain_occlusion_evidence(before, after, candidate, proposal, operation, audit):
    if operation not in ('occluder_mask', 'occluder_polygons') or proposal is not candidate:
        return False, 'not_an_evidence_only_occlusion_operation'
    if not audit or audit.get('no_op'):
        return False, 'no_new_occlusion_evidence'
    allowed = {'left_occluder', 'right_occluder'}
    old, new = before['case'], after.case
    if {k: v for k, v in old.items() if k != 'inputs'} != {k: v for k, v in new.items() if k != 'inputs'}:
        return False, 'protected_case_metadata_changed'
    if {k: v for k, v in old['inputs'].items() if k not in allowed} != {k: v for k, v in new['inputs'].items() if k not in allowed}:
        return False, 'protected_observation_changed'
    changed = False
    for side in ('left', 'right'):
        path = new['inputs'].get(side + '_occluder')
        if not path or not Path(path).is_file():
            return False, 'missing_stereo_occlusion_mask'
        mask = cv2.imread(path, 0)
        original = cv2.imread(old['inputs'][side], 0)
        if mask is None or original is None or mask.shape != original.shape:
            return False, 'invalid_full_frame_mask'
        thread = cv2.imread(old['inputs']['thread_' + side + '_mask'], 0)
        needle = cv2.imread(old['inputs']['needle_' + side + '_mask'], 0)
        if thread is None or needle is None or thread.shape != mask.shape or needle.shape != mask.shape:
            return False, 'missing_protected_class_masks'
        if np.any((mask > 0) & ((thread > 0) | (needle > 0))):
            return False, 'occluder_erases_observed_thread_or_needle'
        previous = old['inputs'].get(side + '_occluder')
        prior = cv2.imread(previous, 0) if previous else np.zeros_like(mask)
        if prior is None or prior.shape != mask.shape:
            return False, 'invalid_previous_occluder'
        changed |= not np.array_equal(prior > 0, mask > 0)
    if not changed:
        return False, 'no_new_occlusion_evidence'
    if after.version <= before['version'] or not Path(after.working_path).is_file():
        return False, 'unversioned_evidence_hypothesis'
    if json.loads(Path(after.working_path).read_text()) != after.case:
        return False, 'derived_case_does_not_bind_working_evidence'
    return True, 'bounded_occlusion_hypothesis_retained_for_next_repair'
