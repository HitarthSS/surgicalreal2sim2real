"""Bounded shared perception tool contract; no model or filesystem dependencies."""
import hashlib
import json
import math

PROMPTS = {n:['surgical grasper','robotic surgical grasper','surgical instrument'] for n in ['tool_A','tool_B']}

THRESHOLDS = [.1, .2, .35]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_box(value, size):
    if not isinstance(value, dict) or set(value) != {'xyxy', 'positive'} or type(value['positive']) is not bool:
        raise ValueError('Box requires xyxy and boolean positive')
    xy = value['xyxy']
    if not isinstance(xy, list) or len(xy) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in xy):
        raise ValueError('Four finite full-frame pixel coordinates required')
    x0, y0, x1, y1 = xy
    w, h = size
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h and x1-x0 >= 2 and y1-y0 >= 2):
        raise ValueError('Box outside original image or degenerate')
    return [(x0+x1)/(2*w), (y0+y1)/(2*h), (x1-x0)/w, (y1-y0)/h]


def validate_request(request, sizes):
    if not isinstance(request, dict) or set(request) != {'objects'} or not request['objects'] or set(request['objects']) - set(PROMPTS):
        raise ValueError('Select thread, needle, or both objects')
    for target, views in request['objects'].items():
        if set(views) != {'left', 'right'}:
            raise ValueError('Both complete stereo views required')
        for side, params in views.items():
            if set(params) != {'prompt', 'confidence_threshold', 'boxes'} or params['prompt'] not in PROMPTS[target]:
                raise ValueError('Unknown prompt or parameter fields')
            threshold = params['confidence_threshold']
            if type(threshold) not in (int, float) or threshold not in THRESHOLDS:
                raise ValueError('Confidence threshold outside declared menu')
            if not isinstance(params['boxes'], list) or len(params['boxes']) > 4:
                raise ValueError('At most four box prompts per object/view')
            for box in params['boxes']:
                validate_box(box, sizes[side])
    return request


def validate_decision(value, state):
    if not isinstance(value, dict) or set(value) != {'scene_version', 'action', 'parameters', 'evidence_ids', 'reason'}:
        raise ValueError('Exact perception decision fields required')
    if value['scene_version'] != state['scene_version']:
        raise ValueError('Stale perception state')
    if not isinstance(value['evidence_ids'], list) or not {'image.left', 'image.right'} <= set(value['evidence_ids']) or set(value['evidence_ids']) - set(state['evidence_ids']):
        raise ValueError('Cite both registered original stereo views')
    if not isinstance(value['reason'], str) or not 1 <= len(value['reason']) <= 700:
        raise ValueError('Concise visual reason required')
    params = value['parameters']
    if value['action'] == 'repair_masks':
        if state['segmentation_calls'] >= state['max_segmentation_calls']:
            raise ValueError('Perception tool budget exhausted')
        validate_request(params, state['image_sizes'])
    elif value['action'] == 'accept_proposal':
        if not state.get('proposal_available'):
            raise ValueError('No executed proposal to accept')
        if not state.get('numerically_eligible'):
            raise ValueError('Cannot override failed numerical eligibility')
        if not isinstance(params, dict) or set(params) != {'reviewed_removed_pixels', 'visible_objects_preserved'} or params['reviewed_removed_pixels'] is not True:
            raise ValueError('Explicit removed-pixel visual review required')
        views = params['visible_objects_preserved']
        if not isinstance(views, dict) or set(views) != {'left', 'right'} or any(v is not True for v in views.values()):
            raise ValueError('Both-view object preservation assessment required')
    elif value['action'] == 'defer':
        if params != {}:
            raise ValueError('Defer takes no parameters')
    else:
        raise ValueError('Unknown perception action')
    return value


def numerical_eligibility(baseline, proposal, targets):
    """Consistency-only promotion: recover a targeted failed estimator, no regression."""
    before = {'needle': baseline['needle']['accepted'], 'thread': baseline['thread']['checks_passed']}
    after = {'needle': proposal['needle']['accepted'], 'thread': proposal['thread']['checks_passed']}
    recovered = [target for target in targets if after[target] and not before[target]]
    regressions = [target for target in before if before[target] and not after[target]]
    return dict(eligible=bool(recovered and not regressions), recovered=recovered, regressions=regressions,
                independent_accuracy_measured=False, visual_review_still_required=True)
