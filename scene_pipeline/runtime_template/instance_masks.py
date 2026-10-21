"""Select observed SAM3 components belonging to one bounded tool hypothesis.

No pixels are drawn, filled, cropped or clipped. The native fitter and stereo
visual verifier still decide whether the selected evidence supports a tool.
"""
import cv2
import numpy as np


def select_instance(mask, boxes, *, entry_border=None, max_gap_px=32., max_parts=4):
    mask = np.asarray(mask, bool)
    if mask.ndim != 2 or entry_border not in (None, 'left', 'right'):
        raise ValueError('Full-frame binary mask and known visual entry required')
    if not 0 <= max_gap_px <= 64 or not 1 <= max_parts <= 4:
        raise ValueError('Component grouping bounds exceeded')
    h, w = mask.shape
    positive, negative = np.zeros_like(mask), np.zeros_like(mask)
    for box in boxes:
        x0, y0, x1, y1 = box['xyxy']
        if not all(np.isfinite([x0, y0, x1, y1])) or not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError('Box outside full original image')
        region = positive if box['positive'] else negative
        region[int(np.floor(y0)):int(np.ceil(y1)), int(np.floor(x0)):int(np.ceil(x1))] = True
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'), 8)
    rows = []
    for i in range(1, count):
        part = labels == i
        area = int(stats[i, cv2.CC_STAT_AREA])
        hit, excluded = int((part & positive).sum()), int((part & negative).sum())
        left, right = bool(part[:, :2].any()), bool(part[:, -2:].any())
        eligible = 64 <= area <= mask.size * .30 and hit >= 32 and excluded <= max(8, area * .1) and not (left and right)
        rows.append(dict(component=i, area=area, positive_support_pixels=hit,
                         negative_support_pixels=excluded, touches_left=left, touches_right=right,
                         touches_both_entry_borders=left and right, eligible=eligible, score=hit - 2 * excluded))
    candidates = sorted([r for r in rows if r['eligible']], key=lambda r: r['score'], reverse=True)
    if not candidates:
        raise ValueError('No target-supported SAM3 component')
    anchors = [r for r in candidates if entry_border and r['touches_' + entry_border]]
    if len(anchors) > 1:
        raise ValueError('Multiple tool entries on the requested border; improve bounds')
    if not anchors:
        # Preserve the previous single-component behavior when the entry is not
        # visible. Do not guess a multipart identity without an entry anchor.
        if len(candidates) > 1 and candidates[1]['score'] >= candidates[0]['score'] * .9:
            raise ValueError('Ambiguous SAM3 instance components; request better bounds')
        chosen = candidates[:1]
        mode = 'single_component_entry_unobserved'
    else:
        chosen = anchors[:]
        mode = 'entry_anchored_observed_component_group'
        opposite = 'right' if entry_border == 'left' else 'left'
        while True:
            selected = np.isin(labels, [r['component'] for r in chosen])
            distance = cv2.distanceTransform((~selected).astype('uint8'), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            neighbors = []
            for row in candidates:
                if row in chosen or row['touches_' + opposite]:
                    continue
                # A disconnected addition needs substantial explicit target-box
                # support; the existing anchored component is never clipped.
                if row['positive_support_pixels'] / row['area'] < .75:
                    continue
                gap = float(distance[labels == row['component']].min())
                if gap <= max_gap_px:
                    neighbors.append((gap, row))
            if not neighbors:
                break
            if len(chosen) >= max_parts:
                raise ValueError('Too many nearby target components; request more specific bounds')
            neighbors.sort(key=lambda item: (item[0], -item[1]['score']))
            chosen.append(neighbors[0][1])
            if sum(r['area'] for r in chosen) > mask.size * .30:
                raise ValueError('Tool component group exceeds original area bound')
    result = np.isin(labels, [r['component'] for r in chosen])
    assert not (result & ~mask).any()
    return result, dict(selected_component=chosen[0], selected_components=chosen,
                        components=rows, selection_policy=mode, entry_border=entry_border,
                        max_gap_px=max_gap_px, max_parts=max_parts, grouping_is_hypothesis=True,
                        stereo_identity_verification_required=True,
                        discarded_components_preserved_in_union=True, mask_clipped_to_box=False,
                        synthesized_pixels=0)
