"""Select whole native instances identified by explicit positive/negative boxes.

Never clip a native mask to a box or select its largest connected component.
All components of a selected instance survive; unselected instances are saved.
"""
import hashlib
import numpy as np

def numpy(value):return value.detach().float().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)

def select(output,height,width,threshold,boxes):
    scores=numpy(output['scores']).reshape(-1).astype('float32')
    bounds=numpy(output['boxes']).reshape(-1,4).astype('float32')
    probabilities=numpy(output['masks_logits']).reshape(-1,height,width).astype('float32')
    if not len(scores)==len(bounds)==len(probabilities):raise ValueError('Inconsistent native instance counts')
    selected=[];rows=[];positive=[b for b in boxes if b['positive']];negative=[b for b in boxes if not b['positive']]
    def pixels(mask,regions):
        return sum(int(mask[int(np.floor(b['xyxy'][1])):int(np.ceil(b['xyxy'][3]))+1,
                            int(np.floor(b['xyxy'][0])):int(np.ceil(b['xyxy'][2]))+1].sum()) for b in regions)
    for index in np.flatnonzero(scores>=threshold):
        mask=probabilities[index]>.5;pos=pixels(mask,positive);neg=pixels(mask,negative);count=int(mask.sum())
        reason='selected_native_instance'
        if not count:reason='empty_native_instance'
        elif positive and not pos:reason='no_foreground_in_any_explicit_positive_box'
        elif positive and neg>=pos:reason='more_foreground_on_explicit_negative_than_positive_objects'
        elif not positive and neg>count/2:reason='native_instance_mostly_on_explicit_negative_object'
        keep=reason=='selected_native_instance'
        if keep:selected.append(int(index))
        rows.append(dict(native_index=int(index),score=float(scores[index]),selected=keep,reason=reason,
            foreground_pixels=count,positive_box_foreground_pixels=pos,negative_box_foreground_pixels=neg,
            complete_native_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),mask_clipped=False))
    accepted=probabilities[selected];accepted_scores=scores[selected]
    confidence=np.max(accepted*accepted_scores[:,None,None],axis=0) if selected else np.zeros((height,width),'float32')
    mask=np.any(accepted>.5,axis=0) if selected else np.zeros((height,width),bool)
    prediction=dict(mask=mask,confidence=confidence,scores=accepted_scores,boxes=bounds[selected],candidate_scores=scores)
    audit=dict(schema='whole_native_instance_selection_v1',instances=rows,selected_native_indices=selected,
        complete_native_components_preserved=True,largest_component_filter=False,box_clipping=False,
        selection_is_hypothesis=True,independent_accuracy_measured=False)
    raw_indices=np.flatnonzero(scores>=threshold)
    raw=dict(native_indices=raw_indices,scores=scores[raw_indices],boxes=bounds[raw_indices],probabilities=probabilities[raw_indices])
    return prediction,audit,raw
