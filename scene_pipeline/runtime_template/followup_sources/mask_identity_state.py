"""Carry native class confidences across partial perception updates."""
import numpy as np
from pathlib import Path
import json,hashlib

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_confidence_state(case_path,case):
    """Carry confidence only when bound to these exact images and class masks."""
    path=Path(case_path).parent/'class_confidence_state.json'
    if not path.exists():return {},{}
    value=json.loads(path.read_text())
    if value.get('schema')!='native_class_confidence_state_v1' or value.get('case_sha256')!=sha(case_path):
        raise ValueError('Class confidence state belongs to another case revision')
    expected={'left','right'}|{t+'_'+s+'_mask' for t in ('thread','needle') for s in ('left','right')}
    if set(value.get('bindings',{}))!=expected:raise ValueError('Complete image/class mask bindings required')
    protected={'class_confidence_state':str(path)}
    for key,item in value['bindings'].items():
        if item['path']!=case['inputs'][key] or sha(item['path'])!=item['sha256']:
            raise ValueError('Class confidence image/mask binding changed')
    arrays={}
    for key,item in value['artifacts'].items():
        if key not in {t+'.'+s for t in ('thread','needle') for s in ('left','right')} or sha(item['path'])!=item['sha256']:
            raise ValueError('Carried native confidence changed')
        arrays[key]=np.load(item['path'],allow_pickle=False);protected['confidence.'+key]=item['path']
    return arrays,protected

def resolve_incremental_classes(masks,confidences):
    """Use the same joint-class rule after updating either class alone.

    Missing confidence remains explicit. A binary old mask must never be given
    fabricated probability 0/1 merely to force the new class to win.
    """
    result={name:np.asarray(value,bool).copy() for name,value in masks.items()}
    if set(result)!={'thread','needle'}:raise ValueError('Both current class masks required')
    if result['thread'].shape!=result['needle'].shape:raise ValueError('Class mask shape mismatch')
    overlap=result['thread']&result['needle'];count=int(overlap.sum())
    available=set(confidences)=={'thread','needle'}
    report=dict(overlap_pixels_before=count,confidences_available=sorted(confidences),
        class_resolution_pending=bool(count and not available),native_confidences_are_not_ground_truth=True)
    if available:
        for value in confidences.values():
            if np.asarray(value).shape!=overlap.shape or not np.isfinite(value).all():raise ValueError('Invalid carried class confidence')
        thread_wins=np.asarray(confidences['thread'])>=np.asarray(confidences['needle'])
        result['thread'][overlap&~thread_wins]=False;result['needle'][overlap&thread_wins]=False
    report['overlap_pixels_after']=int((result['thread']&result['needle']).sum())
    return result,report
