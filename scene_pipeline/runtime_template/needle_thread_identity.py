"""Audited removal of long duplicate needle geometry from a thread hypothesis."""
from pathlib import Path
import hashlib,json
import cv2,numpy as np
from scipy.spatial import cKDTree

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def duplicate_runs(parts,needle,K,baseline,shape):
    from geometry_repairs import project
    n=np.asarray(needle['points_m']);observed=~np.asarray(needle['inferred'],bool)
    nt=np.gradient(n,axis=0);nt/=np.maximum(np.linalg.norm(nt,axis=1,keepdims=True),1e-12)
    tree=cKDTree(n);rows=[]
    for f,part in enumerate(parts):
        p=np.asarray(part['points_m']);distance,indices=tree.query(p)
        tangent=np.gradient(p,axis=0);tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-12)
        hit=(distance<.0005)&observed[indices]&(np.abs(np.sum(tangent*nt[indices],axis=1))>.9)
        # The physical swage vicinity is not a duplicate strand.
        hit&=(np.minimum(np.linalg.norm(p-n[0],axis=1),np.linalg.norm(p-n[-1],axis=1))>.001)
        for b in (0.,baseline):
            xy=project(p,K,b);xy_n=project(n,K,b)
            hit&=(np.linalg.norm(xy-xy_n[indices],axis=1)<=2.)
            hit&=(xy[:,0]>=0)&(xy[:,1]>=0)&(xy[:,0]<shape[1])&(xy[:,1]<shape[0])
        boundaries=np.diff(np.r_[False,hit,False].astype(int));starts=np.flatnonzero(boundaries==1);ends=np.flatnonzero(boundaries==-1)
        for start,end in zip(starts,ends):
            length=float(np.linalg.norm(np.diff(p[start:end],axis=0),axis=1).sum())
            if end-start>=8 and length>=.001:rows.append(dict(fragment=f,start=int(start),end_exclusive=int(end),length_m=length))
    return rows

def trim_parts(parts,runs):
    result=[]
    for f,part in enumerate(parts):
        count=len(part['points_m']);keep=np.ones(count,bool)
        for run in runs:
            if run['fragment']==f:keep[run['start']:run['end_exclusive']]=False
        boundaries=np.diff(np.r_[False,keep,False].astype(int))
        for start,end in zip(np.flatnonzero(boundaries==1),np.flatnonzero(boundaries==-1)):
            if end-start<3:continue
            result.append({k:v[start:end].copy() if isinstance(v,np.ndarray) and v.ndim and len(v)==count else v for k,v in part.items()})
    if not result:raise ValueError('Identity cleanup cannot erase the whole thread')
    return result

def confirmed_exclusions(case,receipt_path):
    """Only this tool's image/mask/geometry-bound receipt can change coverage labels."""
    receipt=json.loads(Path(receipt_path).read_text())
    if receipt.get('schema')!='needle_thread_identity_cleanup_v1' or receipt.get('controller')!='agent':raise ValueError('Actual agent identity selection required')
    if receipt.get('evidence_ids')!=['image.left','image.right']:raise ValueError('Both stereo images must be reviewed')
    for item in receipt['protected_sources'].values():
        if sha(item['path'])!=item['sha256']:raise ValueError('Identity evidence changed')
    for side in ('left','right'):
        if receipt['image_sha256'][side]!=sha(case['inputs'][side]):raise ValueError('Identity receipt belongs to other images')
    masks={}
    for side,item in receipt['exclusions'].items():
        if sha(item['path'])!=item['sha256']:raise ValueError('Identity exclusion mask changed')
        mask=cv2.imread(item['path'],0)
        if mask is None or mask.shape!=cv2.imread(case['inputs'][side],0).shape:raise ValueError('Identity mask shape mismatch')
        masks[side]=mask>0
    return masks

def execute_identity_cleanup(tools,candidate,out,evidence):
    from thread_visibility_repairs_v5 import load_parts,save_parts,overlays,endpoint_table,load_inputs
    from geometry_repairs import project
    from complex_case_diagnostics import write_json
    if not evidence or evidence.get('controller')!='agent' or not {'image.left','image.right'}.issubset(set(evidence.get('evidence_ids',[]))):
        raise ValueError('Stereo-reviewed agent selection required for object cleanup')
    with np.load(tools.needle_path,allow_pickle=False) as d:needle={k:d[k].copy() for k in d.files}
    parts=load_parts(candidate['fragment_path']);_,images,masks,_,_,K,b=load_inputs(tools.case,tools.pipeline)
    runs=duplicate_runs(parts,needle,K,b,masks['left'].shape)
    if not runs:raise ValueError('No confirmed long stereo/3D needle duplicate; preserve crossings')
    newparts=trim_parts(parts,runs);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    exclusions={};receipt=dict(schema='needle_thread_identity_cleanup_v1',controller='agent',evidence_ids=['image.left','image.right'],
        reason=evidence.get('reason'),runs=runs,image_sha256={s:sha(tools.case['inputs'][s]) for s in ('left','right')},
        protected_sources={k:dict(path=str(p),sha256=sha(p)) for k,p in dict(source_thread=candidate['fragment_path'],source_needle=tools.needle_path).items()},
        inferred_object_assignment=True,independent_accuracy_measured=False)
    for side in ('left','right'):
        removed=np.zeros(masks[side].shape,'uint8')
        for run in runs:
            points=parts[run['fragment']]['points_m'][run['start']:run['end_exclusive']]
            xy=project(points,K,b if side=='right' else 0);cv2.polylines(removed,[np.rint(xy).astype('int32')],False,1,3)
        original=cv2.imread(tools.original['inputs']['thread_'+side+'_mask'],0)>0
        removed=(removed>0)&original;path=out/(side+'_needle_duplicate_exclusion.png');cv2.imwrite(str(path),removed.astype('uint8')*255)
        exclusions[side]=dict(path=str(path.resolve()),sha256=sha(path))
        updated=(masks[side]&~removed);path=out/(side+'_thread_mask.png');cv2.imwrite(str(path),updated.astype('uint8')*255)
        tools.case['inputs']['thread_'+side+'_mask']=str(path.resolve())
    receipt['exclusions']=exclusions;write_json(out/'identity_receipt.json',receipt)
    tools.case['inputs']['object_identity_receipt']=str((out/'identity_receipt.json').resolve())
    tools.version+=1;tools.working_path=out/'derived_case.json';write_json(tools.working_path,tools.case)
    tools.register_derived({str(p):str(p) for p in out.iterdir() if p.is_file()})
    save_parts(out/'fragments.npz',newparts);overlays(newparts,images,K,b,out)
    receipt['derived_fragment_sha256']=sha(out/'fragments.npz');write_json(out/'identity_receipt.json',receipt)
    tools.register_derived({str(out/'identity_receipt.json'):str(out/'identity_receipt.json')})
    report=dict(candidate['report'],schema='needle_identity_cleanup_v1',fragment_count=len(newparts),
        endpoints=endpoint_table(newparts,K,b,masks['left'].shape),object_identity_cleanup=str(out/'identity_receipt.json'))
    proposal=tools.package(out/'fragments.npz',report,out);proposal=tools.audit_geometry(proposal)
    return proposal,dict(operation='needle_duplicate_trim',identity_receipt=str(out/'identity_receipt.json'),original_geometry_preserved=True)
