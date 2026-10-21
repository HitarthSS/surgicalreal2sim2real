"""Shared fixed-size stereo needle hypotheses; no scene-specific rules.

The nominal catalog is an experimental prior, not an assertion of the needle's
physical specification. All sources stay immutable. Hidden material remains
inferred even when a stereo-consistent candidate is selected.
"""
from pathlib import Path
import hashlib,json,math,time
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from geometry_repairs import project,mask_points,_circle2d,_angle_interval
from rigid_needle_contract import from_pose,audit as audit_rigid

SIDES=('left','right')
CATALOG={'radius_mm':[4.,6.,8.,10.,12.,15.,18.],
         'span_degrees':[90.,135.,180.],
         'tube_radius_m':.00035,'actual_needle_specification_known':False}
LIMITS={'visible_tolerance_px':4.,'minimum_observed_coverage':.80,
        'maximum_visible_coverage_loss':.05,'minimum_explained_fraction':.80,
        'max_connector_m':.004,'max_depth_relative_difference':.25,
        'minimum_tangent_cosine':.25,'maximum_background_fraction_both':.10}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load_npz(path):
    if not path or not Path(path).is_file():return None
    with np.load(path,allow_pickle=False) as d:return {k:d[k].copy() for k in d.files}
def sample(mask,xy):
    h,w=mask.shape;inside=np.isfinite(xy).all(1)&(xy[:,0]>=0)&(xy[:,0]<=w-1)&(xy[:,1]>=0)&(xy[:,1]<=h-1)
    result=np.zeros(len(xy),bool);ij=np.rint(xy[inside]).astype(int)
    result[inside]=mask[ij[:,1],ij[:,0]]
    return result,inside
def distances(mask,xy):
    dt=cv2.distanceTransform((~mask).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    return sample_distances(dt,xy)
def sample_distances(dt,xy):
    return cv2.remap(dt,xy[:,0].astype('float32')[None],xy[:,1].astype('float32')[None],cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,borderValue=1e6).ravel()

def distance_fields(masks,thread):
    return {kind:{s:cv2.distanceTransform((~v).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
                  for s,v in values.items()}
            for kind,values in [('needle',masks),('connector',{s:masks[s]|thread[s] for s in SIDES})]}

def inputs(case):
    paths=case['inputs'];images={s:cv2.imread(paths[s]) for s in SIDES}
    if any(v is None for v in images.values()):raise ValueError('Both original stereo images required')
    shape=images['left'].shape[:2]
    if images['right'].shape[:2]!=shape:raise ValueError('Stereo image size mismatch')
    def mask(path):
        v=cv2.imread(str(path),0)
        if v is None or v.shape!=shape:raise ValueError('Missing or mismatched mask: '+str(path))
        return v>0
    masks={s:mask(paths['needle_'+s+'_mask']) for s in SIDES}
    thread={s:mask(paths['thread_'+s+'_mask']) for s in SIDES}
    occ={s:mask(paths[s+'_occluder']) if paths.get(s+'_occluder') else np.zeros(shape,bool) for s in SIDES}
    K=np.loadtxt(paths['K'],delimiter=',');depth=np.squeeze(np.load(paths['depth_npy'],allow_pickle=False));b=float(case['baseline_m'])
    if depth.shape!=shape or not np.isfinite(K).all() or not 0<b<.1:raise ValueError('Invalid calibration or depth dimensions')
    return images,masks,thread,occ,K,depth,b

def thread_endpoints(parts):
    rows=[]
    for f,part in enumerate(parts):
        p=np.asarray(part['points_m'],float)
        if p.ndim!=2 or p.shape[1]!=3 or len(p)<3 or not np.isfinite(p).all():raise ValueError('Invalid thread centerline')
        for end,i,j in [(0,0,min(4,len(p)-1)),(1,len(p)-1,max(0,len(p)-5))]:
            tangent=p[i]-p[j];norm=np.linalg.norm(tangent)
            if norm<1e-9:continue
            rows.append(dict(id=f'f{f}.e{end}',fragment=f,index=i,end=end,point=p[i].tolist(),tangent=(tangent/norm).tolist()))
    return rows

def evaluate(points,masks,occ,thread,K,b,endpoints,baseline_views=None,fields=None):
    """Visible fit, explicit hidden support, depth and a unique physical-end pair."""
    points=np.asarray(points,float);views={};supported=[];explained=[];inside=[]
    for s in SIDES:
        xy=project(points,K,b if s=='right' else 0.)
        d=sample_distances(fields['needle'][s],xy) if fields else distances(masks[s],xy)
        visible=d<=LIMITS['visible_tolerance_px'];hidden,in_frame=sample(occ[s],xy)
        observed=mask_points(masks[s],256)
        coverage=float((cKDTree(xy).query(observed)[0]<=LIMITS['visible_tolerance_px']).mean()) if len(observed) else 0.
        explain=(visible|hidden)&in_frame
        views[s]=dict(observed_coverage=coverage,visible_fraction=float(visible.mean()),
                      occluded_fraction=float((hidden&~visible).mean()),explained_fraction=float(explain.mean()))
        supported.append(d<=1.5);explained.append(explain);inside.append(in_frame)
    blockers=[]
    if not np.isfinite(points).all() or not ((points[:,2]>=.03)&(points[:,2]<=.5)).all():blockers.append('needle_depth_outside_working_range')
    for s,v in views.items():
        if v['observed_coverage']<LIMITS['minimum_observed_coverage']:blockers.append(s+'_visible_needle_not_explained')
        if baseline_views and v['observed_coverage']+LIMITS['maximum_visible_coverage_loss']<baseline_views[s]['observed_coverage']:
            blockers.append(s+'_visible_fit_worse_than_preserved_baseline')
    background=~(explained[0]|explained[1]);background_fraction=float(background.mean())
    if background_fraction>LIMITS['maximum_background_fraction_both']:blockers.append('hidden_arc_crosses_background_in_both_views')
    if not (inside[0]|inside[1]).all():blockers.append('needle_exits_both_frames')
    shape_blockers=list(blockers)
    pairs=[]
    for row in endpoints:
        p=np.asarray(row['point']);t=np.asarray(row['tangent'])
        for end,ni in [(0,0),(1,len(points)-1)]:
            delta=points[ni]-p;gap=float(np.linalg.norm(delta));connector=np.linspace(p,points[ni],33)
            cos=float(t@delta/max(gap,1e-12));fail=[];connection_views={}
            if gap>LIMITS['max_connector_m']:fail.append('connector_exceeds_existing_4mm_limit')
            if abs(delta[2])/min(p[2],points[ni,2])>LIMITS['max_depth_relative_difference']:fail.append('endpoint_depth_disagreement')
            if gap>.001 and cos<LIMITS['minimum_tangent_cosine']:fail.append('thread_tangent_points_away_from_needle')
            for s in SIDES:
                xy=project(connector,K,b if s=='right' else 0.)
                support=(sample_distances(fields['connector'][s],xy) if fields else distances(thread[s]|masks[s],xy))<=4.
                hidden,in_frame=sample(occ[s],xy)
                connection_views[s]=float(((support|hidden)&in_frame).mean())
            if max(connection_views.values())<LIMITS['minimum_explained_fraction']:fail.append('attachment_crosses_unsupported_background')
            pairs.append(dict(thread_endpoint_id=row['id'],fragment=row['fragment'],thread_endpoint_index=row['index'],
                needle_end=end,needle_endpoint_index=ni,distance_m=gap,tangent_cosine=cos,
                connector_explained_fraction=connection_views,blockers=fail))
    near=[p for p in pairs if p['distance_m']<=LIMITS['max_connector_m']]
    eligible=[p for p in near if not p['blockers']]
    if len(near)!=1:blockers.append('missing_or_ambiguous_endpoint_pair')
    elif not eligible:blockers.extend(near[0]['blockers'])
    connection=eligible[0] if len(near)==len(eligible)==1 else None
    if connection:
        # A nominal radius cannot justify intersecting a different strand.
        contact=np.asarray(endpoints[next(i for i,e in enumerate(endpoints) if e['id']==connection['thread_endpoint_id'])]['point'])
        nonlocal_ends=[np.asarray(e['point']) for e in endpoints if np.linalg.norm(np.asarray(e['point'])-contact)>.003]
        if nonlocal_ends and cKDTree(points).query(nonlocal_ends)[0].min()<.0005:
            blockers.append('needle_intersects_unrelated_thread_endpoint')
    # Unsupported samples are never relabeled observed because a nominal size fits.
    inferred=~(supported[0]&supported[1])
    hidden_fraction=float((~(supported[0]|supported[1])).mean())
    quality=min(v['observed_coverage'] for v in views.values())-2*background_fraction
    if connection:quality+=.15*(1-connection['distance_m']/.004)
    return dict(views=views,blockers=list(dict.fromkeys(blockers)),shape_blockers=shape_blockers,
                geometry_eligible=not shape_blockers,attachment_eligible=not blockers,connection=connection,
                all_endpoint_pairs=pairs,background_fraction_both=background_fraction,
                hidden_fraction_both=hidden_fraction,score=quality,eligible=not blockers),inferred

def _circle_seed(original,masks,K,depth):
    if original is not None:
        p=original['points_m'];c=original.get('fit_center_m');n=original.get('fit_normal')
        if c is None or n is None:
            c=p.mean(0);_,_,vt=np.linalg.svd(p-c);n=vt[-1]
        n=np.asarray(n,float);n/=np.linalg.norm(n);u=p[0]-c;u-=n*(u@n);u/=np.linalg.norm(u)
        v=np.cross(n,u);return np.r_[c,Rotation.from_matrix(np.c_[u,v,n]).as_rotvec()]
    observed=mask_points(masks['left'],256)
    center,_=_circle2d(observed)
    z=depth[masks['left']];z=z[np.isfinite(z)&(z>.03)&(z<.5)]
    if len(z)<5:raise ValueError('No metric needle depth anchors')
    z=float(np.median(z))
    return np.r_[(center[0]-K[0,2])*z/K[0,0],(center[1]-K[1,2])*z/K[1,1],z,[0.,0.,0.]]

def fixed_circle(radius,seed,observed,K,b,max_evaluations=70):
    angle=np.linspace(0,2*np.pi,360,endpoint=False);unit=np.c_[np.cos(angle),np.sin(angle),np.zeros(len(angle))]
    def curve(q):return q[:3]+radius*(unit@Rotation.from_rotvec(q[3:]).as_matrix().T)
    def residual(q):
        p=curve(q);values=[]
        for s in SIDES:
            xy=project(p,K,b if s=='right' else 0.)
            values.extend(cKDTree(xy).query(observed[s])[0]/3.)
        values.append((q[2]-seed[2])/.02)
        values.extend(np.maximum(.03-p[:,2],0.)/.001)
        return np.asarray(values)
    lower=np.r_[-.5,-.5,.03,[-2*np.pi]*3];upper=np.r_[.5,.5,.5,[2*np.pi]*3]
    candidates=[]
    for tilt in [0.,.55,-.55]:
        q=seed.copy();q[3]+=tilt;q=np.clip(q,lower+1e-8,upper-1e-8)
        fit=least_squares(residual,q,bounds=(lower,upper),loss='soft_l1',f_scale=2.,max_nfev=max_evaluations,
                          diff_step=1e-4,x_scale=np.r_[[.02]*3,[1.]*3])
        candidates.append((float(np.mean(residual(fit.x)**2)),fit))
    error,fit=min(candidates,key=lambda v:v[0]);p=curve(fit.x);angles=[]
    for s in SIDES:
        xy=project(p,K,b if s=='right' else 0.);indices=cKDTree(xy).query(observed[s])[1];angles.extend(angle[indices])
    start,span=_angle_interval(np.asarray(angles))
    return fit.x,start,span,dict(radius_m=radius,evaluations=fit.nfev,cost=error,optimizer_success=bool(fit.success),rotation_search_multistarts=3)

def render(images,points,parts,K,b,inferred,pair,label,out):
    canvases=[]
    for s in SIDES:
        image=images[s].copy()
        for part in parts:
            xy=project(part['points_m'],K,b if s=='right' else 0.)
            cv2.polylines(image,[np.rint(xy).astype('int32')],False,(240,220,20),1)
        if points is not None:
            xy=project(points,K,b if s=='right' else 0.)
            for i in range(len(xy)-1):
                color=(220,50,220) if inferred[i] or inferred[i+1] else (30,230,30)
                cv2.line(image,tuple(np.rint(xy[i]).astype(int)),tuple(np.rint(xy[i+1]).astype(int)),color,2)
            for end,i in [(0,0),(1,len(xy)-1)]:
                q=tuple(np.rint(xy[i]).astype(int));cv2.circle(image,q,4,(0,210,255),1);cv2.putText(image,'N'+str(end),q,cv2.FONT_HERSHEY_SIMPLEX,.4,(0,210,255),1)
            if pair:
                p=parts[pair['fragment']]['points_m'][pair['thread_endpoint_index']]
                line=project(np.vstack([p,points[pair['needle_endpoint_index']]]),K,b if s=='right' else 0.)
                cv2.polylines(image,[np.rint(line).astype('int32')],False,(0,170,255),2)
        image=cv2.resize(image,(480,360));canvases.append(image)
    row=np.vstack([np.zeros((32,960,3),np.uint8),np.hstack(canvases)])
    cv2.putText(row,label[:130],(8,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
    cv2.imwrite(str(out),row);return row

def build_catalog(case,needle_path,parts,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);began=time.perf_counter()
    images,masks,thread,occ,K,depth,b=inputs(case)
    fields=distance_fields(masks,thread)
    protected={k:dict(path=v,sha256=sha(v)) for k,v in case['inputs'].items()}
    if needle_path and Path(needle_path).is_file():protected['source_needle']=dict(path=str(needle_path),sha256=sha(needle_path))
    original=load_npz(needle_path);ends=thread_endpoints(parts);baseline=None
    if original is not None:baseline,_=evaluate(original['points_m'],masks,occ,thread,K,b,ends,fields=fields)
    observed={s:mask_points(masks[s],192) for s in SIDES}
    if min(map(len,observed.values()))<6:raise ValueError('Fewer than six needle skeleton samples in a stereo view; segmentation repair required')
    seed=_circle_seed(original,masks,K,depth);rows=[];circle_reports=[]
    baseline_contract=audit_rigid(original,CATALOG)
    if baseline is not None:baseline['rigid_contract']=baseline_contract
    try:image_seed=_circle_seed(None,masks,K,depth)
    except ValueError:image_seed=seed.copy()
    for radius_mm in CATALOG['radius_mm']:
        fits=[fixed_circle(radius_mm/1000.,initial,observed,K,b) for initial in [seed,image_seed]]
        q,start,visible_span,fit=min(fits,key=lambda v:v[3]['cost'])
        circle_reports.append(fit);R=Rotation.from_rotvec(q[3:]).as_matrix()
        for span_deg in CATALOG['span_degrees']:
            span=np.deg2rad(span_deg)
            # Noisy/merged masks may cover more angle than a rigid template.
            # Slide that same fixed arc through the observed interval, then let
            # stereo coverage reject bad fits; never stretch the template.
            extra=span-visible_span
            for extension in [0.,.25,.5,.75,1.]:
                theta=np.linspace(start-extension*extra,start-extension*extra+span,128)
                artifact=from_pose(radius_mm/1000.,span_deg,R,q[:3],start-extension*extra)
                p=artifact['points_m']
                report,inferred=evaluate(p,masks,occ,thread,K,b,ends,baseline['views'] if baseline and baseline_contract['usable'] else None,fields=fields)
                contract=audit_rigid(artifact,CATALOG,require_pose=True)
                assert contract['usable'],contract
                report['rigid_contract']=contract
                artifact.update(inferred=inferred,visible_stereo_support=~inferred,reliability=(~inferred).astype(float),tube_radius_m=np.asarray(CATALOG['tube_radius_m']),size_prior_used=np.asarray(True))
                cid=f'r{radius_mm:g}_s{span_deg:g}_e{extension:g}'
                path=out/(cid+'.npz')
                np.savez_compressed(path,**artifact)
                rows.append(dict(candidate_id=cid,needle_path=str(path.resolve()),needle_sha256=sha(path),radius_mm=radius_mm,
                                 span_degrees=span_deg,extension_fraction=extension,report=report))
    from rigid_stereo_refine import refine_arc_rows
    rows.extend(refine_arc_rows(case,original,parts,out,rows))
    # Show the best available explanation for every radius, even when blocked.
    shortlist=[]
    for radius in CATALOG['radius_mm']:
        group=[v for v in rows if v['radius_mm']==radius]
        if group:shortlist.append(max(group,key=lambda v:(v['report']['geometry_eligible'],v['report']['score'])))
    allrows=[]
    if original is not None:
        flags=original.get('inferred',np.ones(len(original['points_m']),bool))
        allrows.append(render(images,original['points_m'],parts,K,b,flags,baseline.get('connection'),'LEGACY REFERENCE (not a fixed-template selection)' if not baseline_contract['usable'] else 'VALID RIGID BASELINE',out/'baseline.jpg'))
    for row in shortlist:
        artifact=load_npz(row['needle_path']);r=row['report'];name=row['candidate_id']+'.jpg'
        allrows.append(render(images,artifact['points_m'],parts,K,b,artifact['inferred'],r['connection'],
            row['candidate_id']+' '+('rigid needle fits; '+('attachment eligible' if r['eligible'] else 'attachment unresolved') if r['geometry_eligible'] else ','.join(r['shape_blockers'])[:70]),out/name))
        row['overlay']=str((out/name).resolve())
    if allrows:cv2.imwrite(str(out/'catalog.jpg'),np.vstack(allrows))
    result=dict(schema='fixed_stereo_needle_catalog_v2',catalog=CATALOG,limits=LIMITS,baseline=baseline,
        original_needle_path=str(needle_path) if needle_path else None,shortlist=shortlist,candidates=rows,circle_fits=circle_reports,
        evidence=protected,catalog_image=str((out/'catalog.jpg').resolve()),full_stereo_views_uncropped=True,
        runtime_seconds=time.perf_counter()-began,source_geometry_modified=False,independent_accuracy_measured=False,physics_validated=False)
    result['state_version']=hashlib.sha256(json.dumps(dict(catalog=CATALOG,evidence=protected,candidates=[(x['candidate_id'],x['needle_sha256']) for x in rows]),sort_keys=True).encode()).hexdigest()
    save(out/'catalog.json',result)
    return result

def validate_selection(raw,catalog):
    from needle_thread_connection_tool import decode_attachment_response
    if isinstance(raw,str):raw=json.loads(decode_attachment_response(raw)[0])
    if set(raw)!={'state_version','operation','candidate_id','physical_swage_identified','evidence_ids','reason'}:raise ValueError('Exact selection fields required')
    if raw['state_version']!=catalog['state_version']:raise ValueError('Stale needle catalog')
    if not isinstance(raw['reason'],str) or not 1<=len(raw['reason'])<=500:raise ValueError('Reason must be 1-500 characters')
    if not isinstance(raw['evidence_ids'],list) or set(raw['evidence_ids'])!={'image.left','image.right','catalog'}:raise ValueError('Both stereo images and catalog required')
    if type(raw['physical_swage_identified']) is not bool:raise ValueError('Swage identity must be boolean')
    if raw['operation']=='keep_baseline':
        if raw['candidate_id'] is not None or raw['physical_swage_identified']:raise ValueError('Keep-baseline cannot select or certify swage')
        return None
    if raw['operation']!='select_fixed_needle':raise ValueError('Explicit rigid needle selection required')
    choices=[v for v in catalog['shortlist'] if v['candidate_id']==raw['candidate_id']]
    if len(choices)!=1 or not choices[0]['report']['geometry_eligible']:raise ValueError('Unknown or numerically blocked rigid needle candidate')
    if raw['physical_swage_identified'] and not choices[0]['report']['attachment_eligible']:raise ValueError('Attachment has not passed its separate checks')
    artifact=load_npz(choices[0]['needle_path'])
    if not audit_rigid(artifact,CATALOG,require_pose=True)['usable']:raise ValueError('Selected needle violates the rigid-template contract')
    row=choices[0]
    if sha(row['needle_path'])!=row['needle_sha256']:raise ValueError('Candidate geometry changed')
    for value in catalog['evidence'].values():
        if sha(value['path'])!=value['sha256']:raise ValueError('Bound source evidence changed')
    return row

def selection_messages(case,catalog):
    state={k:catalog[k] for k in ['state_version','catalog','limits','baseline']}
    state['candidates']=[{k:v for k,v in r.items() if k not in ('needle_path','overlay')} for r in catalog['shortlist']]
    rules=('You are the needle/thread fitting specialist called by the coded scene orchestrator. '
      'A shared CPU tool has run the same fixed radii/spans against both stereo views. Compare ALL candidate sizes. '
      'Choose a sensible needle and physical thread swage, especially where a tool hides the junction. '
      'An inferred hidden needle is allowed; perfect visibility is not required. Green=stereo-supported needle; '
      'magenta=inferred needle; cyan=preserved thread; yellow N0/N1=needle ends; orange=proposed connector. '
      'Do not choose the sharp free tip as the swage or fit the thread as a needle. '
      'Select only geometry_eligible candidates. Attachment eligibility is a SEPARATE decision: '
      'physical_swage_identified may be true only when attachment_eligible is true and both views support that swage; '
      'otherwise select the rigid needle with false swage, without constructing an attachment. '
      'Needle geometry cannot bend, change radius or move endpoints independently. Its numeric plausibility is not independent confirmation. '
      'If no fixed-size hypothesis is sensible, retain the complete preserved baseline, not an empty scene. '
      'Return only JSON with state_version, operation(select_fixed_needle or keep_baseline), candidate_id(ID or null), '
      'physical_swage_identified(boolean), evidence_ids:[image.left,image.right,catalog], reason(1-500 chars). '
      'Keep-baseline uses null ID and false swage. Sizes are experimental hypotheses, not verified specifications. STATE:'+json.dumps(state,separators=(',',':')))
    content=[dict(type='text',text=rules)]
    for s in SIDES:content.extend([dict(type='text',text='image.'+s+' complete original stereo'),dict(type='image',url=case['inputs'][s])])
    content.extend([dict(type='text',text='catalog: full-field stereo overlays for every fixed radius; baseline first'),dict(type='image',url=catalog['catalog_image'])])
    return [dict(role='user',content=content)]
