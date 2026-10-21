"""Join numerical duplicate endpoints without inventing a sampled hidden bridge.

A crossing with an abrupt route is deliberately ineligible. It needs a separate
topology repair; proximity alone cannot establish material connectivity.
"""
import numpy as np

def project(p,K,b=0):
    p=np.asarray(p).copy();p[:,0]-=b;q=p@K.T;return q[:,:2]/q[:,2,None]

def outward(part,end,distance=.002):
    points=part['points_m'] if end==0 else part['points_m'][::-1]
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    index=min(int(np.searchsorted(arc,distance)),len(points)-1)
    v=points[index]-points[0];norm=np.linalg.norm(v)
    return v/norm if norm>1e-10 else None

def stitch(parts,first,second,K,baseline):
    a,ea=first;b,eb=second
    if a==b:return None,'same_fragment'
    pa=parts[a]['points_m'][0 if ea==0 else -1];pb=parts[b]['points_m'][0 if eb==0 else -1]
    gap=float(np.linalg.norm(pa-pb))
    if gap>1e-6:return None,'not_numerically_coincident'
    for offset in (0.,baseline):
        xy=project(np.vstack([pa,pb]),K,offset)
        if np.linalg.norm(xy[0]-xy[1])>.01:return None,'stereo_endpoints_differ'
    # Other endpoints in the same cluster make this a topology selection.
    for f,part in enumerate(parts):
        for end in (0,1):
            if (f,end) in (first,second):continue
            p=part['points_m'][0 if end==0 else -1]
            if min(np.linalg.norm(p-pa),np.linalg.norm(p-pb))<=1e-6:return None,'competing_coincident_endpoint'
    u=outward(parts[a],ea);v=outward(parts[b],eb)
    if u is None or v is None:return None,'degenerate_tangent'
    turn=float(np.arccos(np.clip(-u@v,-1.,1.)))
    if turn>.2:return None,'crossing_path_order_requires_repair'
    left={k:(x[::-1] if ea==0 else x) for k,x in parts[a].items()}
    right={k:(x[::-1] if eb==1 else x) for k,x in parts[b].items()}
    keys=('points_m','depth_inferred','geometry_inferred','observed_left','observed_right')
    joined={k:np.concatenate([left[k],right[k]]) for k in keys}
    joined['reference_xy']=project(joined['points_m'],K,baseline)
    result=[joined if f==min(a,b) else part for f,part in enumerate(parts) if f!=max(a,b)]
    return result,dict(operation='coincident_endpoint_stitch',endpoint_gap_m=gap,neighborhood_turn_rad=turn,
        new_curve_samples=0,source_points_preserved_exactly=True,topology_is_hypothesis=True,
        source_fragment_count=len(parts),fragment_count=len(result),joined_source_fragments=[a,b])

def proposals(parts,K,baseline):
    result=[]
    for a in range(len(parts)):
        for b in range(a+1,len(parts)):
            for ea in (0,1):
                for eb in (0,1):
                    output,report=stitch(parts,(a,ea),(b,eb),K,baseline)
                    if output is not None:result.append(dict(first=[a,ea],second=[b,eb],report=report))
    return result
