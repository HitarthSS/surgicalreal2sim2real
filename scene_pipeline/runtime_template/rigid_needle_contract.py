"""A needle is one catalog arc and one SE(3) pose, never editable vertices."""
import numpy as np

def template_points(radius_m,span_degrees,count=128):
    if not np.isfinite([radius_m,span_degrees]).all() or radius_m<=0 or not 0<span_degrees<360 or count<8:
        raise ValueError('Invalid fixed needle template')
    t=np.linspace(0,np.deg2rad(span_degrees),count)
    return np.c_[radius_m*np.cos(t),radius_m*np.sin(t),np.zeros(count)]

def from_pose(radius_m,span_degrees,rotation,translation,start_angle=0.,count=128):
    rotation=np.asarray(rotation,float);translation=np.asarray(translation,float)
    if rotation.shape!=(3,3) or translation.shape!=(3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError('Finite rigid pose required')
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-9) or not np.isclose(np.linalg.det(rotation),1.,atol=1e-9):
        raise ValueError('Needle pose may not scale, shear or reflect the template')
    c,s=np.cos(start_angle),np.sin(start_angle)
    rotation=rotation@np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    local=template_points(radius_m,span_degrees,count);points=local@rotation.T+translation
    T=np.eye(4);T[:3,:3]=rotation;T[:3,3]=translation
    return dict(points_m=points,fit_center_m=translation.copy(),fit_normal=rotation[:,2].copy(),
        fit_radius_m=np.asarray(radius_m),template_points_m=local,T_camera_from_template=T,
        template_radius_m=np.asarray(radius_m),template_span_degrees=np.asarray(span_degrees),
        nominal_span_degrees=np.asarray(span_degrees),template_id=np.asarray(f'r{radius_m*1000:g}_s{span_degrees:g}'),
        needle_representation=np.asarray('rigid_catalog_arc_v2'))

def audit(artifact,catalog,require_pose=False):
    if artifact is None:return dict(circle_valid=False,catalog_valid=False,usable=False,failures=['needle_missing'])
    failures=[];p=np.asarray(artifact.get('points_m',[]),float)
    if p.ndim!=2 or p.shape[1:]!=(3,) or len(p)<8 or not np.isfinite(p).all():
        return dict(circle_valid=False,catalog_valid=False,usable=False,failures=['invalid_centerline'])
    c=artifact.get('fit_center_m');n=artifact.get('fit_normal');r=artifact.get('fit_radius_m')
    if c is None or n is None or r is None:
        center=p.mean(0);_,_,vt=np.linalg.svd(p-center);n=vt[-1];u=vt[0];v=np.cross(n,u)
        xy=np.c_[(p-center)@u,(p-center)@v]
        a,b,d=np.linalg.lstsq(np.c_[2*xy,np.ones(len(p))],(xy*xy).sum(1),rcond=None)[0]
        c=center+a*u+b*v;r=np.sqrt(max(0.,d+a*a+b*b))
    c=np.asarray(c,float);n=np.asarray(n,float);r=float(r)
    if c.shape!=(3,) or n.shape!=(3,) or not np.isfinite(np.r_[c,n,r]).all() or r<=0 or np.linalg.norm(n)<1e-12:
        return dict(circle_valid=False,catalog_valid=False,usable=False,failures=['invalid_circle_metadata'])
    n=n/np.linalg.norm(n);plane=np.abs((p-c)@n);radial=np.abs(np.linalg.norm(p-c,axis=1)-r)
    u=p[0]-c;u-=n*(u@n)
    if np.linalg.norm(u)<1e-12:return dict(circle_valid=False,catalog_valid=False,usable=False,failures=['degenerate_arc_basis'])
    u/=np.linalg.norm(u);v=np.cross(n,u)
    theta=np.unwrap(np.arctan2((p-c)@v,(p-c)@u));step=np.diff(theta);span=float(abs(np.degrees(theta[-1]-theta[0])))
    if plane.max()>1e-8:failures.append('needle_not_planar')
    if radial.max()>1e-8:failures.append('needle_radius_changes_along_curve')
    if not ((step>1e-10).all() or (step<-1e-10).all()):failures.append('needle_order_reverses_or_repeats')
    if span>=360-1e-5:failures.append('needle_wraps_or_closes')
    circle_valid=not failures
    radius_match=any(np.isclose(r*1000,x,rtol=0,atol=1e-5) for x in catalog['radius_mm'])
    span_match=any(np.isclose(span,x,rtol=0,atol=1e-4) for x in catalog['span_degrees'])
    if not radius_match:failures.append('radius_not_a_fixed_catalog_size')
    if not span_match:failures.append('span_not_a_fixed_catalog_arc')
    pose_valid=False
    if 'T_camera_from_template' in artifact and 'template_points_m' in artifact:
        T=np.asarray(artifact['T_camera_from_template']);local=np.asarray(artifact['template_points_m'])
        pose_valid=bool(T.shape==(4,4) and local.shape==p.shape and np.isfinite(T).all()
            and np.allclose(T[3],[0,0,0,1],atol=1e-10)
            and np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-9)
            and np.isclose(np.linalg.det(T[:3,:3]),1.,atol=1e-9)
            and np.allclose(local,template_points(r,span,len(p)),atol=1e-8)
            and np.allclose(p,local@T[:3,:3].T+T[:3,3],atol=1e-9))
        if not pose_valid:failures.append('vertices_do_not_match_one_rigid_template_pose')
        identity_valid=(str(artifact.get('template_id',''))==f'r{r*1000:g}_s{span:g}'
            and np.isclose(float(artifact.get('template_radius_m',-1)),r,atol=1e-10)
            and np.isclose(float(artifact.get('template_span_degrees',-1)),span,atol=1e-8))
        if not identity_valid:failures.append('template_identity_does_not_match_geometry')
    elif require_pose:failures.append('rigid_template_pose_metadata_missing')
    return dict(circle_valid=circle_valid,catalog_valid=bool(circle_valid and radius_match and span_match),
        rigid_pose_valid=pose_valid,usable=not failures,failures=failures,radius_mm=r*1000,span_degrees=span,
        maximum_plane_error_m=float(plane.max()),maximum_radius_error_m=float(radial.max()),
        depth_range_m=float(np.ptp(p[:,2])),nominal_dimensions_are_hypotheses=True)
