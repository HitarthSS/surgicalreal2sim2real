"""Canonicalize only roundoff in a derived shared joint; preserve source arrays."""
from copy import deepcopy
import numpy as np

def canonicalize(bundle,metadata,epsilon_m=1e-9):
 if not 0<epsilon_m<=1e-9:raise ValueError('Roundoff limit cannot be enlarged')
 out={k:np.asarray(v).copy() for k,v in bundle.items()};meta=deepcopy(metadata)
 ti=meta['thread_endpoint_index'];ni=meta['needle_endpoint_index']
 t=out['points_newton_m'];n=out['needle_points_newton_m'];tc=out['points_camera_m'];nc=out['needle_points_camera_m']
 if ti not in (0,len(t)-1) or ni not in (0,len(n)-1):raise ValueError('Only material endpoints')
 T=out['T_newton_from_camera'];expected=nc[ni]@T[:3,:3].T+T[:3,3]
 if np.linalg.norm(expected-n[ni])>1e-12:raise ValueError('Needle frames disagree; not roundoff')
 delta=float(np.linalg.norm(t[ti]-n[ni]));dc=float(np.linalg.norm(tc[ti]-nc[ni]))
 if not np.isfinite([delta,dc]).all() or max(delta,dc)>epsilon_m:raise ValueError('Attachment mismatch exceeds roundoff; rebuild connector')
 # This changes only the derived junction representation, never either source artifact.
 out.setdefault('source_points_camera_m',tc.copy());out.setdefault('source_needle_points_camera_m',nc.copy())
 out['points_newton_m'][ti]=n[ni];out['points_camera_m'][ti]=nc[ni]
 meta['endpoint_distance_m']=0.
 meta['shared_anchor_contract']=dict(exact_camera=True,exact_newton=True,roundoff_limit_m=epsilon_m,
   original_camera_delta_m=dc,original_newton_delta_m=delta,source_artifacts_modified=False)
 return out,meta
