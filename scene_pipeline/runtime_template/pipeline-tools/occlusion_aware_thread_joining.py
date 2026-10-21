"""Experimental, conservative joining of thread fragments behind supplied occluders.

No changes to the legacy connector. All inferred edges carry provenance.
"""
from dataclasses import dataclass, asdict
import cv2
import networkx as nx
import numpy as np
import reconstruct_thread_from_files as base


@dataclass(frozen=True)
class JoinSettings:
    max_gap_px: float = 240.0
    max_gap_m: float = 0.04
    min_occluder_fraction: float = 0.55
    min_direction_cosine: float = 0.25
    max_depth_relative_difference: float = 0.25
    right_tolerance_px: float = 8.0
    tangent_length_px: float = 15.0
    occluder_padding_px: int = 3
    ambiguity_margin: float = 0.10
    min_depth_m: float = 0.03
    max_depth_m: float = 0.50

    def validate(self):
        values = np.array(list(asdict(self).values()), dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Joining settings must be finite')
        for name in ('max_gap_px', 'max_gap_m', 'right_tolerance_px', 'tangent_length_px'):
            if getattr(self, name) <= 0: raise ValueError(f'{name} must be positive')
        if not 0 <= self.min_occluder_fraction <= 1: raise ValueError('Invalid occluder fraction')
        if not -1 <= self.min_direction_cosine <= 1: raise ValueError('Invalid direction cosine')
        if not 0 <= self.max_depth_relative_difference <= 1: raise ValueError('Invalid depth difference')
        if self.occluder_padding_px < 0 or self.ambiguity_margin < 0: raise ValueError('Invalid padding/margin')
        if not 0 < self.min_depth_m < self.max_depth_m: raise ValueError('Invalid depth range')


def mask_at(mask, rc):
    indices = np.rint(rc).astype(int)
    valid = (indices[:, 0] >= 0) & (indices[:, 0] < mask.shape[0]) & (indices[:, 1] >= 0) & (indices[:, 1] < mask.shape[1])
    out = np.zeros(len(rc), dtype=bool)
    out[valid] = mask[indices[valid, 0], indices[valid, 1]]
    return out


def padded(mask, radius):
    if radius == 0: return mask.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    return cv2.dilate(mask.astype('uint8'), kernel).astype(bool)


def endpoint_direction(graph, endpoint, length):
    """Outward tangent from visible fragment only; stop at any branch."""
    current = endpoint; previous = None; traveled = 0.
    while traveled < length:
        neighbors = [n for n in graph.neighbors(current) if n != previous]
        if len(neighbors) != 1: break
        following = neighbors[0]
        traveled += graph.edges[current, following]['weight']
        previous, current = current, following
    delta = np.asarray(endpoint, float) - current
    norm = np.linalg.norm(delta)
    return None if norm < 3 else delta/norm


def hermite(a, b, outward_a, outward_b):
    distance = np.linalg.norm(b-a)
    t = np.linspace(0, 1, max(3, int(np.ceil(distance*2))+1))[:, None]
    # Arrival direction is the opposite of the second outward tangent.
    m0, m1 = outward_a*distance*.65, -outward_b*distance*.65
    return (2*t**3-3*t**2+1)*a + (t**3-2*t**2+t)*m0 + (-2*t**3+3*t**2)*b + (t**3-t**2)*m1


def propose_joins(graph, left_mask, right_mask, depth, K, baseline, left_occluder,
                  right_occluder=None, settings=JoinSettings()):
    settings.validate()
    if baseline <= 0 or not np.isfinite(baseline): raise ValueError('Baseline must be positive and finite')
    if not np.any(left_occluder): raise ValueError('A nonempty left occluder mask is required')
    occ = padded(left_occluder, settings.occluder_padding_px)
    right_occ = None if right_occluder is None else padded(right_occluder, settings.occluder_padding_px)
    components = list(nx.connected_components(graph)); candidates = []
    endpoints = []
    for ci, nodes in enumerate(components):
        for node in sorted(nodes):
            if graph.degree[node] == 1:
                direction = endpoint_direction(graph, node, settings.tangent_length_px)
                if direction is not None: endpoints.append((ci, node, direction))
    for i, (ci, a, ta) in enumerate(endpoints):
        for cj, b, tb in endpoints[i+1:]:
            if ci == cj: continue
            a_arr, b_arr = np.asarray(a, float), np.asarray(b, float)
            gap = float(np.linalg.norm(b_arr-a_arr))
            record = {'from_rc':list(map(int,a)), 'to_rc':list(map(int,b)), 'distance_px':gap, 'rejections':[]}
            candidates.append(record)
            if gap == 0 or gap > settings.max_gap_px:
                record['rejections'].append('pixel_safety_cap'); continue
            delta = (b_arr-a_arr)/gap
            align = [float(ta@delta), float(tb@-delta)]
            record['direction_cosines'] = align
            if min(align) < settings.min_direction_cosine:
                record['rejections'].append('endpoint_directions'); continue
            curve = hermite(a_arr, b_arr, ta, tb)
            interior = curve[max(1,int(.05*len(curve))):min(len(curve)-1,int(.95*len(curve)))]
            coverage = float(np.mean(mask_at(occ, interior)))
            record['occluder_fraction'] = coverage
            if coverage < settings.min_occluder_fraction:
                record['rejections'].append('gap_not_explained_by_occluder'); continue
            z, _ = base.sample_depth(depth, left_mask, np.array([a,b],float), 5, settings.min_depth_m, settings.max_depth_m)
            if not np.isfinite(z).all():
                record['rejections'].append('missing_endpoint_depth'); continue
            points = base.backproject(np.array([a,b],float), z, K)
            metric_gap = float(np.linalg.norm(points[1]-points[0]))
            record['endpoint_depth_m'] = z.tolist(); record['distance_m'] = metric_gap
            if metric_gap > settings.max_gap_m or abs(z[1]-z[0])/min(z) > settings.max_depth_relative_difference:
                record['rejections'].append('incompatible_metric_depth'); continue
            right_dist, _ = base.right_mask_distances(np.array([a,b],float), z, right_mask, K[0,0], baseline)
            record['right_endpoint_distance_px'] = right_dist.tolist()
            if np.max(right_dist) > settings.right_tolerance_px:
                record['rejections'].append('stereo_endpoint_mismatch'); continue
            # Stronger optional evidence if the right occluder has been segmented.
            if right_occ is not None:
                curve_z = np.linspace(z[0],z[1],len(curve))
                rd, rxy = base.right_mask_distances(curve, curve_z, right_mask, K[0,0], baseline)
                fraction = float(np.mean((rd <= settings.right_tolerance_px) | mask_at(right_occ, rxy[:,::-1])))
                record['right_gap_supported_fraction'] = fraction
                if fraction < .8:
                    record['rejections'].append('right_gap_unsupported'); continue
            record['score'] = float(.4*(1-coverage)+.3*(1-np.mean(align))+.2*metric_gap/settings.max_gap_m+.1*gap/settings.max_gap_px)
            record['_curve'] = curve
    feasible = [c for c in candidates if not c['rejections']]
    # Do not choose an arbitrary pair when another candidate for either endpoint is close.
    for c in feasible:
        competitors = [o for o in feasible if o is not c and any(p in (o['from_rc'],o['to_rc']) for p in (c['from_rc'],c['to_rc']))]
        if competitors and min(o['score'] for o in competitors)-c['score'] < settings.ambiguity_margin:
            c['rejections'].append('ambiguous_endpoint_match')
    uf = nx.utils.UnionFind(graph.nodes); used=set(); joins=[]
    for nodes in components:
        first=next(iter(nodes))
        for node in nodes: uf.union(first,node)
    for c in sorted(feasible,key=lambda c:c['score']):
        if c['rejections']: continue
        a,b=tuple(c['from_rc']),tuple(c['to_rc'])
        if a in used or b in used or uf[a]==uf[b]:
            c['rejections'].append('endpoint_already_used_or_cycle'); continue
        graph.add_edge(a,b,weight=float(np.linalg.norm(np.diff(c['_curve'],axis=0),axis=1).sum()),synthetic=True,curve_rc=c['_curve'])
        uf.union(a,b); used.update((a,b)); joins.append(c)
    public = [{k:v for k,v in c.items() if not k.startswith('_')} for c in candidates]
    return [{k:v for k,v in c.items() if not k.startswith('_')} for c in joins], public


def ordered_path(graph):
    if nx.number_connected_components(graph) != 1:
        raise RuntimeError('Unresolved fragments: refusing to silently drop smaller components')
    endpoints = [n for n in graph if graph.degree[n]==1]
    if len(endpoints)!=2:
        raise RuntimeError('Thread topology is ambiguous: expected exactly two terminal endpoints')
    nodes=nx.shortest_path(graph,*endpoints,weight='weight'); path=[np.array(nodes[0],float)]; inferred=[False]
    for a,b in zip(nodes[:-1],nodes[1:]):
        edge=graph.edges[a,b]
        curve=edge.get('curve_rc',np.array([a,b],float))
        if np.linalg.norm(curve[0]-a)>1e-6: curve=curve[::-1]
        path.extend(curve[1:]); inferred.extend([bool(edge.get('synthetic',False))]*(len(curve)-1))
    return np.array(path),np.array(inferred,bool)
