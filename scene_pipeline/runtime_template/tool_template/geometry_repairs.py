"""Coverage-aware stereo geometry tools. No scene IDs, model calls or size template."""
from pathlib import Path
import itertools
import math
import cv2
import networkx as nx
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

SIDES = ("left", "right")


def project(points, K, baseline=0.):
    points = np.asarray(points, float)
    return np.c_[(points[:, 0]-baseline)*K[0, 0]/points[:, 2]+K[0, 2],
                 points[:, 1]*K[1, 1]/points[:, 2]+K[1, 2]]


def mask_points(mask, maximum=400):
    from skimage.morphology import skeletonize
    y, x = np.nonzero(skeletonize(np.asarray(mask, bool)))
    p = np.c_[x, y].astype(float)
    if len(p) > maximum:
        p = p[np.linspace(0, len(p)-1, maximum).astype(int)]
    return p


def curve_coverage(points, masks, K, baseline, tolerance=4.):
    """Measure both directions: a tiny supported arc cannot explain a full needle."""
    result = {}
    for side in SIDES:
        target = mask_points(masks[side])
        xy = project(points, K, baseline if side == "right" else 0.)
        h, w = masks[side].shape
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        if not len(target) or not len(xy) or not np.isfinite(xy).all():
            result[side] = dict(observed_coverage=0., curve_support=0., median_error_px=None)
            continue
        reverse = cKDTree(xy).query(target)[0]
        forward = cKDTree(target).query(xy)[0]
        result[side] = dict(observed_coverage=float((reverse <= tolerance).mean()),
                            curve_support=float(((forward <= tolerance) & inside).mean()),
                            median_error_px=float(np.median(np.r_[forward, reverse])))
    return result


def _circle2d(points):
    center = points.mean(0)
    p = points-center
    x, y = p.T
    a, b, c = np.linalg.lstsq(np.c_[2*x, 2*y, np.ones(len(p))], x*x+y*y, rcond=None)[0]
    radius = math.sqrt(max(c+a*a+b*b, 1e-8))
    return center+np.array([a, b]), radius


def _angle_interval(angles):
    values = np.sort(np.mod(angles, 2*np.pi))
    gaps = np.diff(np.r_[values, values[0]+2*np.pi])
    end = int(np.argmax(gaps))
    start = values[(end+1) % len(values)]
    return float(start), float(2*np.pi-gaps[end])


def fit_stereo_needle(masks, K, baseline, depth, original=None, max_evaluations=160):
    """Fit a free 3D circle to full two-view evidence, including reverse coverage.

    There is no nominal needle radius and no manual endpoint or scene-specific
    rule. Estimated depth initializes the search; stereo masks determine scale.
    Missing visible evidence requests perception repair instead of marking a
    tiny supported subarc complete.
    """
    observed = {s: mask_points(masks[s], 256) for s in SIDES}
    if min(map(len, observed.values())) < 12:
        raise ValueError("Need both visible needle masks; request segmentation repair")
    y, x = np.nonzero(masks["left"])
    z = depth[y, x]
    z = z[np.isfinite(z) & (z > .03) & (z < .5)]
    if len(z) < 5:
        raise ValueError("Need metric depth anchors for needle initialization")
    z0 = float(np.median(z))
    center, radius = _circle2d(observed["left"])
    t0, span = _angle_interval(np.arctan2(*(observed["left"]-center)[:, ::-1].T))
    seed = np.r_[(center[0]-K[0, 2])*z0/K[0, 0], (center[1]-K[1, 2])*z0/K[1, 1],
                 z0, radius*z0/K[0, 0], np.zeros(3), t0, span]
    seeds = [seed]
    if original is not None:
        points = np.asarray(original["points_m"])
        c = np.asarray(original["fit_center_m"])
        normal = np.asarray(original["fit_normal"])
        u = points[0]-c
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        v /= np.linalg.norm(v)
        rotation = Rotation.from_matrix(np.c_[u, v, normal]).as_rotvec()
        theta = np.unwrap(np.arctan2((points-c)@v, (points-c)@u))
        if theta[-1] < theta[0]:
            v = -v; normal = -normal
            rotation = Rotation.from_matrix(np.c_[u, v, normal]).as_rotvec()
            theta = -theta
        seeds.append(np.r_[c, float(original["fit_radius_m"]), rotation, theta[0], theta[-1]-theta[0]])
    trees = {s: cKDTree(p) for s, p in observed.items()}

    def curve(q, count=128):
        r = Rotation.from_rotvec(q[4:7]).as_matrix()
        angle = np.linspace(q[7], q[7]+q[8], count)
        return q[:3]+q[3]*(np.cos(angle)[:, None]*r[:, 0]+np.sin(angle)[:, None]*r[:, 1])

    def residual(q):
        points = curve(q)
        values = []
        for side in SIDES:
            xy = project(points, K, baseline if side == "right" else 0.)
            values.extend(trees[side].query(xy)[0]/3.)
            values.extend(cKDTree(xy).query(observed[side])[0]/3.)
        # A weak seed prior prevents a low-texture degeneracy; it cannot win
        # over the full left/right silhouettes by discarding most of the arc.
        values.append((q[2]-z0)/.04)
        values.extend(np.maximum(.03-points[:, 2], 0.)/.001)
        return np.asarray(values)

    lower = np.r_[-.5, -.5, .03, .0001, [-2*np.pi]*3, -4*np.pi, .2]
    upper = np.r_[.5, .5, .5, .2, [2*np.pi]*3, 4*np.pi, 2*np.pi-.02]
    alternatives = []
    for q in seeds:
        q = np.maximum(lower+1e-8, np.minimum(upper-1e-8, q))
        fit = least_squares(residual, q, bounds=(lower, upper), loss="soft_l1", f_scale=2.,
                            max_nfev=max_evaluations, diff_step=1e-4,
                            x_scale=np.r_[np.full(4, .02), np.ones(5)])
        points = curve(fit.x, 256)
        report = curve_coverage(points, masks, K, baseline)
        quality = min(min(v["observed_coverage"], v["curve_support"]) for v in report.values())
        alternatives.append((quality, -float(np.mean(residual(fit.x)**2)), fit, report))
    _, _, fit, report = max(alternatives, key=lambda x: x[:2])
    points = curve(fit.x, 64)
    rotation = Rotation.from_rotvec(fit.x[4:7]).as_matrix()
    dense = curve(fit.x, 256)
    strict = curve_coverage(dense, masks, K, baseline, 1.5)
    inferred = np.zeros(len(points), bool)
    for side in SIDES:
        inferred |= trees[side].query(project(points, K, baseline if side == "right" else 0.))[0] > 1.5
    full = bool(min(min(v["observed_coverage"], v["curve_support"]) for v in report.values()) >= .85
                and np.degrees(fit.x[8]) >= 45. and ((dense[:,2]>=.03)&(dense[:,2]<=.5)).all())
    artifact = dict(points_m=points, inferred=inferred, visible_stereo_support=~inferred,
                    reliability=(~inferred).astype(float), fit_center_m=fit.x[:3],
                    fit_normal=rotation[:, 2], fit_radius_m=fit.x[3], tube_radius_m=np.array(.00035))
    return artifact, dict(complete_visible_needle=bool(full), views=report, strict_views=strict,
                         radius_m=float(fit.x[3]), span_degrees=float(np.degrees(fit.x[8])),
                         evaluations=fit.nfev, nominal_radius_prior=False,
                         independent_accuracy_measured=False, physics_validated=False)


def trace_strands(mask, minimum_length=8., max_spur=2.):
    """Retain every strand; a skeleton crossing is not a material connection.

    Split uncertain/odd junctions instead of dropping the entire reconstruction.
    Pair only opposed tangents with a unique smooth continuation. Distinct
    channels can share image coordinates without sharing a physical vertex.
    """
    from skimage.morphology import skeletonize
    pixels = set(map(tuple, np.argwhere(skeletonize(np.asarray(mask, bool)))))
    graph = nx.Graph()
    graph.add_nodes_from(pixels)
    for a in sorted(pixels):
        for dy, dx in [(0, 1), (1, -1), (1, 0), (1, 1)]:
            b = (a[0]+dy, a[1]+dx)
            if b in pixels:
                if dy and dx and ((a[0], b[1]) in pixels or (b[0], a[1]) in pixels):
                    continue
                graph.add_edge(a, b, weight=math.hypot(dy, dx))
    # Remove only tiny terminal spurs; preserve their coordinates in the audit.
    pruned = []
    for start in sorted(list(graph)):
        if start not in graph or graph.degree[start] != 1:
            continue
        path = [start]; length = 0.
        while graph.degree[path[-1]] <= 2:
            choices = [n for n in graph[path[-1]] if len(path) == 1 or n != path[-2]]
            if not choices:
                break
            n = choices[0];length += graph[path[-1]][n]["weight"];path.append(n)
            if length > max_spur:
                break
        if length <= max_spur and graph.degree[path[-1]] > 2:
            graph.remove_nodes_from(path[:-1]);pruned.extend(path[:-1])
    branches = {n for n in graph if graph.degree[n] > 2}
    clusters = nx.Graph();clusters.add_nodes_from(branches)
    for a, b in itertools.combinations(sorted(branches), 2):
        if np.linalg.norm(np.array(a)-b) <= 8:
            route = nx.shortest_path(graph, a, b, weight="weight") if nx.has_path(graph, a, b) else []
            if route and sum(graph[u][v]["weight"] for u, v in zip(route, route[1:])) <= 8:
                clusters.add_edge(a, b)
    zones = []
    for group in nx.connected_components(clusters):
        zone = set(group)
        for a, b in itertools.combinations(sorted(group), 2):
            zone.update(nx.shortest_path(graph, a, b, weight="weight"))
        zones.append(zone)
    outside = set(graph)-set().union(*zones) if zones else set(graph)
    lifted = nx.Graph()
    node = lambda n: (n[0], n[1], -1)
    lifted.add_nodes_from(node(n) for n in outside)
    for a, b, info in graph.edges(data=True):
        if a in outside and b in outside:
            lifted.add_edge(node(a), node(b), **info)
    junctions = []
    for index, zone in enumerate(zones):
        ports = sorted((a, b) for a in zone for b in graph[a] if b not in zone)
        directions = []
        for inner, outer in ports:
            path = [inner, outer];length = graph[inner][outer]["weight"]
            while length < 15 and graph.degree[path[-1]] == 2:
                n = next(n for n in graph[path[-1]] if n != path[-2])
                if n in zone:
                    break
                length += graph[path[-1]][n]["weight"];path.append(n)
            tangent = np.asarray(path[-1], float)-inner
            directions.append(tangent/max(np.linalg.norm(tangent), 1e-12))
        pairs = []
        available = set(range(len(ports)))
        while len(available) >= 2:
            ranked = sorted((1.+float(directions[a]@directions[b]), a, b)
                            for a, b in itertools.combinations(sorted(available), 2))
            score, a, b = ranked[0]
            competitors = [x[0] for x in ranked[1:] if a in x[1:] or b in x[1:]]
            margin = min(competitors)-score if competitors else 2.
            if score > .6 or margin < .12:
                break
            pairs.append((a, b));available -= {a, b}
        for channel, (a, b) in enumerate(pairs):
            ia, oa = ports[a];ib, ob = ports[b]
            middle = nx.shortest_path(graph.subgraph(zone), ia, ib, weight="weight")
            route = [node(oa)]+[(n[0], n[1], index*32+channel) for n in middle]+[node(ob)]
            for a, b in zip(route, route[1:]):
                lifted.add_edge(a, b, weight=float(np.linalg.norm(np.array(a[:2])-b[:2])))
        # Unpaired arms remain visible fragments and request stereo/VLM evidence.
        for port in available:
            inner, outer = ports[port]
            lifted.add_edge(node(outer), (inner[0], inner[1], index*32+16+port), weight=1.)
        junctions.append(dict(port_count=len(ports), paired_ports=[list(p) for p in pairs],
                              unresolved_ports=sorted(available), physical_connection_established=False))
    paths = []
    for nodes in sorted(nx.connected_components(lifted), key=lambda x: min(x)):
        sub = lifted.subgraph(nodes)
        if any(sub.degree[n] > 2 for n in sub):
            raise ValueError("Internal channel graph branches")
        ends = sorted(n for n in sub if sub.degree[n] == 1)
        if ends:
            route = nx.shortest_path(sub, ends[0], ends[-1], weight="weight")
        else:
            if not len(sub):
                continue
            route = [min(sub)];previous = None
            while True:
                choices = sorted(n for n in sub[route[-1]] if n != previous)
                if not choices:
                    break
                following = choices[0];previous = route[-1];route.append(following)
                if following == route[0]:
                    break
        p = np.asarray([n[:2] for n in route], float)
        if len(p) >= 2 and np.linalg.norm(np.diff(p, axis=0), axis=1).sum() >= minimum_length:
            paths.append(p)
    retained = np.vstack(paths) if paths else np.empty((0, 2))
    coverage = float((cKDTree(retained).query(np.asarray(sorted(pixels)))[0] <= 2.).mean()) if len(retained) and pixels else 0.
    return paths, dict(junctions=junctions, source_skeleton_coverage=coverage,
                       pruned_spur_pixels=[list(map(int,p)) for p in pruned],
                       no_implicit_physical_crossing_weld=True, unresolved_topology=any(j["unresolved_ports"] for j in junctions))


def distal_region(mask, entry_border, fraction=.45):
    y, x = np.nonzero(mask)
    if len(x) < 16:
        return np.zeros_like(mask, bool)
    p = np.c_[x, y].astype(float);center = p.mean(0)
    axis = np.linalg.svd(p-center, full_matrices=False)[2][0]
    if axis[0]*(1 if entry_border == "left" else -1) < 0:
        axis = -axis
    along = (p-center)@axis
    selected = along >= np.quantile(along, 1-fraction)
    out = np.zeros_like(mask, bool);out[y[selected], x[selected]] = True
    return out


def tool_coverage(rendered, target, entry_border, tolerance=4.):
    """Shaft overlap alone cannot certify a missing or displaced distal tool."""
    distal = distal_region(target, entry_border)
    tips = distal_region(target, entry_border, .15)
    distance = cv2.distanceTransform((~rendered).astype("uint8"), cv2.DIST_L2, 5)
    union = (rendered | target).sum()
    return dict(iou=float((rendered & target).sum()/max(1, union)),
                observed_coverage=float((distance[target] <= tolerance).mean()) if target.any() else 0.,
                distal_coverage=float((distance[distal] <= tolerance).mean()) if distal.any() else 0.,
                tip_region_coverage=float((distance[tips] <= tolerance).mean()) if tips.any() else 0.,
                rendered_pixels=int(rendered.sum()),
                reconstruction_visible=bool(rendered.sum() >= 64 and distal.any() and tips.any() and (distance[distal] <= tolerance).mean() >= .5 and (distance[tips] <= tolerance).mean() >= .5))


def tool_mask_diagnostics(masks,thread_masks=None,entry_border=None):
    """Screen for incompatible stereo instances before expensive CAD fitting.

    This is a request for new perception evidence, not an assertion that masks
    are ground truth. In particular, a thread-shaped left mask cannot be paired
    with a broad instrument silhouette merely because both touch the border.
    """
    areas={s:int(np.asarray(masks[s],bool).sum()) for s in SIDES}
    ratio=max(areas.values())/max(1,min(areas.values()))
    failures=[];views={}
    if min(areas.values())<64:failures.append('empty_or_tiny_tool_instance')
    if ratio>4.:failures.append('incompatible_stereo_instance_area')
    for side in SIDES:
        mask=np.asarray(masks[side],bool)
        overlap=float((mask&thread_masks[side]).sum()/max(1,areas[side])) if thread_masks else None
        thickness=float(np.median(cv2.distanceTransform(mask.astype('uint8'),cv2.DIST_L2,5)[mask])*2) if mask.any() else 0.
        if overlap is not None and overlap>.5 and thickness<10.:failures.append(side+'_tool_mask_tracks_thread')
        if entry_border in ('left','right'):
            edge=mask[:,:2] if entry_border=='left' else mask[:,-2:]
            if not edge.any():failures.append(side+'_mask_omits_declared_tool_entry')
        views[side]=dict(area=areas[side],thread_overlap=overlap,interior_thickness_px=thickness)
    return dict(requires_resegmentation=bool(failures),failures=failures,views=views,area_ratio=ratio,
                mask_semantic_correctness_established=False,original_masks_preserved=True)


def select_native_tool_instances(probabilities,scores,boxes,entry_border,threshold):
    """Preserve SAM3 instance identities before a semantic-class union loses them.

    Select an observed border-anchored instance using explicit positive/negative
    guidance. A native prediction may itself be multipart; keep all its pixels.
    Distinct unrelated instances are never connected by a union of thread masks.
    """
    probabilities=np.asarray(probabilities,float);scores=np.asarray(scores,float)
    if probabilities.ndim!=3 or scores.shape!=(len(probabilities),) or not np.isfinite(scores).all():
        raise ValueError('Finite native instance masks and scores required')
    h,w=probabilities.shape[1:];positive=np.zeros((h,w),bool);negative=np.zeros((h,w),bool)
    for box in boxes:
        x0,y0,x1,y1=box['xyxy']
        if not 0<=x0<x1<=w or not 0<=y0<y1<=h:raise ValueError('Guidance outside original frame')
        target=positive if box['positive'] else negative
        target[int(np.floor(y0)):int(np.ceil(y1)),int(np.floor(x0)):int(np.ceil(x1))]=True
    candidates=[]
    for index,(p,score) in enumerate(zip(probabilities,scores)):
        if score<threshold:continue
        mask=p>.5;area=int(mask.sum())
        if not 64<=area<=h*w*.30:continue
        hit=float((mask&positive).sum()/area);excluded=float((mask&negative).sum()/area)
        entered=bool((mask[:,:2] if entry_border=='left' else mask[:,-2:]).any())
        opposite=bool((mask[:,-2:] if entry_border=='left' else mask[:,:2]).any())
        if hit<.5 or excluded>.1 or opposite:continue
        rank=float(score)+.3*hit-2.*excluded+.3*entered
        candidates.append(dict(index=index,score=float(score),positive_fraction=hit,negative_fraction=excluded,entry_observed=entered,rank=rank))
    anchored=[v for v in candidates if v['entry_observed']]
    if not anchored:raise ValueError('No native SAM3 instance covers the declared tool entry; request better bounds')
    selected=max(anchored,key=lambda v:v['rank']);mask=probabilities[selected['index']]>.5
    return mask,dict(selected_instance=selected,candidates=candidates,native_instance_identity_preserved=True,
                     multipart_pixels_preserved=True,source_instances_merged=False,synthesized_pixels=0)


def inferred_connection_guard(curve, observed_parts, endpoint_margin_m=.002, crossing_margin_m=.0005):
    """Do not silently route a new gap connector through an existing strand.

    Endpoint neighborhoods are exempt. A close interior crossing needs explicit
    stereo depth/topology evidence, not merely a lower fragment count.
    """
    curve = np.asarray(curve, float)
    if len(curve) < 3:
        return dict(allowed=False, reason="connector_needs_ordered_samples")
    reference = []
    for part in observed_parts:
        points = np.asarray(part, float)
        keep = np.minimum(np.linalg.norm(points-curve[0], axis=1), np.linalg.norm(points-curve[-1], axis=1)) > endpoint_margin_m
        reference.extend(points[keep])
    if not reference:
        return dict(allowed=True, reason="no_nonlocal_strand_intersection")
    d = cKDTree(np.asarray(reference)).query(curve[1:-1])[0]
    collision = bool((d < crossing_margin_m).any())
    return dict(allowed=not collision, reason="interior_strand_crossing_requires_stereo_review" if collision else "no_nonlocal_strand_intersection",
                minimum_nonlocal_distance_m=float(d.min()), topology_changed_by_fragment_count_alone=False)
