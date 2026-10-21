"""Bounded projected-path hypotheses; never identify crossing depth as measured.

The binary source masks remain immutable. A crossing may have two distinct
strands at the same image pixels, so replacing it with an unbranched binary mask
would erase evidence. This tool retains alternative ordered path pairings.
"""
import itertools
from pathlib import Path
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from reconstruction_verification import load_base


def pairings(items):
    if not items:
        yield []
        return
    a=items[0]
    for i,b in enumerate(items[1:],start=1):
        for rest in pairings(items[1:i]+items[i+1:]):yield [(a,b)]+rest


def simplify_pixel_graph(graph,max_spur_px):
    if type(max_spur_px) is not int or max_spur_px not in (2,6):
        raise ValueError('Use bounded 2px/6px spur hypotheses')
    g=graph.copy();redundant=[];removed=[]
    # Remove diagonal shortcuts only when an existing orthogonal two-edge
    # route preserves their connectivity and all source pixels.
    for a,b in list(g.edges):
        if abs(a[0]-b[0])==abs(a[1]-b[1])==1:
            if any(n in g and g.has_edge(a,n) and g.has_edge(n,b)
                   for n in [(a[0],b[1]),(b[0],a[1])]):
                g.remove_edge(a,b);redundant.append([list(map(int,a)),list(map(int,b))])
    while True:
        candidates=[]
        for start in list(g):
            if g.degree[start]!=1:continue
            path=[start];length=0.
            while g.degree[path[-1]]<=2:
                next_nodes=[n for n in g.neighbors(path[-1]) if len(path)==1 or n!=path[-2]]
                if not next_nodes:break
                n=next_nodes[0];length+=g[path[-1]][n]['weight'];path.append(n)
                if length>max_spur_px:break
            if length<=max_spur_px and len(path)>1 and g.degree[path[-1]]>2:
                candidates.append((length,path))
        if not candidates:break
        length,path=min(candidates,key=lambda item:(item[0],item[1]))
        g.remove_nodes_from(path[:-1]);removed.append(dict(length_px=float(length),pixels_rc=[list(map(int,n)) for n in path[:-1]]))
    return g,dict(redundant_diagonal_edges=redundant,pruned_spurs=removed,
                  pruned_source_pixels=sum(len(p['pixels_rc']) for p in removed))


def topology_proposals(mask,pipeline,max_spur_px):
    base=load_base(Path(pipeline));_,original,_,_,discarded=base.skeleton_graph(mask,0.,30)
    g,audit=simplify_pixel_graph(original,max_spur_px)
    branch_nodes=[n for n in g if g.degree[n]>2]
    clusters=nx.Graph();clusters.add_nodes_from(branch_nodes)
    for a,b in itertools.combinations(branch_nodes,2):
        if np.linalg.norm(np.asarray(a)-b)<=20 and nx.has_path(g,a,b):
            distance=nx.shortest_path_length(g,a,b,weight='weight')
            if distance<=24:clusters.add_edge(a,b)
    zones=[]
    for group in nx.connected_components(clusters):
        zone=set(group)
        for a,b in itertools.combinations(sorted(group),2):
            zone.update(nx.shortest_path(g,a,b,weight='weight'))
        zones.append(zone)
    if len(zones)>2:return [],dict(audit,unsupported='more_than_two_crossing_zones')
    choices=[];port_metadata=[]
    for zone_id,zone in enumerate(zones):
        ports=sorted((n,b) for n in zone for b in g.neighbors(n) if b not in zone)
        if len(ports)!=4:return [],dict(audit,unsupported='not_a_four_arm_crossing',port_count=len(ports))
        directions=[]
        for index,(inner,outer) in enumerate(ports):
            path=[inner,outer];length=g[inner][outer]['weight']
            while length<15 and g.degree[path[-1]]==2:
                n=next(n for n in g.neighbors(path[-1]) if n!=path[-2])
                if n in zone:break
                length+=g[path[-1]][n]['weight'];path.append(n)
            tangent=np.asarray(path[-1],float)-np.asarray(inner,float)
            tangent/=max(float(np.linalg.norm(tangent)),1e-12);directions.append(tangent)
            port_metadata.append(dict(zone=zone_id,port=index,xy=list(map(int,outer[::-1])),tangent_xy=tangent[::-1].tolist()))
        options=[]
        for matching in pairings(list(range(4))):
            score=sum(1.+float(np.dot(directions[a],directions[b])) for a,b in matching)
            options.append(dict(matching=matching,score=score,ports=ports,zone=zone))
        choices.append(sorted(options,key=lambda row:row['score']))
    combinations=list(itertools.product(*choices)) if choices else [()]
    rows=[]
    for combination in combinations:
        outside=set(g)-set().union(*zones) if zones else set(g)
        lifted=nx.Graph()
        def ordinary(n):return (n[0],n[1],-1)
        lifted.add_nodes_from(ordinary(n) for n in outside)
        for a,b,data in g.edges(data=True):
            if a in outside and b in outside:lifted.add_edge(ordinary(a),ordinary(b),**data)
        matching_rows=[]
        for zone_id,option in enumerate(combination):
            zone_graph=g.subgraph(option['zone'])
            for pair_id,(a,b) in enumerate(option['matching']):
                ia,oa=option['ports'][a];ib,ob=option['ports'][b]
                inside=nx.shortest_path(zone_graph,ia,ib,weight='weight')
                channel=zone_id*2+pair_id
                route=[ordinary(oa)]+[(n[0],n[1],channel) for n in inside]+[ordinary(ob)]
                for x,y in zip(route,route[1:]):lifted.add_edge(x,y,weight=float(np.linalg.norm(np.asarray(x[:2])-y[:2])))
                matching_rows.append(dict(zone=zone_id,ports=[a,b]))
        paths=[];failed=False
        for nodes in sorted(nx.connected_components(lifted),key=lambda part:min(part)):
            sub=lifted.subgraph(nodes);ends=sorted(n for n in sub if sub.degree[n]==1)
            if len(ends)!=2 or any(sub.degree[n]>2 for n in sub):failed=True;break
            route=nx.shortest_path(sub,*ends,weight='weight')
            paths.append(np.asarray([n[:2] for n in route],float))
        if failed or not paths:continue
        original_pixels=np.asarray(sorted(original),float)
        retained=np.vstack(paths);distances=cKDTree(retained).query(original_pixels)[0]
        coverage=float((distances<=2.).mean())
        if coverage<.95:continue
        score=sum(option['score'] for option in combination)
        identifier='route'+str(len(rows))
        rows.append(dict(proposal_id=identifier,score=score,ordered_paths_rc=paths,
            fragment_count=len(paths),original_skeleton_coverage_2px=coverage,
            pairing_hypothesis=matching_rows,ports=port_metadata,audit=audit,
            crossing_count=len(zones),discarded_short_components=discarded,
            topology_is_inferred=bool(zones),crossing_depth_order_recovered=False,
            source_masks_modified=False,training_eligible=False))
    return sorted(rows,key=lambda row:row['score']),audit
