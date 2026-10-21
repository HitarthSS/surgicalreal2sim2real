#!/usr/bin/env python3
"""Static review OBJ: experimental thread plus saved needle/tool/pad. No physics."""
import argparse, json, sys
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--reconstruction',type=Path,required=True)
    p.add_argument('--pipeline-tools',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    sys.path.insert(0,str(a.pipeline_tools))
    from export_frame_scene_obj import tube_mesh, write_obj
    report=json.loads((a.reconstruction/'quality_report.json').read_text())
    if report['status']!='passed': raise RuntimeError('Cannot export a rejected reconstruction')
    if a.out.exists(): raise RuntimeError('Scene already exists; choose a new output path')
    src=a.run/'scene/combined_scene_online_tracker.obj'
    vertices=[]; groups=[]; group=None; material='tool'
    for line in src.read_text().splitlines():
        parts=line.split()
        if not parts: continue
        if parts[0]=='o':
            group={'name':parts[1],'faces':[],'material':material}; groups.append(group)
        elif parts[0]=='usemtl':
            material=parts[1]
            if group is not None: group['material']=material
        elif parts[0]=='v': vertices.append(list(map(float,parts[1:4])))
        elif parts[0]=='f':
            if group is None: raise RuntimeError('Faces without an object group')
            group['faces'].append([int(v.split('/')[0])-1 for v in parts[1:]])
    vertices=np.array(vertices); meshes=[]
    for group in groups:
        if group['name']=='newton_thread': continue
        faces=np.array(group['faces'],dtype=int)
        if faces.ndim!=2 or faces.shape[1]!=3: raise RuntimeError('Expected triangle OBJ')
        ids=np.unique(faces); local=np.searchsorted(ids,faces)
        meshes.append((group['name'],group['material'],vertices[ids],local))
    with np.load(a.run/'scene/newton_frame_scene.npz') as scene:
        radius=float(np.asarray(scene['radius']).reshape(-1)[0]) if 'radius' in scene else .0003
    with np.load(a.reconstruction/'thread_samples.npz') as data:
        points=data['points_m']; inferred=data['inferred']
    if not np.isfinite(points).all(): raise RuntimeError('Nonfinite thread geometry')
    flags=inferred[:-1] | inferred[1:]
    start=0; index=0
    while start<len(flags):
        end=start+1
        while end<len(flags) and flags[end]==flags[start]: end+=1
        material='thread_inferred' if flags[start] else 'thread_observed'
        v,f=tube_mesh(points[start:end+1],radius,18)
        meshes.append((f'{material}_{index}',material,v,f)); index+=1; start=end
    a.out.parent.mkdir(parents=True,exist_ok=True)
    write_obj(a.out,meshes)
    with a.out.with_suffix('.mtl').open('a') as mtl:
        mtl.write('\nnewmtl thread_observed\nKd 0.10 0.85 0.20\n\nnewmtl thread_inferred\nKd 1.0 0.45 0.05\n')
    a.out.with_suffix('.json').write_text(json.dumps({'kind':'static_review_scene','physics_rerun':False,
        'new_geometry':'experimental thread','reused_geometry':'saved native tool, needle, and pad meshes',
        'observed_thread_material':'green','inferred_thread_material':'orange',
        'inferred_geometry_is_ground_truth':False,'reconstruction_report':str(a.reconstruction/'quality_report.json')},indent=2)+'\n')
    print(a.out)

if __name__=='__main__': main()
