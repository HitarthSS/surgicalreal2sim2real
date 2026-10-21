"""Explicit diagnostic OBJ/NPZ export, separate from the strict scene adapter."""
import argparse
import json
from pathlib import Path
import numpy as np
from thread_visibility_repairs_v5 import load_parts
from newton_scene_adapter import CAMERA_TO_NEWTON,transform,sha256
from view_newton_reconstruction import existing_tube_mesh


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--context-bundle',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mesh-helper',type=Path,required=True)
    a=p.parse_args();s=json.loads(a.summary.read_text());c=s.get('candidate')
    if not c or c['stage']!='connected' or not c['report']['numerical_checks_passed']:
        raise ValueError('Need a checked connected proposal; no fallback')
    raw_path=Path(c['fragment_path'])
    if not raw_path.exists():
        # Downloaded frozen run: translate its /pilot output path only within
        # this controller's saved subtree, never substitute another candidate.
        suffix=raw_path.relative_to('/pilot/'+s['controller'])
        raw_path=a.summary.parent/suffix
    expected=c['fingerprints']['fragments']['sha256']
    if sha256(raw_path)!=expected:raise ValueError('Proposal fingerprint mismatch')
    parts=load_parts(raw_path)
    if len(parts)!=1:raise ValueError('Unresolved fragments cannot be exported as a joined candidate')
    source=parts[0];points=source['points_m'];arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    target=np.linspace(0.,arc[-1],200)
    resampled=np.column_stack([np.interp(target,arc,points[:,i]) for i in range(3)])
    labels={}
    # A rendered/physics interval is inferred if any dense source interval it
    # overlaps is inferred. Never lose provenance during physics resampling.
    for k in ('geometry_inferred','depth_inferred'):
        flags=source[k].astype(bool);seg=flags[:-1]|flags[1:]
        labels[k]=np.array([np.any(seg[(arc[:-1]<b)&(arc[1:]>aa)]) for aa,b in zip(target[:-1],target[1:])],bool)
    with np.load(a.context_bundle,allow_pickle=False) as d:
        context={k:d[k].copy() for k in ('needle_points_newton_m','needle_inferred','needle_tube_radius_m',
            'context_tool_vertices','context_tool_faces','context_thread_points')}
    a.out.mkdir(parents=True,exist_ok=False)
    bundle=dict(points_newton_m=transform(resampled,CAMERA_TO_NEWTON),
        inferred=np.r_[labels['geometry_inferred']|labels['depth_inferred'],False],
        geometry_inferred_segments=labels['geometry_inferred'],depth_inferred_segments=labels['depth_inferred'],
        tool_vertices_newton_m=np.empty((0,3)),tool_faces=np.empty((0,3),np.int32),
        inspection_only=np.array(True),training_eligible=np.array(False),attachment_applied=np.array(False),**context)
    np.savez_compressed(a.out/'inspection_bundle.npz',**bundle)
    tube=existing_tube_mesh(a.mesh_helper);v,f=tube(bundle['points_newton_m'],.0005,10)
    sides=f[:199*20].reshape(199,20,3);shape=labels['geometry_inferred'];depth=labels['depth_inferred']
    meshes=[]
    for label,selected,color in [('estimated_depth_support',~shape&~depth,(.15,.8,.25)),
        ('inferred_depth_visible_curve',~shape&depth,(.8,.2,.75)),('inferred_hidden_shape',shape,(1.,.48,.05))]:
        meshes.append((label,v,sides[selected].reshape(-1,3),color))
    nv,nf=tube(bundle['needle_points_newton_m'],float(bundle['needle_tube_radius_m']),12)
    meshes.extend([('estimated_needle',nv,nf,(.2,.75,.9)),('partial_tool_VISUAL_ONLY',context['context_tool_vertices'],context['context_tool_faces'],(.55,.57,.6))])
    obj=['# Inferred proposal; meters; not approved for training or production assembly','mtllib scene.mtl'];mtl=[];offset=1
    for label,vs,fs,color in meshes:
        obj.extend(['o '+label,'usemtl '+label]);obj.extend('v '+' '.join(f'{x:.12g}'for x in row)for row in vs)
        obj.extend('f '+' '.join(str(int(x)+offset)for x in row)for row in fs);offset+=len(vs)
        mtl.extend(['newmtl '+label,'Kd '+' '.join(map(str,color)),''])
    (a.out/'scene_newton_m.obj').write_text('\n'.join(obj)+'\n');(a.out/'scene.mtl').write_text('\n'.join(mtl)+'\n')
    report=dict(method=s['controller'],actual_model_used=s['actual_model_used'],proposal_report=c['report'],
        original_dense_sample_count=len(points),physics_sample_count=200,
        inferred_hidden_segment_count=int(shape.sum()),inferred_depth_segment_count=int(depth.sum()),
        inspection_only=True,training_eligible=False,shared_reconstruction_accepted=False,
        tool_fit='missing; partial surface visualization only',attachment='not established',
        source_sha256={str(p):sha256(p) for p in (a.summary,raw_path,a.context_bundle,a.mesh_helper,Path(__file__))})
    (a.out/'inspection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('method','actual_model_used','inferred_hidden_segment_count','inferred_depth_segment_count')}))


if __name__=='__main__':main()
