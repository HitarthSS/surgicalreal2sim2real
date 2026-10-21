"""Actual Newton viewer, distinguishing hidden shape from right-only depth."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from newton_scene_adapter import sha256
from view_newton_reconstruction import existing_tube_mesh


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','mesh-helper','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--headless',action='store_true');p.add_argument('--frames',type=int,default=61)
    a=p.parse_args()
    if not 2<=a.frames<=300:raise ValueError('Bounded viewing only')
    import cv2,newton,newton.viewer,warp as wp
    with np.load(a.bundle,allow_pickle=False) as d:b={k:d[k].copy()for k in d.files}
    if not bool(b['inspection_only']) or bool(b['training_eligible']):raise ValueError('Diagnostic-only bundle required')
    a.out.mkdir(parents=True,exist_ok=False);wp.set_device('cpu')
    v=newton.viewer.ViewerGL(width=960,height=720,headless=a.headless);v.set_model(newton.ModelBuilder(gravity=0).finalize(device='cpu'))
    tube=existing_tube_mesh(a.mesh_helper)
    allpoints=np.vstack((b['points_newton_m'],b['needle_points_newton_m'],b['context_tool_vertices']))
    center=(allpoints.min(0)+allpoints.max(0))/2;distance=max(np.linalg.norm(np.ptp(allpoints,axis=0)),.04)*1.7
    def mesh(name,vs,fs,color):
        if len(fs):v.log_mesh(name,wp.array(vs.astype('float32'),dtype=wp.vec3,device='cpu'),
            wp.array(fs.astype('int32').ravel(),dtype=wp.int32,device='cpu'),color=color,backface_culling=False)
    tv,tf=tube(b['points_newton_m'],.0005,10);side=tf[:199*20].reshape(199,20,3)
    shape=b['geometry_inferred_segments'];depth=b['depth_inferred_segments']
    def draw(n):
        angle=.42+2*np.pi*(n%a.frames)/(a.frames-1);pos=center+np.array([np.sin(angle)*.75,-np.cos(angle),.52])*distance;direction=center-pos
        v.set_camera(pos=wp.vec3(*pos),pitch=float(np.degrees(np.arctan2(direction[2],np.linalg.norm(direction[:2])))),
                     yaw=float(np.degrees(np.arctan2(direction[1],direction[0]))));v.camera.fov=38.
        v.begin_frame(n/30.)
        for name,mask,color in [('estimated_depth_support',~shape&~depth,(.15,.8,.25)),
            ('inferred_depth_visible_curve',~shape&depth,(.8,.2,.75)),('inferred_hidden_shape',shape,(1.,.48,.05))]:
            mesh('/thread/'+name,tv,side[mask].reshape(-1,3),color)
        nv,nf=tube(b['needle_points_newton_m'],float(b['needle_tube_radius_m']),12);mesh('/estimated_needle',nv,nf,(.2,.75,.9))
        mesh('/partial_tool_visual_only',b['context_tool_vertices'],b['context_tool_faces'],(.55,.57,.6));v.end_frame()
    movie=None
    try:
        if a.headless:
            movie=cv2.VideoWriter(str(a.out/'newton_orbit.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30.,(960,720))
            if not movie.isOpened():raise RuntimeError('Video encoder unavailable')
            for n in range(a.frames):
                draw(n);frame=cv2.cvtColor(v.get_frame(render_ui=False).numpy(),cv2.COLOR_RGB2BGR)
                cv2.rectangle(frame,(0,0),(960,74),(20,20,20),-1)
                for j,text in enumerate(('CASE 5 | CONNECTED PROPOSAL: REVIEW REQUIRED / DO NOT TRAIN',
                    'green=estimated depth support | pink=inferred depth | orange=inferred hidden shape | cyan=needle',
                    'Actual Newton viewer; camera orbit only. Fitted tool and needle-thread attachment still missing.')):
                    cv2.putText(frame,text,(10,20+22*j),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1)
                movie.write(frame)
                if n in (0,a.frames//2,a.frames-1):cv2.imwrite(str(a.out/f'frame_{n:03d}.png'),frame)
        else:
            t0=time.perf_counter()
            while v.is_running():draw(int((time.perf_counter()-t0)*30));time.sleep(1/30.)
    finally:
        if movie:movie.release()
        v.close()
    (a.out/'viewer_manifest.json').write_text(json.dumps(dict(renderer='newton.viewer.ViewerGL',
        newton_source=newton.__file__,warp_version=wp.__version__,device='cpu',simulated_motion=False,
        inspection_only=True,training_eligible=False,calibrated_policy_camera=False,
        source_sha256={str(p):sha256(p) for p in (a.bundle,a.mesh_helper,Path(__file__))}),indent=2)+'\n')


if __name__=='__main__':main()
