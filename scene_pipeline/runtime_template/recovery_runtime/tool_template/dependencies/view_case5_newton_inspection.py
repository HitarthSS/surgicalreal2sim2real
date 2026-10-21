"""Actual Newton ViewerGL for explicitly rejected candidates. No hardware I/O."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from newton_scene_adapter import sha256
from view_newton_reconstruction import existing_tube_mesh, source_inference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--method',choices=('fixed','heuristic','agent'),required=True)
    p.add_argument('--headless',action='store_true')
    p.add_argument('--device',default='cpu',choices=('cpu','cuda:0'))
    p.add_argument('--frames',type=int,default=61)
    p.add_argument('--physics',action='store_true')
    args=p.parse_args()
    if not 2 <= args.frames <= 300: raise ValueError('Bounded viewer only')
    import cv2
    import newton, newton.viewer, warp as wp
    root=args.root; folder=root/args.method
    out=folder/('viewer_physics' if args.physics else 'viewer_orbit')
    out.mkdir(exist_ok=False)
    a=np.load(folder/'inspection_bundle.npz',allow_pickle=False)
    bundle={k:a[k].copy() for k in a.files}
    if not bool(bundle['inspection_only']) or bool(bundle['training_eligible']):
        raise ValueError('Explicit diagnostic flags required')
    ledger=json.loads((root/'pipeline_ledger.json').read_text())
    report=json.loads((folder/'physics/report.json').read_text()) if args.physics else None
    if args.physics:
        s=np.load(folder/'physics/states.npz',allow_pickle=False)
        states=s['states']; needles=s['needle_states']; times=s['times']
        if not len(states) or not np.isfinite(states).all() or not np.isfinite(needles).all():
            raise ValueError('No finite recorded physics trajectory to show')
    else:
        states=bundle['points_newton_m'][None]; needles=bundle['needle_points_newton_m'][None]; times=np.array([0.])
    tube=existing_tube_mesh(root/'code/export_frame_scene_obj.py')
    segment_flags=source_inference(bundle,states[0])
    wp.set_device(args.device)
    viewer=newton.viewer.ViewerGL(width=960,height=720,headless=args.headless)
    viewer.set_model(newton.ModelBuilder(gravity=0.).finalize(device=args.device))
    viewer.show_grid=False
    lo,hi=np.asarray(ledger['common_camera_bounds_m'])
    center=(lo+hi)/2; distance=max(float(np.linalg.norm(hi-lo)),.04)*1.7
    def camera(angle):
        position=center+np.array([np.sin(angle)*.75,-np.cos(angle),.52])*distance
        direction=center-position
        yaw=float(np.degrees(np.arctan2(direction[1],direction[0])))
        pitch=float(np.degrees(np.arctan2(direction[2],np.linalg.norm(direction[:2]))))
        viewer.set_camera(pos=wp.vec3(*position),pitch=pitch,yaw=yaw); viewer.camera.fov=38.
    def mesh(name,v,f,color):
        if len(f):
            viewer.log_mesh(name,wp.array(v.astype(np.float32),dtype=wp.vec3,device=args.device),
                            wp.array(f.astype(np.int32).ravel(),dtype=wp.int32,device=args.device),
                            color=color,backface_culling=False)
    def draw(n):
        index=int(round((n % args.frames)/(args.frames-1)*(len(states)-1)))
        camera(.42 if args.physics else .42+2*np.pi*(n%args.frames)/(args.frames-1))
        viewer.begin_frame(n/30.)
        mesh('/partial_tool_depth_surface_visual_only',bundle['context_tool_vertices'],bundle['context_tool_faces'],(.55,.57,.6))
        v,f=tube(states[index],.0005,10)
        sides=f[:len(segment_flags)*20].reshape(-1,20,3)
        for flag,label,color in ((False,'input_supported',(.15,.8,.25)),(True,'unsupported_inferred',(1.,.48,.05))):
            mesh('/rejected_thread/'+label,v,sides[segment_flags==flag].reshape(-1,3),color)
        v,f=tube(needles[index],float(bundle['needle_tube_radius_m']),12)
        mesh('/estimated_needle',v,f,(.2,.75,.9))
        # Original visible mask back-projection highlights fragments omitted by the candidate.
        viewer.log_points('/original_mask_depth_estimate',wp.array(bundle['context_thread_points'].astype(np.float32),dtype=wp.vec3,device=args.device),
                          radii=.00012,colors=wp.array(np.tile([.7,.25,.55],(len(bundle['context_thread_points']),1)).astype(np.float32),dtype=wp.vec3,device=args.device))
        viewer.end_frame()
        return index
    movie=None
    try:
        if args.headless:
            movie=cv2.VideoWriter(str(out/'newton.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30.,(960,720))
            if not movie.isOpened(): raise RuntimeError('Video encoder unavailable')
            for n in range(args.frames):
                i=draw(n); rgb=viewer.get_frame(render_ui=False).numpy(); bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                cv2.rectangle(bgr,(0,0),(960,92),(20,20,20),-1)
                lines=[f'CASE 5 | {args.method.upper()} | REJECTED THREAD - DIAGNOSTIC ONLY',
                       'green=input support | orange=unsupported/inferred | cyan=needle | pink=original mask depth',
                       'gray=partial tool surface; fitted tool and attachment missing; DO NOT TRAIN']
                if report: lines.append(f'Newton CPU t={times[i]:.3f}s | static stability={report["stationary_diagnostic_passed"]} | zero gravity')
                else: lines.append('Actual Newton viewer | camera orbit only; no simulated motion')
                for j,line in enumerate(lines): cv2.putText(bgr,line,(10,20+22*j),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1)
                movie.write(bgr)
                if n in (0,args.frames//2,args.frames-1): cv2.imwrite(str(out/f'frame_{n:03d}.png'),bgr)
        else:
            start=time.perf_counter()
            while viewer.is_running():
                draw(int((time.perf_counter()-start)*30)); time.sleep(1/30)
    finally:
        if movie: movie.release()
        viewer.close()
    manifest=dict(renderer='newton.viewer.ViewerGL',newton_source=newton.__file__,warp_version=wp.__version__,
                  device=args.device,method=args.method,simulated_motion=args.physics,physics_report=report,
                  inspection_only=True,training_eligible=False,calibrated_policy_camera=False,
                  source_sha256={str(f.relative_to(root)):sha256(f) for f in
                                 (folder/'inspection_bundle.npz',root/'pipeline_ledger.json',Path(__file__))})
    (out/'viewer_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__': main()
