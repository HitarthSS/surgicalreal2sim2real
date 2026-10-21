"""Actual Newton viewer for a static two-tool reconstruction; no physics claim."""
import argparse
import json
from pathlib import Path
import numpy as np
from newton_scene_adapter import CAMERA_TO_NEWTON, transform, sha256
from view_newton_reconstruction import existing_tube_mesh


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','mesh-helper','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--method',choices=('heuristic','agent'),required=True)
    p.add_argument('--frames',type=int,default=61)
    a=p.parse_args()
    if not 2<=a.frames<=120:raise ValueError('Bounded static inspection only')
    import pyglet
    pyglet.options['headless']=True
    import cv2,newton,newton.viewer,warp as wp
    with np.load(a.bundle,allow_pickle=False) as d:b={k:d[k].copy()for k in d.files}
    if not bool(b['inspection_only']) or bool(b['training_eligible']):raise ValueError('Inspection-only bundle required')
    a.out.mkdir(parents=True,exist_ok=False)
    wp.set_device('cpu')
    viewer=newton.viewer.ViewerGL(width=960,height=720,headless=True)
    viewer.set_model(newton.ModelBuilder(gravity=0).finalize(device='cpu'));viewer.show_grid=False
    tube=existing_tube_mesh(a.mesh_helper)
    meshes=[]
    v,f=tube(b['points_newton_m'],.0005,10);meshes.append(('thread',v,f,(.15,.8,.25)))
    v,f=tube(b['needle_points_newton_m'],float(b['needle_tube_radius_m']),12)
    meshes.append(('needle',v,f,(.2,.75,.9)))
    view_points=[b['points_newton_m'],b['needle_points_newton_m']]
    for name,color in [('tool_A',(1.,.55,.15)),('tool_B',(.70,.35,.90))]:
        vertices=b[name+'_vertices'];meshes.append((name,vertices,b[name+'_faces'],color))
        # Frame bounds affect only the inspection camera; retain the full meshes.
        camera_points=transform(vertices,np.linalg.inv(CAMERA_TO_NEWTON));z=camera_points[:,2]
        K=b['K'];xy=camera_points[:,:2]/z[:,None]*[K[0,0],K[1,1]]+K[:2,2]
        visible=(z>.03)&(z<.5)&(xy[:,0]>=0)&(xy[:,0]<=639)&(xy[:,1]>=0)&(xy[:,1]<=479)
        view_points.append(vertices[visible])
    bounds=np.vstack(view_points);center=(bounds.min(0)+bounds.max(0))/2
    distance=max(np.linalg.norm(np.ptp(bounds,axis=0)),.04)*1.5
    writer=cv2.VideoWriter(str(a.out/'newton_orbit.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30.,(960,720))
    if not writer.isOpened():raise RuntimeError('Video encoder unavailable')
    try:
        for n in range(a.frames):
            angle=.15+2*np.pi*n/(a.frames-1)
            position=center+np.array([np.sin(angle)*.8,-np.cos(angle),.35])*distance
            direction=center-position
            viewer.set_camera(pos=wp.vec3(*position),
                pitch=float(np.degrees(np.arctan2(direction[2],np.linalg.norm(direction[:2])))),
                yaw=float(np.degrees(np.arctan2(direction[1],direction[0]))));viewer.camera.fov=40.
            viewer.begin_frame(n/30.)
            for name,v,f,color in meshes:
                viewer.log_mesh('/'+name,wp.array(v.astype('float32'),dtype=wp.vec3,device='cpu'),
                    wp.array(f.astype('int32').ravel(),dtype=wp.int32,device='cpu'),color=color,backface_culling=False)
            viewer.end_frame()
            frame=cv2.cvtColor(viewer.get_frame(render_ui=False).numpy(),cv2.COLOR_RGB2BGR)
            cv2.rectangle(frame,(0,0),(960,65),(20,20,20),-1)
            for j,text in enumerate((a.method.upper()+' | two independently fitted tool meshes | static reconstruction',
                'green=thread | cyan=needle | orange/purple=tools | no verified attachment or physics')):
                cv2.putText(frame,text,(10,22+25*j),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1)
            writer.write(frame)
            if n in (0,a.frames//2,a.frames-1):cv2.imwrite(str(a.out/f'frame_{n:03d}.png'),frame)
    finally:
        writer.release();viewer.close()
    (a.out/'viewer_manifest.json').write_text(json.dumps(dict(renderer='newton.viewer.ViewerGL',
        warp_version=wp.__version__,newton_source=newton.__file__,simulated_motion=False,
        full_tool_meshes=True,inspection_only=True,training_eligible=False,
        calibrated_policy_camera=False,source_sha256={str(p):sha256(p)for p in (a.bundle,a.mesh_helper,Path(__file__))}),indent=2)+'\n')


if __name__=='__main__':main()
