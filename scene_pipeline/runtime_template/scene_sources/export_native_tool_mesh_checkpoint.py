"""Export the existing articulated tracker mesh in its fitted camera frame."""
import argparse
from pathlib import Path
import sys
import numpy as np
from complex_case_diagnostics import write_json
from reconstruction_tools import fingerprints
from newton_scene_adapter import rigid_transform


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','K','result','out'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    import torch
    torch.cuda.set_per_process_memory_fraction(.10)
    sys.path.insert(0,str(a.repo.resolve()))
    from types import SimpleNamespace
    from diffcali.models.CtRNet import CtRNet
    K=np.loadtxt(a.K,delimiter=',')
    camera=SimpleNamespace(use_gpu=True,use_nvdiffrast=True,height=480,width=640,
        fx=float(K[0,0]),fy=float(K[1,1]),px=float(K[0,2]),py=float(K[1,2]))
    model=CtRNet(camera)
    folder=a.repo/'urdfs/dVRK/meshes'
    assets=[folder/name for name in ('low_res_shaft_multi_cylinder.ply',
        'low_res_logo_low_res_1.ply','low_res_jawright_lowres.ply','low_res_jawleft_lowres.ply')]
    renderer=model.setup_robot_renderer([str(p) for p in assets],downscale_factor=2)
    renderer.set_mesh_visibility([True]*4)
    with np.load(a.result,allow_pickle=False) as data:
        pose=rigid_transform(data['frame4_to_camera'])
        angles=data['distal_joints_rad'].copy()
        loss=float(data['loss'])
    with torch.no_grad():
        mesh=renderer.get_robot_mesh(torch.as_tensor(angles,dtype=torch.float32,device='cuda'))
        vertices=mesh.verts_packed().detach().cpu().numpy()
        faces=mesh.faces_packed().detach().cpu().numpy()
    camera_vertices=vertices@pose[:3,:3].T+pose[:3,3]
    if not np.isfinite(camera_vertices).all():raise ValueError('Nonfinite native tool mesh')
    np.savez_compressed(a.out/'tool_camera_m.npz',vertices=camera_vertices,faces=faces,
        frame4_to_camera=pose,distal_joints_rad=angles)
    write_json(a.out/'report.json',dict(status='mesh_exported',native_frame='Frame 4',
        output_frame='rectified_left_camera',units='m; provisional camera scale',
        asset_type='Existing native LND meshes; physical tool type not independently confirmed',
        vertex_count=len(vertices),triangle_count=len(faces),optimizer_loss=loss,
        independent_accuracy_measured=False,training_eligible=False,
        source_sha256=fingerprints({'runner':Path(__file__),'result':a.result,'K':a.K,
            **{p.name:p for p in assets}})))


if __name__=='__main__':main()
