#!/usr/bin/env python3
"""Pass an existing pipeline frame to the original online_dvrk tracker."""
import argparse
import json
import hashlib
import ast
import gc
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch


def original_initialization(repo, model, renderer, camera, mask, tips):
    from diffcali.eval_dvrk.black_box_optimize import BayesOptBatchProblem
    from TuRBO.turbo.turbo_1 import Turbo1
    from review_sam3_masks import validate_jaw_tips

    tips = validate_jaw_tips(tips, mask.shape) if len(tips) else np.empty((0,2),dtype=np.float32)
    if not len(tips):
        # Upstream loss supports None, but its constructor unconditionally tensorizes it.
        # Supply a temporary shape-only placeholder, then remove it before any loss runs.
        import diffcali.eval_dvrk.black_box_optimize as optimization
        import inspect
        base = optimization.HeterogeneousBatchOptimize
        class OptionalKeypointOptimizer(base):
            def __init__(self, *args, **kwargs):
                bound = inspect.signature(base.__init__).bind(self, *args, **kwargs)
                bound.apply_defaults()
                absent = bound.arguments.get('ref_keypoints') is None
                if absent:bound.arguments['ref_keypoints'] = np.zeros((2,2),dtype=np.float32)
                base.__init__(*bound.args, **bound.kwargs)
                if absent:self.ref_keypoints = None
        optimization.HeterogeneousBatchOptimize = OptionalKeypointOptimizer
    target = torch.as_tensor(mask / 255., dtype=torch.float32, device=model.device)
    print(f'Initialization jaw-tip observations: {tips.tolist()}', flush=True)

    # Load the original function unchanged, without the video's SAM2/UI imports.
    source = repo / 'scripts/video_calibration.py'
    tree = ast.parse(source.read_text(), filename=str(source))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'initialization')
    # Upstream optimizer handles None keypoints; patch only the wrapper assignment.
    if not len(tips):
        assignment=function.body[1] if False else next(n for n in function.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ref_keypoints' for t in n.targets))
        assignment.value=ast.Constant(value=None)
        ast.fix_missing_locations(function)
    namespace = dict(torch=torch, th=torch, np=np, gc=gc, time=time,
                     BayesOptBatchProblem=BayesOptBatchProblem, Turbo1=Turbo1,
                     robot_renderer=renderer, ctrnet_args=camera,
                     args=SimpleNamespace(video_label='frame', machine_label='tool',
                                          batch_size=50, sample_number=1000, batch_iters=100,
                                          batch_opt_lr=3e-3, use_bo_initializer=True,
                                          use_prev_joint_angles=False))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    # The upstream function reads a fixed relative mask path. Keep the repo read-only.
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix='tracker-init-') as work:
        path = Path(work) / 'data/online_videos/frame/tool_ref_mask.png'
        path.parent.mkdir(parents=True)
        if not cv2.imwrite(str(path), mask):
            raise OSError(f'Cannot write initialization mask: {path}')
        try:
            os.chdir(work)
            cTr, angles = namespace['initialization'](model, target, tips, None, None)
        finally:
            os.chdir(cwd)
    return cTr, angles, tips


def export_tracked_obj(run, tool_mesh, pose):
    from export_frame_scene_obj import (
        tube_mesh, pad_mesh_from_scene, scene_needle_centerline,
        scene_needle_radius, transform_points, write_obj,
    )
    with np.load(run / 'scene/newton_frame_scene.npz') as scene:
        radius = float(np.asarray(scene['radius']).reshape(-1)[0]) if 'radius' in scene else .001
        thread_v, thread_f = tube_mesh(np.asarray(scene['states'][-1]), radius, 18)
        pad_v, pad_f = pad_mesh_from_scene(scene, .0025)
        meshes = [('background_pad', 'pad', pad_v, pad_f),
                  ('newton_thread', 'thread', thread_v, thread_f)]
        needle = scene_needle_centerline(scene)
        if needle is not None:
            verts, faces = tube_mesh(needle, scene_needle_radius(scene), 18)
            meshes.append(('rigid_needle', 'needle', verts, faces))
    # The original mesh is already articulated in Frame 4. No Si-URDF conversion.
    verts = transform_points(tool_mesh.verts_packed().detach().cpu().numpy(), pose)
    faces = tool_mesh.faces_packed().detach().cpu().numpy()
    if not np.isfinite(verts).all():
        raise ValueError('Nonfinite tracked tool vertices')
    meshes.append(('tracked_tool', 'tool', verts, faces))
    path = run / 'scene/combined_scene_online_tracker.obj'
    write_obj(path, meshes)
    print(f'OBJ: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--K', type=Path, required=True)
    parser.add_argument('--export-obj-only', action='store_true')
    parser.add_argument('--seed-source', choices=['original', 'pose-estimator', 'tracker'], default='original',
                        help='Use native Bayesian initialization, the learning-based pose, or a saved tracker result')
    parser.add_argument('--initialization-request',type=Path,help='Explicit machine/human input with source hashes; never substitutes a manual-review stamp')
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    from diffcali.models.CtRNet import CtRNet
    from diffcali.eval_dvrk.trackers import Tracker

    mask = cv2.imread(str(args.run / 'robot/sam3_tool/tool_mask.png'), cv2.IMREAD_GRAYSCALE)
    if mask is None or not np.any(mask):
        raise ValueError('Missing or empty SAM3 tool mask')
    tips = np.empty((0, 2), dtype=np.float32)
    if not args.export_obj_only and (args.initialization_request or args.seed_source == 'original'):
        from review_sam3_masks import load_reviewed_jaw_tips, validate_jaw_tips
        directory = args.run / 'robot/sam3_tool'
        if args.initialization_request:
            request=json.loads(args.initialization_request.read_text())
            assert request['input_origin'] in ('agent','heuristic','manual')
            for path,value in request['source_sha256'].items():
                assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==value, 'Initialization source changed'
            tips=validate_jaw_tips(request['jaw_tips_xy'],mask.shape) if request.get('jaw_tips_xy') else np.empty((0,2),dtype=np.float32)
        else:
            tips = load_reviewed_jaw_tips(directory / 'manual_review.json', [
                directory / 'tool_mask.png',
                args.run / 'frames/episode_0000/colors/left_image_000000.jpg'], mask.shape)
    tip_source='VLM_or_explicit_observation' if len(tips) else 'unavailable'
    detector_receipt={}
    if not len(tips) and args.initialization_request and request['input_origin']=='heuristic':
        from diffcali.utils.contour_tip_net import Tip2DNet,detect_keypoints_2d
        weights=args.repo/'ContourTipNet/models/cnn_model.pth'
        detector_receipt=dict(checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),model='native_Tip2DNet',source='independent_heuristic_detector')
        detector=Tip2DNet().to('cuda');detector.load_state_dict(torch.load(weights,map_location='cuda',weights_only=True));detector.eval()
        with torch.no_grad():detected=detect_keypoints_2d(detector,torch.as_tensor(mask/255.,dtype=torch.float32,device='cuda'))
        if detected is not None:
            tips=validate_jaw_tips(detected.detach().cpu().numpy().tolist(),mask.shape);tip_source='native_Tip2DNet'
        del detector
        detector_receipt['pair_available']=bool(len(tips))
    K = np.loadtxt(args.K, delimiter=',')
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device='cuda')
    camera = SimpleNamespace(
        use_gpu=True, use_nvdiffrast=True, height=mask.shape[0], width=mask.shape[1],
        fx=float(K[0, 0]), fy=float(K[1, 1]), px=float(K[0, 2]), py=float(K[1, 2]))
    model = CtRNet(camera)
    meshes = args.repo / 'urdfs/dVRK/meshes'
    renderer = model.setup_robot_renderer([str(meshes / name) for name in (
        'low_res_shaft_multi_cylinder.ply', 'low_res_logo_low_res_1.ply',
        'low_res_jawright_lowres.ply', 'low_res_jawleft_lowres.ply')], downscale_factor=2)
    renderer.set_mesh_visibility([True] * 4)
    if args.export_obj_only:
        with np.load(args.run / 'robot/online_tracker/tracker_result.npz') as result:
            with torch.no_grad():
                mesh = renderer.get_robot_mesh(tensor(result['distal_joints_rad']))
                export_tracked_obj(args.run, mesh, result['frame4_to_camera'])
        return
    # Original video_calibration defaults, with temporal filtering off for one frame.
    options = SimpleNamespace(
        symmetric_jaw=True, use_render_loss=True, use_pts_loss=bool(len(tips)),
        mse_weight=6., dist_weight=0., app_weight=6e-6, pts_weight=3e-3 if len(tips) else 0.,
        downscale_factor=2, use_filter=False, use_mix_angle=False,
        searcher='CMA-ES', cos_reparams=True, use_contour_tip_net=False,
        use_prev_joint_angles=True, popsize=70, final_iters=100,
        contour_tip_net_path=str(args.repo / 'ContourTipNet/models/cnn_model.pth'))
    print(f'Initialization method: {args.seed_source}', flush=True)
    if args.seed_source == 'original':
        cTr, angles, tips = original_initialization(
            args.repo.resolve(), model, renderer, camera, mask, tips)
        pose = model.cTr_to_pose_matrix(cTr[None])[0].detach().cpu().numpy()
        distal = angles.detach().cpu().numpy()
    else:
        seed_path = args.run / ('robot/tool_pose.npz' if args.seed_source == 'pose-estimator'
                                else 'robot/online_tracker/tracker_result.npz')
        with np.load(seed_path) as data:
            pose = np.eye(4)
            if args.seed_source == 'tracker':
                pose = data['frame4_to_camera'].copy()
                distal = data['distal_joints_rad'].copy()
            elif 'cam_to_ee' in data:
                pose = data['cam_to_ee'].copy()
            else:
                pose[:3, :3], pose[:3, 3] = data['R'], data['t'].reshape(3)
        if args.seed_source == 'pose-estimator':
            inputs = args.run / 'frames/episode_0000_inputs'
            q = np.load(inputs / 'joint_000000.npy').reshape(-1)
            jaw = np.load(inputs / 'jaw_000000.npy').reshape(-1)[0]
            distal = np.array([q[4], q[5], jaw / 2, jaw / 2])
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or distal.shape != (4,):
        raise ValueError('Invalid seed pose or joints')
    if not np.isfinite(distal).all() or np.any(distal < [-1.5707, -1.3963, 0, 0]) or np.any(distal > [1.5707, 1.3963, 1.5707, 1.5707]):
        raise ValueError('Seed joints exceed the original tracker bounds')
    tracker = Tracker(
        model, renderer, model.pose_matrix_to_cTr(tensor(pose)[None])[0], tensor(distal),
        num_iters=3, intr=tensor(K), p_local1=tensor([0, .0004, .0096]),
        p_local2=tensor([0, -.0004, .0096]),
        stdev_init=tensor([.1] * 3 + [.001] * 3 + [.1] * 4),
        searcher='CMA-ES', args=options)
    target = tensor(mask / 255.)
    keypoints = tensor(tips) if len(tips) else None
    if args.initialization_request and request.get('keypoints_required') and not len(tips):raise ValueError('Visible native jaw tips were required but missing')
    cTr, angles, loss = tracker.track_frame(target, tensor(distal), is_init=True, keypoints=keypoints)
    if not all(torch.isfinite(v).all() for v in (cTr, angles, torch.as_tensor(loss))):
        raise ValueError('Original tracker returned a nonfinite result')
    out = args.run / 'robot/online_tracker'
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'tracker_result.npz',
             frame4_to_camera=model.cTr_to_pose_matrix(cTr[None])[0].detach().cpu().numpy(),
             distal_joints_rad=angles.detach().cpu().numpy(), loss=float(loss),
             seed_source=args.seed_source, seed_frame4_to_camera=pose, seed_distal_joints_rad=distal,
             seed_keypoints_xy=tips, jaw_tip_evidence_available=bool(len(tips)), keypoint_loss_enabled=bool(options.use_pts_loss and len(tips)), native_tracking_tip_offsets_m=np.array([[0,.0004,.0096],[0,-.0004,.0096]]), keypoint_coordinate_space='original_full_frame_pixels', keypoint_source=tip_source, silhouette_only_initialization=not bool(len(tips)), initialization_origin=(request['input_origin'] if args.initialization_request else 'manual_review'))
    (out/'keypoint_receipt.json').write_text(json.dumps(dict(source=tip_source,original_pixel_xy=tips.tolist(),keypoint_loss_enabled=bool(options.use_pts_loss and len(tips)),tracking_jaw_offsets_m=[[0,.0004,.0096],[0,-.0004,.0096]],initialization_jaw_offsets_m=[[0,.0004,.009],[0,-.0004,.009]],detector=detector_receipt,coordinate_space='original_full_frame_pixels',missing_points_are_explicit=True),indent=2))
    with torch.no_grad():
        cv2.imwrite(str(out / 'overlay.png'), tracker.visualize(target, cTr, angles, keypoints))
    print(f'Tracker result: {out}')


if __name__ == '__main__':
    main()
