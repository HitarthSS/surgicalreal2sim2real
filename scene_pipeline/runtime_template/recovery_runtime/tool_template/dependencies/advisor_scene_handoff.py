"""Portable inspection handoff with explicit per-arm frames and missing inputs.

Does not turn snapshots into demonstrations or claim a Newton-ready robot.
The prior strict one-tool Newton adapter remains the production gate.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.interpolate import make_interp_spline
from PIL import Image
from newton_scene_adapter import CAMERA_TO_NEWTON, centerline, rigid_transform, transform, sha256
from audit_snapshot_dataset import timestamp_ns
from thread_visibility_repairs_v5 import load_parts


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def build(snapshot, out, visibility_summary=None, needle_bundle=None, arm_calibration=None):
    snapshot, out = Path(snapshot), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    sources = {}
    for name in ('left_rect.png','right_rect.png','calibration.json','psm_state.json'):
        p=snapshot/name
        if p.is_file():
            (out/'observations').mkdir(exist_ok=True)
            shutil.copy2(p,out/'observations'/name)
            sources[str(p.resolve())]=sha256(p)
    cal=json.loads((snapshot/'calibration.json').read_text())
    dims={}
    for s in ('left','right'):
        with Image.open(snapshot/(s+'_rect.png')) as im:dims[s]=list(im.size)
    if dims['left']!=dims['right'] or dims['left']!=cal['image_size']:
        raise ValueError('Image/calibration dimensions differ')
    state=json.loads((snapshot/'psm_state.json').read_text()) if (snapshot/'psm_state.json').exists() else {'arms':{}}
    calibration=json.loads(Path(arm_calibration).read_text()) if arm_calibration else None
    if calibration:
        sources[str(Path(arm_calibration).resolve())]=sha256(arm_calibration)
        if calibration.get('units')!='m' or calibration.get('frame')!='rectified_left_camera':
            raise ValueError('Arm calibration requires explicit meter/camera convention')
    manifest=dict(schema='ismr_advisor_scene_v1',scene_id=snapshot.name,
        purpose='inspectable_reconstruction_handoff',geometry_units='m',
        geometry_scale_independently_confirmed=False,
        canonical_frame='rectified_left_camera',
        recorded_stereo_baseline_units='unconfirmed',
        estimated_geometry_stereo_baseline_m=None,
        T_newton_from_camera=CAMERA_TO_NEWTON.tolist(),
        images={'left':'observations/left_rect.png','right':'observations/right_rect.png'},
        image_dimensions=dims,calibration='observations/calibration.json',
        acquisition_topics=cal.get('topics'),thread=None,needle=None,robots=[],
        attachments=[],physics_parameters={'status':'requires_measured_dimensions_and_calibration'},
        static_snapshot=True,simulation_ready=False,training_eligible=False,
        blocking_prerequisites=['validated_metric_scale','articulated_robot_assets_and_action_mapping',
            'per_arm_camera_calibration','support_surface_and_contact_validation',
            'needle_thread_attachment_decision_and_motion_test','synchronized_teleop_observation_state_action_trajectories'])
    left_stamp=timestamp_ns(cal.get('image_headers',{}).get('left'))
    for arm,record in sorted(state.get('arms',{}).items()):
        cart=record.get('cartesian_state') or {}; header=cart.get('header') or {}
        pose=rigid_transform(cart['transform_4x4']).tolist() if cart.get('transform_4x4') is not None else None
        arm_info=dict(arm_id=arm,recorded_pose_frame=header.get('frame_id'),
            T_recorded_frame_from_tool=pose,recorded_translation_units='unconfirmed',
            measured_joint_names=record.get('joint_state',{}).get('name'),
            measured_joint_positions=record.get('joint_state',{}).get('position'),
            jaw_state=record.get('jaw_state'),T_camera_from_tool=None,
            camera_pose_status='missing_camera_from_base_calibration',
            visual_tool_identity='unassigned; do not infer PSM identity from left/right image position',
            fitted_tool_mesh=None,urdf=None,source='observations/psm_state.json')
        stamp=timestamp_ns(header)
        arm_info['image_pose_offset_ms']=None if stamp is None or left_stamp is None else (stamp-left_stamp)/1e6
        # Explicit calibration and declared robot-state units are mandatory;
        # independent base-frame poses cannot be dropped into camera space.
        if calibration and arm in calibration.get('arms',{}):
            ac=calibration['arms'][arm]
            if ac.get('base_frame')!=header.get('frame_id') or ac.get('robot_state_units')!='m':
                raise ValueError('Wrong base frame or undeclared robot pose units')
            if pose is None:raise ValueError('Missing recorded tool transform')
            camera_pose=rigid_transform(ac['T_camera_from_base']) @ rigid_transform(pose)
            arm_info.update(T_camera_from_tool=camera_pose.tolist(),
                recorded_translation_units='m',camera_pose_status='composed_from_declared_calibration')
        manifest['robots'].append(arm_info)
    if not manifest['robots']:manifest['blocking_prerequisites'].append('robot_states_missing')
    if visibility_summary:
        summary_path=Path(visibility_summary);summary=json.loads(summary_path.read_text())
        if summary_path.parent.name not in ('agent','heuristic'):
            raise ValueError('Require explicit controller output folder')
        candidate=summary.get('candidate') or {}
        if not candidate or candidate.get('stage')!='connected' or not candidate['report']['numerical_checks_passed']:
            raise ValueError('Cannot hand off unchecked/unresolved thread as connected')
        if candidate['report'].get('case_id',snapshot.name)!=snapshot.name:
            raise ValueError('Thread belongs to a different snapshot')
        # Bind original images to this acquisition, not just a filename/case ID.
        for view in ('left','right'):
            if summary['provenance'][view]['sha256']!=sha256(snapshot/(view+'_rect.png')):
                raise ValueError('Thread source image does not match snapshot')
        source_case_path=Path(summary['provenance']['case']['path'])
        if sha256(source_case_path)!=summary['provenance']['case']['sha256']:
            raise ValueError('Thread source case changed')
        source_case=json.loads(source_case_path.read_text())
        if source_case['case_id']!=snapshot.name:
            raise ValueError('Wrong scene identifier')
        manifest['estimated_geometry_stereo_baseline_m']=source_case['baseline_m']
        samples=Path(candidate['fragment_path'])
        if sha256(samples)!=candidate['fingerprints']['fragments']['sha256']:
            raise ValueError('Thread proposal provenance mismatch')
        parts=load_parts(samples)
        if len(parts)!=1:raise ValueError('A joined thread must have one explicit ordered part')
        part=parts[0];points=centerline(part['points_m'])
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))];u=arc/arc[-1]
        spline=make_interp_spline(u,points,k=3)
        np.savez_compressed(out/'thread_curve.npz',points_camera_m=points,
            points_newton_m=transform(points,CAMERA_TO_NEWTON),arclength_parameter=u,
            spline_knots=spline.t,spline_coefficients_camera_m=spline.c,spline_degree=np.array(spline.k),
            **{k:part[k] for k in ('depth_inferred','geometry_inferred','observed_left','observed_right')})
        manifest['thread']=dict(file='thread_curve.npz',primary_representation='ordered_sampled_centerline',
            optional_spline='scipy BSpline(knots, coefficients_camera_m, degree); interpolates estimated samples',
            points=len(points),arclength_m=float(arc[-1]),geometry_inferred_points=int(part['geometry_inferred'].sum()),
            depth_inferred_points=int(part['depth_inferred'].sum()),status='numerically_checked_proposal_needs_review',
            spline_physics_verified=False,independent_accuracy_measured=False)
        sources.update({str(summary_path.resolve()):sha256(summary_path),str(samples.resolve()):sha256(samples)})
    else:manifest['blocking_prerequisites'].append('thread_reconstruction_missing')
    if needle_bundle:
        # Context geometry must be explicitly identified as Newton-frame data.
        p=Path(needle_bundle)
        with np.load(p,allow_pickle=False) as data:
            points=transform(centerline(data['needle_points_newton_m']),np.linalg.inv(CAMERA_TO_NEWTON))
            labels=data['needle_inferred'].copy()
            radius=float(data['needle_tube_radius_m'])
        if labels.dtype!=bool or labels.shape!=(len(points),) or not 0<radius<.005:
            raise ValueError('Invalid needle context labels or radius')
        pose=np.eye(4);pose[:3,3]=points.mean(axis=0)
        np.savez_compressed(out/'needle_geometry.npz',points_body_m=points-pose[:3,3],
            points_camera_m=points,T_camera_from_needle_body=pose,inferred=labels,
            tube_radius_m=np.array(radius))
        manifest['needle']=dict(file='needle_geometry.npz',evidence='previous_estimated_context_geometry',
            body_frame='origin at sample centroid; axes parallel to camera axes; packaging convention, not independent pose reference',
            radius_m=radius,radius_is_unmeasured_assumption=True,source_sha256=sha256(p),
            needle_thread_attachment_established=False)
        sources[str(p.resolve())]=sha256(p)
    else:manifest['blocking_prerequisites'].append('needle_reconstruction_missing')
    manifest['source_sha256']=sources
    write(out/'scene.json',manifest)
    (out/'README.md').write_text('Inspection handoff; not a complete robot simulation or valid demonstration.\n\n'
        'Load scene.json first. Geometry uses meters with provisional physical scale in rectified-left-camera coordinates. '
        'T_newton_from_camera maps these points to Newton z-up. thread_curve.npz retains sampled centerline and observed/inferred labels; '
        'optional spline is an interpolant of estimated geometry, not a validated dynamic cable. '
        'needle_geometry.npz carries estimated rigid geometry and an explicit body-frame packaging transform; radius is provisional.\n\n'
        'robots is a list, one entry per arm. PSM1/PSM2 recorded base-frame poses are NOT camera poses. '
        'Never substitute an identity camera/base transform or duplicate one arm for the other. '
        'Camera calibration, robot assets, tool fit, jaw/action mapping, surfaces, physical parameters and attachment are explicit prerequisites. '
        'Missing fields stay null. No needle/thread joint is implied by visual proximity.\n\n'
        'For VR demos record synchronized stereo images, states and teleop commands for both arms; preserve joint names/units, '
        'camera/base/world transforms, timestamps, episode/task IDs, attachment state, success labels and natural failures. '
        'Snapshots contain no time-series actions and cannot train behavior cloning by themselves.\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot','out'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('visibility-summary','needle-bundle','arm-calibration'):p.add_argument('--'+name,type=Path)
    a=p.parse_args();result=build(a.snapshot,a.out,a.visibility_summary,a.needle_bundle,a.arm_calibration)
    print(json.dumps({'scene_id':result['scene_id'],'robot_arm_ids':[x['arm_id'] for x in result['robots']],
        'simulation_ready':result['simulation_ready'],'blocking_prerequisites':result['blocking_prerequisites']}))


if __name__=='__main__':main()
