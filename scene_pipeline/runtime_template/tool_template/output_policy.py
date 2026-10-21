"""Separate reconstruction availability from visual/numerical verification.

No geometry is fabricated or changed. Poor alignment is retained as approximate;
missing or structurally unusable geometry is an explicit failure.
"""
import numpy as np

POLICY_VERSION = 'tool_best_effort_v1'

def classify(selection, mesh):
    failures=[]
    if mesh is None:
        failures.append('no_fitted_geometry')
    else:
        vertices=np.asarray(mesh.get('vertices',[]))
        faces=np.asarray(mesh.get('faces',[]))
        pose=np.asarray(mesh.get('frame4_to_camera',[]))
        if vertices.ndim!=2 or vertices.shape[1:]!=(3,) or len(vertices)<3:
            failures.append('invalid_vertices')
        elif not np.isfinite(vertices).all():failures.append('nonfinite_vertices')
        elif np.all(vertices[:,2]<=0):failures.append('all_geometry_behind_camera')
        if faces.ndim!=2 or faces.shape[1:]!=(3,) or len(faces)==0:
            failures.append('invalid_faces')
        elif not np.issubdtype(faces.dtype,np.integer) or faces.min()<0 or faces.max()>=len(vertices):
            failures.append('invalid_face_indices')
        if pose.shape!=(4,4) or not np.isfinite(pose).all():failures.append('invalid_pose')
        elif not (np.allclose(pose[3],[0,0,0,1],atol=1e-5) and np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-4) and np.isclose(np.linalg.det(pose[:3,:3]),1,atol=1e-4)):
            failures.append('nonrigid_pose')
        joints=np.asarray(mesh.get('distal_joints_rad',[]))
        if joints.size==0 or not np.isfinite(joints).all():failures.append('missing_or_nonfinite_joints')
    verified=bool(selection.get('accepted_tool_hypothesis',False))
    return dict(policy_version=POLICY_VERSION,
        reconstruction_status='failed_no_usable_geometry' if failures else ('verified_hypothesis' if verified else 'approximate_reconstruction'),
        reconstruction_available=not failures,requires_review=not verified or bool(failures),
        prior_agent_review=selection.get('agent_review_status'),
        numerical_checks_passed=bool(selection.get('numerical_recovery_passed',False)),
        accepted_tool_hypothesis=verified,structural_failures=failures,
        independent_accuracy_measured=False,physics_validated=False,
        collider_ready=False,mesh=selection.get('mesh'),
        note='Availability is not evidence of accuracy. Keep imperfect finite fitted geometry as an approximate hypothesis.')
