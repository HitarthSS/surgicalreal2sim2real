"""Shared bounded observation revision and stereo articulated registration.

Baseline modules are imported read-only. No model code or geometry is executed.
The native mesh assets and forward kinematics are reused on CPU.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

from native_keypoint_contract import body_landmarks,camera_landmarks,pair_error,project as project_native_tips
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize, linear_sum_assignment
from scipy.spatial.transform import Rotation

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from shared_tool_estimation import masks_from_request, project, verify, save
from tool_protocol import validate_initialization, digest

BOUNDS = dict(translation_m=.025, rotation_component_rad=.262,
              joint_delta_rad=.262, max_evaluations=1400, max_seconds=180,
              max_jaw_tip_residual_px=20.)
LOW = np.array([-1.5707, -1.3963, 0., 0.])
HIGH = np.array([1.5707, 1.3963, 1.5707, 1.5707])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def revise_observations(current, decision, sizes, scene_version):
    """Replace one visual instance only; validate against the untouched other tool."""
    if set(decision) != {'scene_version', 'action_id', 'parameters', 'evidence_ids', 'reason'}:
        raise ValueError('Exact decision fields required')
    if decision['scene_version'] != scene_version:
        raise ValueError('Stale observation version')
    if not isinstance(decision['reason'], str) or not 1 <= len(decision['reason']) <= 500:
        raise ValueError('Bounded decision reason required')
    ids = decision['evidence_ids']
    allowed = {'image.left', 'image.right', 'fit.left', 'fit.right', 'candidate.left', 'candidate.right'}
    if not isinstance(ids, list) or not {'image.left', 'image.right'} <= set(ids) <= allowed:
        raise ValueError('Registered full stereo evidence required')
    action = decision['action_id']
    if action in ('keep', 'defer', 'refine_stereo_articulation'):
        if decision['parameters'] != {}:
            raise ValueError('This action takes no parameters')
        return deepcopy(current)
    if action != 'revise_tool_observations':
        raise ValueError('Unknown operation')
    p = decision['parameters']
    if not isinstance(p, dict) or set(p) != {'instance_id', 'views', 'reason'} or p['instance_id'] != 'tool_B':
        raise ValueError('Pilot permits only unresolved tool_B observation revision')
    revised = deepcopy(current)
    item = next(x for x in revised['instances'] if x['instance_id'] == p['instance_id'])
    item.update(views=deepcopy(p['views']), reason=p['reason'])
    revised = validate_initialization(revised, sizes)
    masks = masks_from_request(revised, sizes)
    # A tip must be supported by the declared silhouette, allowing contour rounding.
    for side, view in item['views'].items():
        near = cv2.dilate(masks['tool_B'][side].astype('uint8'), np.ones((9, 9), 'uint8'))
        for x, y in view['jaw_tips_xy']:
            if not near[int(round(y)), int(round(x))]:
                raise ValueError('Jaw tip unsupported by proposed visible tool mask')
    return revised


class NativeTool:
    """Exact native FK/PLY geometry, with identity checked against the saved fit."""
    def __init__(self, repo):
        import torch
        from pytorch3d.io import load_ply
        self.torch = torch
        source = Path(repo) / 'diffcali/eval_dvrk/LND_fk.py'
        spec = importlib.util.spec_from_file_location('native_lnd_fk_repair', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.fk = getattr(module.lndFK, '_torchdynamo_orig_callable', module.lndFK)
        self.assets = []
        self.asset_hashes = {str(source): sha(source)}
        faces = []
        offset = 0
        for name in ['low_res_shaft_multi_cylinder.ply', 'low_res_logo_low_res_1.ply',
                     'low_res_jawright_lowres.ply', 'low_res_jawleft_lowres.ply']:
            path = Path(repo) / 'urdfs/dVRK/meshes' / name
            vertices, triangles = load_ply(str(path))
            vertices, triangles = vertices.numpy(), triangles.numpy()
            self.assets.append(vertices)
            faces.append(triangles + offset)
            offset += len(vertices)
            self.asset_hashes[str(path)] = sha(path)
        self.faces = np.concatenate(faces)
        self.slices = []
        offset = 0
        for vertices in self.assets:
            self.slices.append(slice(offset, offset + len(vertices)))
            offset += len(vertices)

    def body(self, q):
        # Native exporter explicitly casts transforms to float32.
        with self.torch.no_grad():
            rotations, translations = self.fk(self.torch.as_tensor(q, dtype=self.torch.float32))
        r, t = rotations.numpy(), translations.numpy()
        return np.concatenate([v @ ri.T + ti for v, ri, ti in zip(self.assets, r, t)])

    def tip_body(self,q):
        with self.torch.no_grad():
            r,t=self.fk(self.torch.as_tensor(q,dtype=self.torch.float32))
        return body_landmarks(r.numpy(),t.numpy(),'tracking')

    def validate_source(self, mesh):
        expected = self.body(mesh['distal_joints_rad']) @ mesh['frame4_to_camera'][:3, :3].T + mesh['frame4_to_camera'][:3, 3]
        error = float(np.max(np.abs(expected - mesh['vertices'])))
        if error > 1e-7 or not np.array_equal(mesh['faces'], self.faces):
            raise ValueError('Native geometry/FK mismatch: ' + str(error))
        return error


def silhouette(vertices, faces, K, baseline, shape):
    """Triangle union, clipped by the image; out-of-frame shaft is not an error."""
    xy = project(vertices, K, baseline)
    mask = np.zeros(shape, 'uint8')
    triangles = xy[faces]
    good = (vertices[faces, 2] > .005).all(axis=1)
    good &= (triangles.max(axis=1) >= [0, 0]).all(axis=1)
    good &= (triangles.min(axis=1) < [shape[1], shape[0]]).all(axis=1)
    for triangle in triangles[good]:
        cv2.fillConvexPoly(mask, np.rint(np.clip(triangle, -100000, 100000) * 16).astype('int32'), 1, shift=4)
    return mask.astype(bool)


def edge(mask):
    result = mask & ~cv2.erode(mask.astype('uint8'), np.ones((3, 3), 'uint8')).astype(bool)
    # Frame clipping boundaries are not physical tool edges.
    result[:2] = False; result[-2:] = False; result[:, :2] = False; result[:, -2:] = False
    return result


def tip_vertices(native, mesh, views, K, baseline):
    """Select distal native surface landmarks; correspondence fixed across stereo."""
    ids = []
    # Geometry landmarks use the original projections, with both stereo views.
    # Only the distal 15% of each native jaw surface is eligible.
    body = native.body(mesh['distal_joints_rad'])
    for section in native.slices[2:]:
        candidates = np.arange(len(body))[section]
        local = native.assets[len(ids) + 2]
        axis = np.ptp(local, axis=0).argmax()
        # Native jaw assets extend along positive x; verify using their range.
        distal = local[:, axis] >= np.quantile(local[:, axis], .85)
        candidates = candidates[distal]
        center = mesh['vertices'][candidates].mean(axis=0)
        ids.append(int(candidates[np.argmin(np.linalg.norm(mesh['vertices'][candidates] - center, axis=1))]))
    costs = np.zeros((2, 2))
    for side, view in views.items():
        if not view['jaw_tips_xy']:continue
        predicted = project(mesh['vertices'][ids], K, baseline if side == 'right' else 0)
        costs += np.linalg.norm(predicted[:, None] - np.asarray(view['jaw_tips_xy'])[None], axis=2)
    _, order = linear_sum_assignment(costs)
    # Pair order can differ between views in machine proposals. Resolve each view
    # against the same physical jaw IDs; do not assume list-order identity.
    targets = {}
    for side, view in views.items():
        if not view['jaw_tips_xy']:continue
        pred = project(mesh['vertices'][ids], K, baseline if side == 'right' else 0)
        _, permutation = linear_sum_assignment(np.linalg.norm(pred[:, None] - np.asarray(view['jaw_tips_xy'])[None], axis=2))
        targets[side] = np.asarray(view['jaw_tips_xy'])[permutation]
    return ids, targets


def refine(native, mesh, masks, views, K, baseline, out, reference_masks=None, search=None, initial=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    source_error = native.validate_source(mesh)
    ids = None # Native jaw-frame endpoints, never approximate mesh vertex IDs.
    tips = {s:np.asarray(v['jaw_tips_xy'],float) for s,v in views.items() if v['jaw_tips_xy']}
    settings = dict(BOUNDS, **(search or {}))
    start = time.monotonic()
    pose0, q0 = mesh['frame4_to_camera'], mesh['distal_joints_rad']
    # Units normalized to bounded [-1,1] parameters. Same bounds for both controllers.
    scales = np.r_[np.full(3, settings['rotation_component_rad']), np.full(3, settings['translation_m']), np.full(4, settings['joint_delta_rad'])]
    lower = np.r_[-np.ones(6), np.maximum(-1, (LOW-q0)/settings['joint_delta_rad'])]
    upper = np.r_[np.ones(6), np.minimum(1, (HIGH-q0)/settings['joint_delta_rad'])]
    shape = (masks['left'].shape[0]//2, masks['left'].shape[1]//2)
    ksmall = K.copy(); ksmall[:2] *= .5; ksmall[:2, 2] -= .25
    targets = {s: cv2.resize(m.astype('uint8'), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool) for s, m in masks.items()}
    boundaries = {s: edge(m) for s, m in targets.items()}
    distances = {s: distance_transform_edt(~e) for s, e in boundaries.items()}
    reference_masks = masks if reference_masks is None else reference_masks
    frozen = {s: cv2.resize(m.astype('uint8'), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool) for s,m in reference_masks.items()}
    initial_iou = {}
    for side, target in frozen.items():
        rendered = silhouette(mesh['vertices'],native.faces,ksmall,baseline if side=='right' else 0,shape)
        initial_iou[side] = (rendered & target).sum()/max(1,(rendered | target).sum())
    def moved(x):
        delta = x * scales
        pose = pose0.copy()
        pose[:3, :3] = Rotation.from_rotvec(delta[:3]).as_matrix() @ pose0[:3, :3]
        pose[:3, 3] += delta[3:6]
        q = q0 + delta[6:]
        vertices = native.body(q) @ pose[:3, :3].T + pose[:3, 3]
        return vertices, pose, q
    best = [float('inf'), np.zeros(10)]
    evaluations = [0]
    def objective(x):
        if time.monotonic()-start > settings['max_seconds']:
            raise TimeoutError('Bounded optimizer wall time reached')
        vertices, pose, q = moved(x); value = 0.
        # The long proximal shaft can extend behind the camera outside its field
        # of view. Require positive distal geometry, not every hidden shaft vertex.
        if not np.isfinite(vertices).all() or not .03 <= pose[2,3] <= .5 or (vertices[native.slices[2].start:, 2] <= .005).any():
            return 1e5
        for side in targets:
            b = baseline if side == 'right' else 0
            rendered = silhouette(vertices, native.faces, ksmall, b, shape)
            boundary = edge(rendered)
            if not boundary.any():
                return 1e5
            union = (rendered | targets[side]).sum()
            # Equal view weights. Bidirectional *silhouette* distances, not all vertices.
            chamfer = distances[side][boundary].mean() + distance_transform_edt(~boundary)[boundaries[side]].mean()
            value += 1 - (rendered & targets[side]).sum()/max(union, 1) + .02*chamfer
            # Optimize toward the existing full-resolution acceptance constraint,
            # instead of spending the entire budget on a candidate it will reject.
            # The 0.02 objective margin is conservative; final 0.03 gates unchanged.
            frozen_iou=(rendered & frozen[side]).sum()/max(1,(rendered | frozen[side]).sum())
            value += 25*max(0., initial_iou[side]-.02-frozen_iou)
            if side in tips:
                predicted=project_native_tips(camera_landmarks(native.tip_body(q),pose),K,b)
                residuals,tip_loss=pair_error(predicted,tips[side])
                value += .04*tip_loss/20
        value += .005*float(np.mean(x*x))
        evaluations[0] += 1
        if value < best[0]: best[:] = [float(value), x.copy()]
        return float(value)
    before = objective(np.zeros(10)); timeout = False
    initial = np.zeros(10) if initial is None else np.clip(np.asarray(initial)/scales, lower, upper)
    objective(initial)
    try:
        result = minimize(objective, initial, method='Powell', bounds=list(zip(lower, upper)),
                          options=dict(maxfev=settings['max_evaluations'], xtol=.01, ftol=.0001))
        message = str(result.message)
    except TimeoutError as error:
        timeout = True; message = str(error)
    vertices, pose, q = moved(best[1])
    candidate = dict(vertices=vertices, faces=native.faces.copy(), frame4_to_camera=pose, distal_joints_rad=q)
    np.savez_compressed(out/'tool_camera_m.npz', **candidate)
    tip_errors = {s:pair_error(project_native_tips(camera_landmarks(native.tip_body(q),pose),K,baseline if s=='right' else 0),tips[s])[0].tolist() for s in tips}
    save(out/'operation.json', dict(operation='reinitialize_stereo_tool',bounds=settings, initial_delta=(initial*scales).tolist(),
         normalized_delta=best[1].tolist(), delta=(best[1]*scales).tolist(),
         objective_before=before, objective_after=best[0], evaluations=evaluations[0],
         runtime_seconds=time.monotonic()-start, wall_time_limit_reached=timeout, optimizer_message=message,
         native_geometry_max_reproduction_error_m=source_error, native_asset_hashes=native.asset_hashes,
         jaw_tip_vertex_ids=None, jaw_tip_semantics="native_jaw_frames2and3_tracking_offsets", jaw_tip_offsets_m=[[0,.0004,.0096],[0,-.0004,.0096]], keypoint_pairing="min_over_two_permutations_plus_midpoint", jaw_tip_residuals_px=tip_errors, jaw_tip_evidence_available=bool(tips),
         frozen_mask_objective_floor={s:float(v-.02) for s,v in initial_iou.items()},
         observations_are_proposals=True, independent_accuracy_measured=False))
    return candidate


def promotion(before, after):
    """Identical existing gate, always evaluated on the SAME reference masks."""
    if not after['accepted_numerical_hypothesis']:
        return False
    for side in ('left', 'right'):
        a, b = before['views'][side], after['views'][side]
        if b['median_absolute_depth_difference_m'] is None or a['median_absolute_depth_difference_m'] is None:
            return False
        values=[a['mask_iou'],b['mask_iou'],a['median_absolute_depth_difference_m'],b['median_absolute_depth_difference_m']]
        if not np.isfinite(values).all():
            return False
        if b['mask_iou'] < a['mask_iou']-.03 or b['median_absolute_depth_difference_m'] > a['median_absolute_depth_difference_m']+.001:
            return False
    return True
