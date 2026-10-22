"""Optional stereo-grounded background geometry and bounded material selection."""
from pathlib import Path
import hashlib
import json
import math
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter

PRESETS = {
    'cloth': (.95, 0., .6), 'silicone': (.7, 0., .8),
    'rigid_plastic': (.5, 0., .4), 'metal': (.25, 1., .3),
    'tissue_like': (.65, 0., .5), 'unknown': (.8, 0., .5),
}
DECISION_KEYS = {'surface_model', 'material_preset', 'roughness', 'metallic', 'reason'}
CONFIG_KEYS = {'enabled', 'vlm', 'decision_file', 'sam3d_candidate', 'stride',
               'min_depth_m', 'max_depth_m', 'foreground_margin_px',
               'plane_tolerance_m', 'max_edge_m', 'min_support_pixels'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def configuration(value=None):
    value = dict(value or {})
    if set(value) - CONFIG_KEYS: raise ValueError('Unknown background configuration')
    defaults = dict(enabled=False, vlm=False, stride=8, min_depth_m=.03,
                    max_depth_m=.5, foreground_margin_px=3, plane_tolerance_m=.002,
                    max_edge_m=.01, min_support_pixels=500)
    defaults.update(value)
    for name, low, high in [('stride', 2, 32), ('foreground_margin_px', 0, 20),
                            ('min_support_pixels', 30, 100000)]:
        if type(defaults[name]) is not int or not low <= defaults[name] <= high:
            raise ValueError('Invalid background ' + name)
    for name in ['min_depth_m', 'max_depth_m', 'plane_tolerance_m', 'max_edge_m']:
        x = defaults[name]
        if type(x) not in (int, float) or not math.isfinite(x) or x <= 0:
            raise ValueError('Invalid background ' + name)
    if defaults['min_depth_m'] >= defaults['max_depth_m']: raise ValueError('Invalid depth bounds')
    for name in ['enabled', 'vlm']:
        if type(defaults[name]) is not bool: raise ValueError('Expected boolean ' + name)
    return defaults


def decision(value):
    if not isinstance(value, dict) or set(value) != DECISION_KEYS:
        raise ValueError('Expected exact background decision keys')
    if value['surface_model'] not in ['auto', 'plane', 'depth_mesh', 'sam3d']:
        raise ValueError('Unknown surface model')
    if value['material_preset'] not in PRESETS: raise ValueError('Unknown material preset')
    for key in ['roughness', 'metallic']:
        if type(value[key]) not in (int, float) or not math.isfinite(value[key]) or not 0 <= value[key] <= 1:
            raise ValueError('Appearance parameter outside [0,1]')
    if not isinstance(value['reason'], str) or not 1 <= len(value['reason']) <= 500:
        raise ValueError('Invalid decision reason')
    return dict(value)


def project(points, K, baseline=0.):
    p = np.array(points, dtype=float, copy=True); p[:, 0] -= baseline
    q = p @ K.T
    return q[:, :2] / q[:, 2, None]


def unproject(xy, z, K):
    rays = np.c_[xy, np.ones(len(xy))] @ np.linalg.inv(K).T
    return rays * (np.asarray(z) / rays[:, 2])[:, None]


def foreground(scene, case, side, shape):
    paths = set()
    for key, value in case['inputs'].items():
        if side in key and ('mask' in key or 'occluder' in key) and Path(value).is_file(): paths.add(Path(value))
    for pattern in ['segmentation/' + side + '*mask.png',
                    'initial_tool_observations/' + side + '_occluder.png',
                    'tool_agent/*/sam3/' + side + '*mask.png']:
        paths.update(scene.glob(pattern))
    mask = np.zeros(shape, bool)
    for path in sorted(paths):
        a = np.asarray(Image.open(path).convert('L'))
        if a.shape != shape: raise ValueError('Foreground mask dimensions differ: ' + str(path))
        mask |= a > 0
    # Never silently fit tools/needle/thread into the background when masks are absent.
    if not paths: raise ValueError('Foreground masks required; run segmentation first')
    return mask, {str(p): sha(p) for p in paths}


def fit_plane(points, tolerance):
    rng = np.random.default_rng(0)
    p = points[rng.choice(len(points), min(8192, len(points)), replace=False)]
    best = None; score = -1
    for _ in range(100):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b-a, c-a); length = np.linalg.norm(n)
        if length < 1e-10: continue
        n /= length; d = -n @ a; inside = abs(p @ n + d) < tolerance
        if inside.sum() > score: best = inside; score = int(inside.sum())
    if best is None or score < 3: raise ValueError('No supported background plane')
    center = p[best].mean(axis=0)
    n = np.linalg.svd(p[best]-center, full_matrices=False)[2][-1]
    if n[2] < 0: n = -n
    d = -n @ center; errors = abs(points @ n + d)
    return n, d, dict(inlier_fraction=float((errors < tolerance).mean()),
                     residual_p90_m=float(np.quantile(errors, .9)))


def similarity(source, target):
    """Fit positive uniform scale, proper rotation and translation to stereo anchors."""
    a, b = np.asarray(source, float), np.asarray(target, float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or len(a) < 6:
        raise ValueError('SAM3D needs at least six matched 3D/pixel anchors')
    if not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('Invalid anchors')
    ac, bc = a.mean(0), b.mean(0); x, y = a-ac, b-bc
    if np.linalg.matrix_rank(x, tol=1e-8) < 2: raise ValueError('Degenerate anchors')
    u, s, vt = np.linalg.svd(y.T @ x / len(a)); sign = np.eye(3)
    sign[-1, -1] = np.linalg.det(u @ vt); rotation = u @ sign @ vt
    scale = float((s * np.diag(sign)).sum() / np.mean(np.sum(x*x, axis=1)))
    if not math.isfinite(scale) or scale <= 0: raise ValueError('Invalid aligned scale')
    translation = bc - scale * (rotation @ ac)
    return scale, rotation, translation


def sam3d_candidate(path, depth, support, K, tolerance):
    with np.load(path, allow_pickle=False) as z:
        v = z['vertices'].astype(float); f = z['faces']; anchors = z['anchor_points_local'].astype(float)
        xy = z['anchor_pixels_left'].astype(float)
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all(): raise ValueError('Invalid SAM3D mesh')
    if f.ndim != 2 or f.shape[1] != 3 or f.dtype.kind not in 'iu' or len(f) == 0 or f.min() < 0 or f.max() >= len(v):
        raise ValueError('Invalid SAM3D faces')
    if xy.shape != (len(anchors), 2) or not np.isfinite(xy).all(): raise ValueError('Invalid SAM3D pixel anchors')
    ij = np.rint(xy).astype(int); h, w = depth.shape
    if ((ij < 0).any() or (ij[:, 0] >= w).any() or (ij[:, 1] >= h).any()
            or not support[ij[:, 1], ij[:, 0]].all()): raise ValueError('SAM3D anchors lack stereo background support')
    target = unproject(xy, depth[ij[:, 1], ij[:, 0]], K)
    scale, rotation, translation = similarity(anchors, target)
    aligned = scale * (v @ rotation.T) + translation
    err = np.linalg.norm(scale*(anchors @ rotation.T)+translation-target, axis=1)
    if np.max(err) > tolerance: raise ValueError('SAM3D alignment failed stereo anchors')
    return aligned, f.astype(np.int32), dict(scale=scale, rotation=rotation.tolist(),
            translation_m=translation.tolist(), anchor_max_error_m=float(err.max()),
            candidate_sha256=sha(path), hidden_surface_is_generated=True)


def write_obj(path, vertices, faces, uv, *, offset=0, name='background'):
    lines = ['o ' + name, 'usemtl background']
    lines += ['v ' + ' '.join(format(float(x), '.10g') for x in p) for p in vertices]
    lines += ['vt ' + ' '.join(format(float(x), '.10g') for x in p) for p in uv]
    lines += ['f ' + ' '.join(str(int(i)+1+offset)+'/'+str(int(i)+1) for i in f) for f in faces]
    Path(path).write_text('mtllib background.mtl\n' + '\n'.join(lines) + '\n')


def build(case_path, output, config=None, selector=None):
    started = time.monotonic(); cfg = configuration(config)
    case_path, out = Path(case_path).resolve(), Path(output).resolve(); scene = case_path.parent
    out.mkdir(parents=True, exist_ok=False)
    case = json.loads(case_path.read_text()); inputs = case['inputs']
    protected = {str(case_path): sha(case_path)}
    for key in ['left', 'right', 'depth_npy', 'K']: protected[str(Path(inputs[key]).resolve())] = sha(inputs[key])
    images = {s: np.asarray(Image.open(inputs[s]).convert('RGB')) for s in ['left', 'right']}
    if images['left'].shape != images['right'].shape: raise ValueError('Stereo image dimensions differ')
    h, w = images['left'].shape[:2]; depth = np.load(inputs['depth_npy'], allow_pickle=False).astype(float)
    K = np.loadtxt(inputs['K'], delimiter=','); baseline = float(case['baseline_m'])
    if depth.shape != (h, w) or K.shape != (3, 3) or not np.isfinite(K).all() or np.linalg.det(K) <= 0 or not 0 < baseline < .1:
        raise ValueError('Invalid depth, intrinsics or metric baseline')
    masks = {}
    for side in images:
        masks[side], hashes = foreground(scene, case, side, (h, w)); protected.update(hashes)
        if cfg['foreground_margin_px']: masks[side] = binary_dilation(masks[side], iterations=cfg['foreground_margin_px'])
        Image.fromarray((~masks[side]).astype('uint8')*255).save(out/(side+'_background_mask.png'))
    support = ~masks['left'] & np.isfinite(depth) & (depth >= cfg['min_depth_m']) & (depth <= cfg['max_depth_m'])
    yy, xx = np.where(support)
    if len(xx) < cfg['min_support_pixels']: raise ValueError('Insufficient background stereo support')
    points = unproject(np.c_[xx, yy], depth[support], K)
    n, d, plane_stats = fit_plane(points, cfg['plane_tolerance_m'])
    diagnostics = dict(background_depth_pixels=len(xx), background_fraction=float(support.mean()), **plane_stats)
    chosen = dict(surface_model='auto', material_preset='unknown', roughness=.8, metallic=0., reason='Deterministic stereo baseline; no material class inferred from colour alone.')
    source = 'deterministic'; selection_error = None
    try:
        if cfg.get('decision_file'):
            protected[str(Path(cfg['decision_file']).resolve())] = sha(cfg['decision_file'])
            chosen = decision(json.loads(Path(cfg['decision_file']).read_text())); source = 'decision_file'
        elif selector:
            chosen = decision(selector(case, diagnostics)); source = 'vlm'
    except Exception as error:
        selection_error = type(error).__name__; source = 'deterministic_after_selector_failure'
    save(out/'decision.json', dict(source=source, selection=chosen, selector_error_type=selection_error))
    model = chosen['surface_model']; plane_ok = plane_stats['inlier_fraction'] >= .85
    fallbacks = []; alignment = None
    if model == 'auto': model = 'plane' if plane_ok else 'depth_mesh'
    if model == 'plane' and not plane_ok:
        fallbacks.append('Requested plane did not fit stereo support'); model = 'depth_mesh'
    stride = min(cfg['stride'], max(1, int(cfg['max_edge_m']*min(K[0, 0], K[1, 1])/np.median(depth[support])*.6)))
    diagnostics['effective_stride_px'] = stride
    ys = np.unique(np.r_[np.arange(0, h, stride), h-1]); xs = np.unique(np.r_[np.arange(0, w, stride), w-1])
    gx, gy = np.meshgrid(xs, ys); xy = np.c_[gx.ravel(), gy.ravel()]; sampled = support[gy, gx].ravel()
    if model == 'sam3d':
        try:
            path = cfg.get('sam3d_candidate')
            if not path: raise ValueError('No SAM3D candidate configured')
            protected[str(Path(path).resolve())] = sha(path)
            vertices, faces, alignment = sam3d_candidate(path, depth, support, K, cfg['plane_tolerance_m'])
            if (vertices[:, 2] < cfg['min_depth_m']).any() or (vertices[:, 2] > cfg['max_depth_m']).any():
                raise ValueError('SAM3D candidate outside supported camera depth bounds')
            xy = project(vertices, K); sampled = np.zeros(len(vertices), bool)
        except (ValueError, KeyError, OSError) as error:
            fallbacks.append('SAM3D candidate unavailable or failed stereo alignment: '+type(error).__name__)
            model = 'plane' if plane_ok else 'depth_mesh'
    if model != 'sam3d':
        if model == 'plane':
            rays = unproject(xy, np.ones(len(xy)), K); denom = rays @ n
            z = np.divide(-d, denom, out=np.full(len(xy), np.nan), where=abs(denom)>1e-9)
        else:
            weights = gaussian_filter(support.astype(float), 1.)
            smoothed = gaussian_filter(np.where(support, depth, 0.), 1.) / np.maximum(weights, 1e-12)
            z = smoothed[gy, gx].ravel()
        vertices = unproject(xy, z, K)
        valid = np.isfinite(vertices).all(1) & (z >= cfg['min_depth_m']) & (z <= cfg['max_depth_m'])
        if model == 'depth_mesh': valid &= sampled
        indices = np.arange(len(xy)).reshape(len(ys), len(xs))
        a, b, c, e = indices[:-1, :-1].ravel(), indices[:-1, 1:].ravel(), indices[1:, :-1].ravel(), indices[1:, 1:].ravel()
        faces = np.vstack([np.c_[a, c, b], np.c_[b, c, e]]).astype(np.int32)
        ok = valid[faces].all(1)
        # Test all three edges: no triangles across foreground holes or depth jumps.
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            ok &= np.linalg.norm(vertices[faces[:, i]]-vertices[faces[:, j]], axis=1) <= cfg['max_edge_m']
        if model == 'depth_mesh':
            mid = np.rint(xy[faces].mean(1)).astype(int)
            ok &= support[mid[:, 1], mid[:, 0]]
            # Any foreground pixel inside a grid cell prevents either triangle bridging it.
            cells = np.array([support[ys[i]:ys[i+1]+1, xs[j]:xs[j+1]+1].all()
                              for i in range(len(ys)-1) for j in range(len(xs)-1)])
            ok &= np.r_[cells, cells]
        faces = faces[ok]
    if not len(faces): raise ValueError('No supported background triangles')
    used = np.unique(faces); remap = np.full(len(vertices), -1, int); remap[used] = np.arange(len(used))
    vertices, xy, sampled, faces = vertices[used], xy[used], sampled[used], remap[faces]
    uv = np.c_[xy[:, 0]/max(1, w-1), 1-xy[:, 1]/max(1, h-1)]
    quality = evaluate(vertices, images, masks, depth, support, K, baseline, cfg['plane_tolerance_m'])
    accepted = quality['depth_support_fraction'] >= .8 and quality['right_background_support_fraction'] >= .5
    material = dict(preset=chosen['material_preset'], rendering=dict(roughness=chosen['roughness'], metallic=chosen['metallic']),
                    physics=dict(static_friction=PRESETS[chosen['material_preset']][2], source='assumed_preset', calibrated=False, deformable_model_enabled=False))
    save(out/'material.json', material)
    texture = images['left'].copy(); nearest = distance_transform_edt(masks['left'], return_distances=False, return_indices=True)
    texture[masks['left']] = images['left'][nearest[0][masks['left']], nearest[1][masks['left']]]
    Image.fromarray(texture).save(out/'texture.png')
    np.savez_compressed(out/'background_geometry.npz', vertices_camera_m=vertices, faces=faces, uv=uv,
                        observed_depth_support=sampled, geometry_inferred=~sampled,
                        K=K, baseline_m=baseline, plane_normal=n, plane_offset_m=d)
    write_obj(out/'background_camera.obj', vertices, faces, uv)
    roughness = chosen['roughness']; (out/'background.mtl').write_text('newmtl background\nKd 1 1 1\nmap_Kd texture.png\nPr '+str(roughness)+'\nPm '+str(chosen['metallic'])+'\nNs '+str(max(0.,2/max(.01,roughness)**2-2))+'\n')
    preview(out/'stereo_overlay.jpg', vertices, faces, images, K, baseline, masks)
    assert all(sha(p) == value for p, value in protected.items()), 'Background stage changed an input'
    manifest = dict(schema='background_surface_v1', status='accepted' if accepted else 'rejected_preserved',
        frame='rectified_left_camera', units='meters', model=model, requested_model=chosen['surface_model'],
        sam3d_model_executed=False, sam3d_candidate_alignment=alignment, fallback_reasons=fallbacks,
        geometry='background_geometry.npz', inspection_obj='background_camera.obj', material='material.json',
        texture='texture.png', texture_under_foreground='nearest_visible_background_infill',
        physics_validated=False, independent_accuracy_measured=False, diagnostics=diagnostics, consistency=quality,
        acceptance_limits=dict(min_depth_support_fraction=.8, min_right_background_support_fraction=.5,
                               depth_tolerance_m=cfg['plane_tolerance_m']*2),
        selection_source=source, elapsed_seconds=time.monotonic()-started, input_sha256=protected,
        files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    save(out/'background.json', manifest)
    return manifest


def evaluate(vertices, images, masks, depth, support, K, baseline, tolerance):
    h, w = depth.shape; left = np.rint(project(vertices, K)).astype(int); right = np.rint(project(vertices, K, baseline)).astype(int)
    def inside(p): return (p[:, 0]>=0)&(p[:, 0]<w)&(p[:, 1]>=0)&(p[:, 1]<h)
    eligible = inside(left)
    ids = np.where(eligible)[0]; eligible[ids] &= support[left[ids, 1], left[ids, 0]]
    if not eligible.any(): return dict(depth_support_fraction=0., right_background_support_fraction=0., stereo_color_median_error=None)
    ids = np.where(eligible)[0]; errors = abs(vertices[ids, 2]-depth[left[ids, 1], left[ids, 0]])
    visible = inside(right); rid = np.where(visible & eligible)[0]
    visible[rid] &= ~masks['right'][right[rid, 1], right[rid, 0]]
    both = np.where(eligible & visible)[0]
    color = None
    if len(both): color = float(np.median(abs(images['left'][left[both, 1], left[both, 0]].astype(float)-images['right'][right[both, 1], right[both, 0]].astype(float)))/255.)
    return dict(depth_support_fraction=float((errors <= tolerance*2).mean()), depth_residual_median_m=float(np.median(errors)),
                right_background_support_fraction=float(len(both)/len(ids)), stereo_color_median_error=color,
                stereo_color_is_diagnostic_only=True, supported_vertices=len(ids))


def preview(path, vertices, faces, images, K, baseline, masks=None):
    left = project(vertices, K)
    edges = set()
    for f in faces:
        for i, j in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
            edges.add((min(int(i), int(j)), max(int(i), int(j))))
    selected = []
    for i, j in sorted(edges):
        a, b = left[i], left[j]
        vertical = abs(a[0]-b[0]) < 1 and min(a[0] % 32, 32-a[0] % 32) < 1
        horizontal = abs(a[1]-b[1]) < 1 and min(a[1] % 32, 32-a[1] % 32) < 1
        if len(faces) < 2000 or vertical or horizontal: selected.append((i, j))
    views = []
    for side in ['left', 'right']:
        im = Image.fromarray(images[side]); draw = ImageDraw.Draw(im); xy = project(vertices, K, baseline if side == 'right' else 0.)
        for i, j in selected:
            draw.line([tuple(map(float, xy[i])), tuple(map(float, xy[j]))], fill=(50, 220, 130), width=1)
        if masks is not None:
            pixels = np.array(im); pixels[masks[side]] = images[side][masks[side]]; im = Image.fromarray(pixels)
        views.append(im)
    full = Image.new('RGB', (views[0].width*2, views[0].height+26), '#111111')
    ImageDraw.Draw(full).text((8, 6), 'BACKGROUND CANDIDATE | complete LEFT / RIGHT | stereo consistency preview', fill='white')
    for i, im in enumerate(views): full.paste(im, (i*im.width, 26))
    full.save(path, quality=90)
