"""Audit estimated native capsule overlaps without modifying scene geometry."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np


def segment_distances(a, b):
    """All pairwise finite-segment distances, including clipped endpoints.

For two nonparallel lines an interior stationary point is a candidate;
the four constrained boundaries also cover parallel and disjoint cases.
"""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    for points in (a, b):
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError('Require finite 3D polylines')
        if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 1e-10):
            raise ValueError('Zero-length segments are unsupported')
    u, v = np.diff(a, axis=0), np.diff(b, axis=0)
    w = a[:-1, None, :] - b[None, :-1, :]
    aa = np.sum(u*u, axis=1)[:, None]
    bb = u @ v.T
    cc = np.sum(v*v, axis=1)[None, :]
    dd = np.einsum('ijk,ik->ij', w, u)
    ee = np.einsum('ijk,jk->ij', w, v)
    def distance2(s, t):
        d = w + s[..., None] * u[:, None, :] - t[..., None] * v[None, :, :]
        return np.sum(d*d, axis=2)
    zeros, ones = np.zeros_like(bb), np.ones_like(bb)
    candidates = [distance2(zeros, np.clip(ee/cc, 0, 1)),
                  distance2(ones, np.clip((ee+bb)/cc, 0, 1)),
                  distance2(np.clip(-dd/aa, 0, 1), zeros),
                  distance2(np.clip((bb-dd)/aa, 0, 1), ones)]
    determinant = aa*cc-bb*bb
    nonparallel = determinant > 1e-12*aa*cc
    safe = np.where(nonparallel, determinant, 1.)
    s, t = (bb*ee-cc*dd)/safe, (aa*ee-bb*dd)/safe
    interior = nonparallel & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    candidates.append(np.where(interior, distance2(s, t), np.inf))
    return np.sqrt(np.maximum(np.minimum.reduce(candidates), 0.))


def clearance(thread, needle, thread_radius_m, needle_radius_m):
    for radius in (thread_radius_m, needle_radius_m):
        if not np.isfinite(radius) or not 0 < radius <= .005:
            raise ValueError('Explicit radii in meters must be positive and at most 5 mm')
    distances = segment_distances(thread, needle)
    gaps = distances - thread_radius_m - needle_radius_m
    index = np.unravel_index(int(np.argmin(gaps)), gaps.shape)
    return {'minimum_centerline_separation_m': float(distances[index]),
            'minimum_capsule_surface_gap_m': float(gaps[index]),
            'penetrating_segment_pairs': int(np.count_nonzero(gaps < -1e-9)),
            'nearest_thread_segment': int(index[0]), 'nearest_needle_segment': int(index[1]),
            'maximum_geometric_penetration_m': max(0., float(-gaps.min())),
            'no_initial_needle_thread_penetration': bool(np.all(gaps >= -1e-9))}


def native_resampler(path):
    """Reuse only the existing pure numpy resampler, without importing Newton."""
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'resample_arclength']
    if len(nodes) != 1:
        raise ValueError('Expected one existing native resample_arclength implementation')
    namespace = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['resample_arclength']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--thread-radius-m', type=float, required=True)
    p.add_argument('--needle-collision-segments', type=int, default=24)
    a = p.parse_args()
    if a.out.exists() or not 2 <= a.needle_collision_segments <= 64:
        raise ValueError('Preserve prior outputs and use 2–64 needle collision segments')
    with np.load(a.bundle, allow_pickle=False) as z:
        thread = z['points_newton_m'].copy()
        needle = z['needle_points_newton_m'].copy()
        needle_radius = float(z['needle_tube_radius_m'])
    samples = min(a.needle_collision_segments, len(needle)-1) + 1
    collision_needle = native_resampler(a.native)(needle, samples)
    result = clearance(thread, collision_needle, a.thread_radius_m, needle_radius)
    result.update(bundle_sha256=hashlib.sha256(a.bundle.read_bytes()).hexdigest(),
                  native_sha256=hashlib.sha256(a.native.read_bytes()).hexdigest(),
                  auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  thread_radius_m=a.thread_radius_m, needle_radius_m=needle_radius,
                  needle_collision_segments=samples-1, thread_segments=len(thread)-1,
                  radii_evidence='unmeasured_sensitivity_values',
                  physics_validated=False, training_eligible=False,
                  independent_accuracy_measured=False,
                  limitations=['Estimated geometry and unmeasured radii; not a physical collision measurement.',
                               'Reuses native needle resampling; no geometry repair or contact suppression.',
                               'No tool, self-contact, attachment-joint or contact-margin analysis.',
                               'A clear initial gap does not establish dynamic validity.'])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
