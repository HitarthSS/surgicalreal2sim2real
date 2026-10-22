"""Copy a verified background into a reconstruction handoff without source changes."""
from pathlib import Path
import json
import shutil
import numpy as np
from .background import sha, write_obj


def export_background(scene, output, transform):
    source = Path(scene)/'background'; target = Path(output)/'background'
    manifest_path = source/'background.json'
    if not manifest_path.exists(): return None
    report = json.loads(manifest_path.read_text())
    for rel, expected in report['files'].items():
        path = (source/rel).resolve()
        if not path.is_relative_to(source.resolve()): raise ValueError('Background artifact escapes source directory')
        if sha(path) != expected: raise ValueError('Background artifact changed: '+rel)
    shutil.copytree(source, target)
    if report['status'] != 'accepted':
        return dict(status=report['status'], manifest='background/background.json', geometry=None)
    with np.load(target/'background_geometry.npz', allow_pickle=False) as data:
        arrays = {k:data[k].copy() for k in data.files}
    T = np.asarray(transform, float)
    vertices = arrays['vertices_camera_m'] @ T[:3, :3].T + T[:3, 3]
    arrays.update(vertices_newton_m=vertices, T_newton_from_camera=T)
    np.savez_compressed(target/'background_geometry_newton.npz', **arrays)
    write_obj(target/'background_newton_m.obj', vertices, arrays['faces'], arrays['uv'])
    # Keep the original camera-space provenance manifest intact. Delivery has its own hashes.
    delivered = {str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file()}
    (target/'delivery_verified.json').write_text(json.dumps(dict(source_manifest_sha256=sha(manifest_path),files=delivered),indent=2)+'\n')
    return dict(status='accepted', manifest='background/background.json', material='background/material.json',
                geometry='background/background_geometry_newton.npz', inspection_obj='background/background_newton_m.obj',
                frame='newton', texture='background/texture.png', static_background=True)
