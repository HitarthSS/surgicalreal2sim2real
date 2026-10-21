#!/usr/bin/env python3
"""Pause the pipeline for browser-based mask review, then resume on save."""
import argparse
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import secrets
import shutil
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np


def fingerprint(paths):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def validate_jaw_tips(points, shape):
    tips = np.asarray(points, dtype=float)
    if tips.shape != (2, 2) or not np.isfinite(tips).all():
        raise ValueError('Mark both jaw tips before saving')
    height, width = shape[:2]
    if np.any(tips < 0) or np.any(tips[:, 0] > width - 1) or np.any(tips[:, 1] > height - 1):
        raise ValueError('Jaw tips must be inside the image')
    return tips


def load_reviewed_jaw_tips(stamp, inputs, shape):
    if not stamp.exists():
        raise ValueError('Finalize the tool mask and mark both jaw tips in mask review first')
    review = json.loads(stamp.read_text())
    if review.get('files') != fingerprint(inputs):
        raise ValueError('Image or tool mask changed; reopen mask review and confirm the jaw tips')
    return validate_jaw_tips(review.get('jaw_tips_xy'), shape)


def serve(args):
    if args.kind != 'tool':
        raise ValueError('Only tool masks can be edited')
    directory = args.mask_dir
    views = [('left', args.left)]
    classes = ['tool']
    paths, images = {}, {}
    for view, path in views:
        images[view] = cv2.imread(str(path))
        if images[view] is None:
            raise ValueError(f'Cannot read image: {path}')
        for label in classes:
            key = f'{view}_{label}'
            paths[key] = directory / 'tool_mask.png'
            mask = cv2.imread(str(paths[key]), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != images[view].shape[:2]:
                raise ValueError(f'Missing or mismatched mask: {paths[key]}')
    inputs = list(paths.values()) + [path for _, path in views]
    stamp = directory / 'manual_review.json'
    initial = fingerprint(inputs)
    tips = None
    try:
        tips = load_reviewed_jaw_tips(stamp, inputs, images['left'].shape).tolist()
    except (ValueError, TypeError):
        pass
    if tips is not None and not args.force:
        print('Tool mask and jaw tips already reviewed; continuing.', flush=True)
        return
    token = secrets.token_urlsafe(24)
    state = {'finished': False, 'saved': False}
    colors = {'tool': (0, 165, 255), 'thread': (255, 220, 0), 'needle': (220, 50, 220)}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *unused):
            pass

        def reply(self, code, data, mime='application/json'):
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            return parse_qs(urlparse(self.path).query).get('token') == [token]

        def do_GET(self):
            if not self.authorized():
                return self.reply(403, b'{}')
            route = urlparse(self.path).path
            if route == '/':
                return self.reply(200, Path(__file__).with_suffix('.html').read_bytes(), 'text/html')
            if route == '/data':
                return self.reply(200, json.dumps(dict(views=[v for v, _ in views], classes=classes,
                                                     title=args.title, jaw_tips_xy=tips)).encode())
            key = route.rsplit('/', 1)[-1]
            if route.startswith('/image/') and key in images:
                data = cv2.imencode('.png', images[key])[1].tobytes()
            elif route.startswith('/mask/') and key in paths:
                data = paths[key].read_bytes()
            else:
                return self.reply(404, b'{}')
            self.reply(200, data, 'image/png')

        def do_POST(self):
            if not self.authorized():
                return self.reply(403, b'{}')
            route = urlparse(self.path).path
            if route == '/cancel':
                state['finished'] = True
                return self.reply(200, b'{}')
            if route != '/save':
                return self.reply(404, b'{}')
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 40_000_000:
                    raise ValueError('Invalid payload size')
                payload = json.loads(self.rfile.read(length))
                if set(payload) != set(paths) | {'jaw_tips_xy'}:
                    raise ValueError('Save must include the tool mask and both jaw tips')
                saved_tips = validate_jaw_tips(payload['jaw_tips_xy'], images['left'].shape).tolist()
                if fingerprint(inputs) != initial:
                    raise ValueError('Source files changed during editing; cancel and reopen')
                decoded = {}
                for key in paths:
                    data = payload[key]
                    raw = base64.b64decode(data.split(',', 1)[1], validate=True)
                    mask = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
                    if mask is None or mask.shape != images[key.split('_')[0]].shape[:2]:
                        raise ValueError(f'Wrong image dimensions for {key}')
                    decoded[key] = (mask > 127).astype(np.uint8) * 255
                    if not np.any(decoded[key]):
                        raise ValueError('Tool mask cannot be empty')
                for key, path in paths.items():
                    backup = path.with_name(path.stem + '.sam3.png')
                    if not backup.exists():
                        shutil.copy2(path, backup)
                    temporary = path.with_name(path.stem + '.reviewing.png')
                    if not cv2.imwrite(str(temporary), decoded[key]):
                        raise OSError(f'Could not save {path}')
                    temporary.replace(path)
                for view, _ in views:
                    overlay = images[view].copy()
                    for label in classes:
                        selected = decoded[f'{view}_{label}'] > 0
                        overlay[selected] = (overlay[selected] * .5 + np.array(colors[label]) * .5).astype(np.uint8)
                    name = 'tool_overlay.png'
                    if not cv2.imwrite(str(directory / name), overlay):
                        raise OSError(f'Could not save {name}')
                report = directory / 'report.json'
                if report.exists():
                    data = json.loads(report.read_text())
                    data['manual_review'] = dict(approved=True, areas_px={k: int(np.count_nonzero(v)) for k, v in decoded.items()},
                                                note='Model scores and original metrics precede manual edits.')
                    report.write_text(json.dumps(data, indent=2) + '\n')
                temporary_stamp = stamp.with_suffix('.reviewing.json')
                temporary_stamp.write_text(json.dumps({'files': fingerprint(inputs),
                                                       'jaw_tips_xy': saved_tips}, indent=2) + '\n')
                temporary_stamp.replace(stamp)
                state.update(finished=True, saved=True)
                self.reply(200, b'{"saved":true}')
            except Exception as exc:
                self.reply(400, json.dumps({'error': str(exc)}).encode())

    server = HTTPServer((args.host, args.port), Handler)
    print(f'Waiting for mask review: http://localhost:{args.port}/?token={token}', flush=True)
    print('From your Mac, forward this port with: ssh -N -L 8765:127.0.0.1:8765 hitarth@blackdragon1', flush=True)
    try:
        while not state['finished']:
            server.handle_request()
    finally:
        server.server_close()
    if not state['saved']:
        raise SystemExit('Mask review cancelled; pipeline stopped.')
    print('Tool mask and jaw tips saved; continuing pipeline.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--left', type=Path, required=True)
    p.add_argument('--mask-dir', type=Path, required=True)
    p.add_argument('--kind', choices=['tool'], required=True)
    p.add_argument('--title', default='Mask Review')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--force', action='store_true')
    serve(p.parse_args())
