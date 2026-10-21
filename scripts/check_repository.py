#!/usr/bin/env python3
"""Check the shareable source tree without contacting APIs or loading models."""
from pathlib import Path
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DENIED = {'.pt', '.pth', '.ckpt', '.safetensors', '.onnx', '.sif', '.npz', '.npy',
          '.bag', '.db3', '.mcap', '.pem', '.key', '.zip', '.log'}
SECRET = re.compile(rb'(?:sk-(?:proj-)?[A-Za-z0-9_-]{24,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
SKIP = {'.git', '__pycache__', '.venv', 'venv', 'build', 'dist', '.pytest_cache', '.ruff_cache'}
errors = []
count = 0
for path in ROOT.rglob('*'):
    relative = path.relative_to(ROOT)
    if not path.is_file() or set(relative.parts) & SKIP or path.suffix in {'.pyc', '.pyo'}:
        continue
    count += 1
    if path.suffix.lower() in DENIED or path.name.startswith('.env') or path.name.endswith('.local.json'):
        errors.append(f'Non-source file: {relative}')
    if path.stat().st_size > 5 * 1024 * 1024:
        errors.append(f'Large file: {relative}')
    if SECRET.search(path.read_bytes()):
        errors.append(f'Possible embedded credential: {relative}')
    if path.suffix == '.json':
        try:
            json.loads(path.read_text())
        except (ValueError, UnicodeError):
            errors.append(f'Invalid JSON: {relative}')
if errors:
    print('\n'.join(errors), file=sys.stderr)
    raise SystemExit(1)
print(f'Checked {count} source/configuration files; no model assets, local secrets or generated arrays found.')
