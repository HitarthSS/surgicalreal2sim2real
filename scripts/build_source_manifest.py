#!/usr/bin/env python3
"""Record hashes of shareable sources; generated data and Git internals are excluded."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {'.git', '__pycache__', '.venv', 'venv', 'build', 'dist', '.pytest_cache', '.ruff_cache'}
target = ROOT / 'SOURCE_MANIFEST.json'
files = {
    str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in ROOT.rglob('*')
    if path.is_file() and path != target and not set(path.relative_to(ROOT).parts) & EXCLUDE
    and path.suffix not in {'.pyc', '.pyo'} and path.name != '.DS_Store'
}
target.write_text(json.dumps(dict(schema='source_manifest_v1', files=files), indent=2) + '\n')
print(f'Recorded {len(files)} source/configuration hashes.')
