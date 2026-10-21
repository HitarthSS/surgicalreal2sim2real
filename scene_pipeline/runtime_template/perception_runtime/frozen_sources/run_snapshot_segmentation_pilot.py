"""Reuse existing SAM3 functions with one checkpoint load for a small pilot."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--pipeline-tools', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--case-ids', nargs='+', required=True)
    a = p.parse_args()
    if not 1 <= len(a.case_ids) <= 5 or len(set(a.case_ids)) != len(a.case_ids):
        raise ValueError('Small pilot only: one to five distinct cases')
    for case_id in a.case_ids:
        if Path(case_id).name != case_id:
            raise ValueError('Case identifiers must be directory names')
        for view in ('left', 'right'):
            if not (a.input_root / case_id / (view + '_rect.png')).is_file():
                raise FileNotFoundError(case_id + '/' + view)
    a.out.mkdir(parents=True, exist_ok=False)
    source = a.pipeline_tools / 'segment_needle_thread_with_sam3.py'
    spec = importlib.util.spec_from_file_location('existing_sam3_tools', source)
    existing = importlib.util.module_from_spec(spec); spec.loader.exec_module(existing)
    import numpy as np
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Allocate exactly one idle GPU')
    torch.manual_seed(0); np.random.seed(0)
    manifest = {'existing_source_sha256': sha256(source), 'runner_sha256': sha256(__file__),
                'checkpoint_sha256': sha256(a.checkpoint), 'checkpoint': str(a.checkpoint),
                'settings': {'thread_prompts': ['suture thread'],
                             'needle_prompts': ['curved surgical suture needle'],
                             'candidate_threshold': .01, 'confidence_threshold': .20,
                             'resolution': 1008, 'thread_max_instances': 0, 'needle_max_instances': 0},
                'seed': 0, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
                'case_ids': a.case_ids, 'model_load_count': 1,
                'independent_accuracy_measured': False, 'physics_run': False,
                'api_cost_usd': 0., 'compute_cost_usd': None}
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    started = time.perf_counter()
    model = build_sam3_image_model(device='cpu', checkpoint_path=None,
                                   load_from_HF=False, enable_segmentation=True)
    checkpoint_format = existing.load_image_checkpoint_strict(model, str(a.checkpoint))
    model = model.to('cuda').eval()
    processor = Sam3Processor(model, resolution=1008, device='cuda', confidence_threshold=.01)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model_load_seconds = time.perf_counter() - started
    rows = []
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for case_id in a.case_ids:
            out = a.out / case_id; out.mkdir()
            row = {'case_id': case_id, 'views': {}, 'checkpoint_format': checkpoint_format}
            start = time.perf_counter()
            try:
                for view in ('left', 'right'):
                    path = a.input_root / case_id / (view + '_rect.png')
                    image, thread, needle, report = existing.segment_image(
                        processor, path, ['suture thread'], ['curved surgical suture needle'], .20, 0, 0)
                    existing.save_mask(out / (view + '_thread_mask.png'), thread)
                    existing.save_mask(out / (view + '_needle_mask.png'), needle)
                    existing.save_overlay(out / (view + '_overlay.png'), image, thread, needle)
                    row['views'][view] = report
                row['status'] = 'segmentation_executed'
            except Exception as error:
                row.update(status='failed', error=type(error).__name__ + ': ' + str(error))
            row.update(runtime_seconds=time.perf_counter() - start,
                       independent_accuracy_measured=False, training_eligible=False)
            (out / 'report.json').write_text(json.dumps(row, indent=2) + '\n')
            rows.append(row)
            print(json.dumps(row), flush=True)
    summary = {**manifest, 'model_load_seconds': model_load_seconds,
               'runtime_seconds': time.perf_counter() - started, 'rows': rows,
               'limitations': ['Segmentation confidence is not accuracy.',
                               'Selected development cases; no claim about all 47 snapshots.',
                               'No new geometry, tool fitting, Newton rollout or policy demonstrations.']}
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
