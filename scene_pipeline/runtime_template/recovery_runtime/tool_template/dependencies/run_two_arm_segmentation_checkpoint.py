"""One full-frame stereo SAM3 pass, preserving separate tool hypotheses."""
import argparse
import importlib.util
from pathlib import Path
import time

from complex_case_diagnostics import write_json
from reconstruction_tools import fingerprints
from refine_thread_sam3_v4 import build_model_without_cpu_cuda_cache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot', 'pipeline-tools', 'checkpoint', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    source = a.pipeline_tools / 'segment_needle_thread_with_sam3.py'
    protected = {'runner': Path(__file__), 'helper': source, 'checkpoint': a.checkpoint,
                 **{s: a.snapshot / (s + '_rect.png') for s in ('left', 'right')}}
    before = fingerprints(protected)
    report = dict(status='failed', input_sha256=before, seed=0, device=a.device,
                  full_frame=True, image_crop=None, views={}, independent_accuracy_measured=False,
                  training_eligible=False, geometry_generated=False)
    start = time.perf_counter()
    try:
        import numpy as np
        import torch
        from PIL import Image
        from contextlib import nullcontext
        import sam3.model_builder as builder
        from sam3.model.sam3_image_processor import Sam3Processor
        torch.manual_seed(0)
        np.random.seed(0)
        torch.set_num_threads(8)
        if a.device == 'cuda':
            # This is a brief shared-GPU job, not an idle-GPU allocation.
            free, total = torch.cuda.mem_get_info()
            if free < 6 * 1024**3:
                raise RuntimeError('Need 6 GiB free for a bounded shared-GPU pass')
            torch.cuda.set_per_process_memory_fraction(8 * 1024**3 / total)
            report.update(gpu=torch.cuda.get_device_name(), memory_cap_gib=8,
                          shared_gpu=True, free_memory_before_bytes=free)
        spec = importlib.util.spec_from_file_location('existing_segmentation', source)
        existing = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(existing)
        model = build_model_without_cpu_cuda_cache(builder, a.device)
        report['checkpoint_format'] = existing.load_image_checkpoint_strict(model, str(a.checkpoint))
        model = model.to(a.device).eval()
        processor = Sam3Processor(model, resolution=1008, device=a.device, confidence_threshold=.01)
        report.update(torch=torch.__version__, model_load_seconds=time.perf_counter()-start,
                      dtype='bfloat16 autocast' if a.device == 'cuda' else 'float32', resolution=1008)
        context = torch.autocast('cuda', dtype=torch.bfloat16) if a.device == 'cuda' else nullcontext()
        with torch.inference_mode(), context:
            for side in ('left', 'right'):
                with Image.open(a.snapshot / (side + '_rect.png')) as im:
                    image = im.convert('RGB')
                w, h = image.size
                state = processor.set_image(image)
                thread = existing.run_prompt_sweep(processor, state, ['suture thread'], h, w, .20, 0)
                needle = existing.run_prompt_sweep(processor, state, ['curved surgical suture needle'], h, w, .20, 0)
                tm, nm, overlap = existing.resolve_class_overlap(thread, needle)
                existing.save_mask(a.out / (side + '_thread_mask.png'), tm)
                existing.save_mask(a.out / (side + '_needle_mask.png'), nm)
                existing.save_overlay(a.out / (side + '_overlay.png'), np.asarray(image), tm, nm)
                # Tool hypotheses are not assigned to PSM1/PSM2 by image side.
                processor.reset_all_prompts(state)
                raw = processor.set_text_prompt(state=state, prompt='robotic surgical instrument')
                scores = existing.as_numpy(raw['scores']).reshape(-1)
                masks = existing.as_numpy(raw['masks_logits']).reshape(-1, h, w)
                indices = np.flatnonzero(scores >= .20)
                indices = sorted(indices, key=lambda i: -float(scores[i]))
                union = np.zeros((h, w), bool)
                instances = []
                for rank, index in enumerate(indices):
                    mask = masks[index] > .5
                    existing.save_mask(a.out / f'{side}_tool_hypothesis_{rank:02d}.png', mask)
                    union |= mask
                    instances.append(dict(rank=rank, score=float(scores[index]), pixels=int(mask.sum())))
                existing.save_mask(a.out / (side + '_occluder.png'), union)
                report['views'][side] = dict(image_size=[w, h], thread=existing.prediction_report(thread),
                    needle=existing.prediction_report(needle), overlap_resolved_pixels=overlap,
                    tool_prompt='robotic surgical instrument', tool_hypotheses=instances,
                    tool_identity='unassigned; requires inspection', thread_pixels=int(tm.sum()),
                    needle_pixels=int(nm.sum()))
                write_json(a.out / 'progress.json', report)
        if fingerprints(protected) != before:
            raise ValueError('Inputs or source changed during execution')
        report['status'] = 'segmentation_executed'
    except Exception as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        report['runtime_seconds'] = time.perf_counter() - start
        write_json(a.out / 'report.json', report)


if __name__ == '__main__':
    main()
