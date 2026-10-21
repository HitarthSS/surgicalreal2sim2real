"""Execute one explicit stereo SAM3 request with text and native box guidance."""
from pathlib import Path
from contextlib import nullcontext
import argparse
import importlib.util
import json
import sys
import time
from geometry_repairs import select_native_tool_instances

sys.path.insert(0, str(Path(__file__).resolve().parent / 'frozen_sources'))
from reconstruction_tools import fingerprints, sha256
from refine_thread_sam3_v4 import build_model_without_cpu_cuda_cache
from tool_perception_protocol import validate_request, validate_box


def main():
    parser = argparse.ArgumentParser()
    for name in ['case', 'request', 'checkpoint', 'pipeline', 'out']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    import sam3.model_builder as builder
    case = json.loads(args.case.read_text())
    sizes = {side: list(Image.open(case['inputs'][side]).size) for side in ['left', 'right']}
    request = validate_request(json.loads(args.request.read_text()), sizes)
    helper = args.pipeline / 'segment_needle_thread_with_sam3.py'
    protected = dict(case=args.case, request=args.request, checkpoint=args.checkpoint, helper=helper,
                     runner=Path(__file__), protocol=Path(__file__).with_name('tool_perception_protocol.py'), **case['inputs'])
    protected['instance_selector']=Path(__file__).with_name('geometry_repairs.py')
    before = fingerprints(protected)
    report = dict(status='failed', request=request, source_fingerprints=before, full_frame=True, image_crop=None,
                  actual_model_producer=True, independent_accuracy_measured=False, views={}, seed=0, memory_cap_gib=8)
    start = time.perf_counter()
    try:
        assert sha256(args.checkpoint) == '64617502feb29df7ecd4b0ee831aa512f86f0b86c67cdd3af1a85d9e2802b9f7'
        torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(8)
        free, total = torch.cuda.mem_get_info()
        if free < 6 * 1024**3:
            raise RuntimeError('Need6GiBfree; shared GPU admission failed')
        torch.cuda.set_per_process_memory_fraction(8 * 1024**3 / total)
        spec = importlib.util.spec_from_file_location('existing_segmentation', helper)
        existing = importlib.util.module_from_spec(spec); spec.loader.exec_module(existing)
        model = build_model_without_cpu_cuda_cache(builder, 'cuda')
        report['checkpoint_format'] = existing.load_image_checkpoint_strict(model, str(args.checkpoint))
        model = model.to('cuda').eval()
        processor = Sam3Processor(model, resolution=1008, device='cuda', confidence_threshold=.01)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for side in ['left', 'right']:
                image = Image.open(case['inputs'][side]).convert('RGB'); w, h = image.size
                state = processor.set_image(image)
                report['views'][side] = {}
                for target, views in request['objects'].items():
                    params = views[side]
                    processor.reset_all_prompts(state)
                    raw = processor.set_text_prompt(state=state, prompt=params['prompt'])
                    normalized = []
                    for box in params['boxes']:
                        norm = validate_box(box, [w, h]); normalized.append(dict(cxcywh=norm, positive=box['positive']))
                        raw = processor.add_geometric_prompt(box=norm, label=box['positive'], state=state)
                    prediction = existing.unpack_prediction(raw, h, w, params['confidence_threshold'], 0)
                    native_selection=None
                    if target in ('tool_A','tool_B'):
                        probabilities=existing.as_numpy(raw['masks_logits']).reshape(-1,h,w)
                        native_scores=existing.as_numpy(raw['scores']).reshape(-1)
                        existing.save_mask(args.out/f'{side}_{target}_union_mask.png',prediction['mask'])
                        np.savez_compressed(args.out/f'{side}_{target}_native_instances.npz',masks=probabilities>.5,scores=native_scores)
                        prediction['mask'],native_selection=select_native_tool_instances(probabilities,native_scores,params['boxes'],'left' if target=='tool_A' else 'right',params['confidence_threshold'])
                        chosen=native_selection['selected_instance']['index']
                        prediction['confidence']=probabilities[chosen]*native_scores[chosen]
                    if prediction['mask'].shape != (h, w) or not np.isfinite(prediction['candidate_scores']).all():
                        raise ValueError('Invalid SAM3 mask or scores')
                    existing.save_mask(args.out / f'{side}_{target}_mask.png', prediction['mask'])
                    np.save(args.out / f'{side}_{target}_confidence.npy', prediction['confidence'])
                    report['views'][side][target] = dict(mask_pixels=int(prediction['mask'].sum()),
                        candidate_scores=prediction['candidate_scores'].tolist(), normalized_boxes=normalized,
                        native_instance_selection=native_selection,
                        mask_sha256=sha256(args.out / f'{side}_{target}_mask.png'))
        assert fingerprints(protected) == before, 'Protected input changed'
        report['status'] = 'proposal_generated'
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['runtime_seconds'] = time.perf_counter()-start
        (args.out / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
