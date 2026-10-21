"""One bounded stereo SAM3 resegmentation proposal using the existing checkpoint.

This executable changes only derived thread masks. The supplied needle masks,
original scene, model checkpoint and calibration remain frozen. No prompt sweep
is hidden inside an agent action; one phrase/threshold is tested in both views.
"""
import argparse
import importlib.util
from pathlib import Path
import time
import traceback

from complex_case_diagnostics import write_json
from reconstruction_tools import fingerprints, sha256


def build_model_without_cpu_cuda_cache(builder,device):
    """Avoid SAM3's hardcoded CUDA-only position-cache precomputation on CPU.

    Its ordinary forward computes the same positional encoding on x.device.
    We change cache initialization only, inside this isolated subprocess, and
    restore the builder function even if construction fails.
    """
    kwargs={'device':'cpu','checkpoint_path':None,'load_from_HF':False,'enable_segmentation':True}
    if device!='cpu':return builder.build_sam3_image_model(**kwargs)
    original=builder._create_position_encoding
    def lazy_position_encoding(precompute_resolution=None):
        return original(precompute_resolution=None)
    builder._create_position_encoding=lazy_position_encoding
    try:return builder.build_sam3_image_model(**kwargs)
    finally:builder._create_position_encoding=original


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('left','right','checkpoint','pipeline-tools','out'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--prompt',choices=('suture thread','thin suture thread'),required=True)
    p.add_argument('--confidence-threshold',type=float,choices=(.20,.35),required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--expected-checkpoint-sha256',required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    source=a.pipeline_tools/'segment_needle_thread_with_sam3.py'
    before=fingerprints({'left':a.left,'right':a.right,'checkpoint':a.checkpoint,'source':source,'runner':__file__})
    if sha256(a.checkpoint)!=a.expected_checkpoint_sha256:raise ValueError('Wrong segmentation checkpoint')
    started=time.perf_counter()
    report={'status':'failed','parameters':{'prompt':a.prompt,'confidence_threshold':a.confidence_threshold},
            'input_sha256':before,'device':a.device,'seed':0,'views':{},'api_cost_usd':0.,
            'compute_cost_usd':None,'independent_accuracy_measured':False,'training_eligible':False}
    try:
        import numpy as np
        import torch
        from PIL import Image
        from sam3.model.sam3_image_processor import Sam3Processor
        import sam3.model_builder as builder
        torch.set_num_threads(8);torch.manual_seed(0);np.random.seed(0)
        spec=importlib.util.spec_from_file_location('existing_sam3_segmentation',source)
        existing=importlib.util.module_from_spec(spec);spec.loader.exec_module(existing)
        model=build_model_without_cpu_cuda_cache(builder,a.device)
        report.update(torch=torch.__version__,dtype='float32',resolution=1008,
                      cpu_position_cache='lazy_on_input_device' if a.device=='cpu' else 'existing_builder')
        report['checkpoint_format']=existing.load_image_checkpoint_strict(model,str(a.checkpoint))
        model=model.to(a.device).eval()
        processor=Sam3Processor(model,resolution=1008,device=a.device,confidence_threshold=.01)
        report['model_load_seconds']=time.perf_counter()-started
        with torch.inference_mode():
            for side in ('left','right'):
                start=time.perf_counter();image=Image.open(getattr(a,side)).convert('RGB');w,h=image.size
                state=processor.set_image(image)
                prediction=existing.run_prompt_sweep(processor,state,[a.prompt],h,w,a.confidence_threshold,0)
                if prediction['mask'].shape!=(h,w) or not np.isfinite(prediction['candidate_scores']).all():
                    raise ValueError('SAM3 returned invalid mask shape or nonfinite scores')
                existing.save_mask(a.out/(side+'_thread_mask.png'),prediction['mask'])
                report['views'][side]={**existing.prediction_report(prediction),'mask_shape':[h,w],
                                      'mask_pixels':int(prediction['mask'].sum()),
                                      'runtime_seconds':time.perf_counter()-start}
        if fingerprints({'left':a.left,'right':a.right,'checkpoint':a.checkpoint,'source':source,'runner':__file__})!=before:
            raise ValueError('Segmentation inputs/sources changed during execution')
        report['status']='proposal_generated'
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    report['runtime_seconds']=time.perf_counter()-started;write_json(a.out/'report.json',report)
    if report['status']!='proposal_generated':raise SystemExit(2)


if __name__=='__main__':main()
