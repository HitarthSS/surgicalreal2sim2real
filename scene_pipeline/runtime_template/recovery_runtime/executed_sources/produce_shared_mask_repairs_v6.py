"""Actual bounded SAM3 proposals shared by controllers with exact provenance.

One model/image encoding is reused; each phrase/threshold remains an explicit
operation. Cached geometry is never substituted for a model-produced mask.
"""
import argparse
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import platform
import time

from complex_case_diagnostics import write_json
from reconstruction_tools import fingerprints,sha256
from refine_thread_sam3_v4 import build_model_without_cpu_cuda_cache


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['input-root','pipeline-tools','checkpoint','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--case-ids',nargs='+',required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    import sam3.model_builder as builder
    from sam3.model.sam3_image_processor import Sam3Processor
    torch.manual_seed(0);np.random.seed(0);torch.set_num_threads(8)
    free,total=torch.cuda.mem_get_info()
    if free<6*1024**3:raise RuntimeError('Need6GiBfree for bounded shared-GPU inference')
    torch.cuda.set_per_process_memory_fraction(8*1024**3/total)
    helper=a.pipeline_tools/'segment_needle_thread_with_sam3.py'
    spec=importlib.util.spec_from_file_location('existing_segmentation',helper)
    existing=importlib.util.module_from_spec(spec);spec.loader.exec_module(existing)
    checkpoint_sha=sha256(a.checkpoint)
    started=time.perf_counter();model=build_model_without_cpu_cuda_cache(builder,'cuda')
    fmt=existing.load_image_checkpoint_strict(model,str(a.checkpoint));model=model.to('cuda').eval()
    processor=Sam3Processor(model,resolution=1008,device='cuda',confidence_threshold=.01)
    runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
        model_load_seconds=time.perf_counter()-started,checkpoint_format=fmt,checkpoint_sha256=checkpoint_sha,
        producer_sha256=sha256(__file__),helper_sha256=sha256(helper),
        dtype='bfloat16 autocast',resolution=1008,seed=0,memory_cap_gib=8,
        shared_gpu=True,device=torch.cuda.get_device_name(),image_crop=None)
    write_json(a.out/'runtime.json',runtime)
    cards={'segment.precision':dict(target='thread',prompt='suture thread',confidence_threshold=.35),
        'segment.recall':dict(target='thread',prompt='thin suture thread',confidence_threshold=.20),
        'occluder.sam3.instrument':dict(target='occluder',prompt='robotic surgical instrument',confidence_threshold=.20),
        'occluder.sam3.grasper':dict(target='occluder',prompt='surgical grasper',confidence_threshold=.20)}
    rows=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for cid in a.case_ids:
            raw=a.input_root/cid;folder=a.out/cid;folder.mkdir()
            reports={};began=time.perf_counter()
            images={s:raw/(s+'_rect.png') for s in ['left','right']}
            input_hash=fingerprints(images)
            for card,parameters in cards.items():
                reports[card]=dict(status='proposal_generated',parameters=parameters,views={},
                    actual_model_producer=True,input_sha256=input_hash,checkpoint_sha256=checkpoint_sha,
                    producer_sha256=runtime['producer_sha256'],helper_sha256=runtime['helper_sha256'],
                    runtime_sha256=sha256(a.out/'runtime.json'),full_frame=True,image_crop=None,
                    source_masks_modified=False,independent_accuracy_measured=False,training_eligible=False)
                (folder/card).mkdir()
            for side in ['left','right']:
                image=Image.open(images[side]).convert('RGB');w,h=image.size
                state=processor.set_image(image)
                for card,params in cards.items():
                    prediction=existing.run_prompt_sweep(processor,state,[params['prompt']],h,w,params['confidence_threshold'],0)
                    mask=prediction['mask'].copy();rejections=[]
                    if params['target']=='occluder':
                        # Generic mask sanity only; semantic correctness still needs image review.
                        labels_count,labels=cv2.connectedComponents(mask.astype('uint8'))
                        filtered=np.zeros_like(mask)
                        for label in range(1,labels_count):
                            part=labels==label;area=int(part.sum())
                            width=float(cv2.distanceTransform(part.astype('uint8'),cv2.DIST_L2,5).max())
                            boundary=bool(part[0].any() or part[-1].any() or part[:,0].any() or part[:,-1].any())
                            if 256<=area<=h*w*.30 and width>=3 and boundary:filtered|=part
                            else:rejections.append(dict(area_px=area,interior_radius_px=width,touches_boundary=boundary))
                        mask=filtered
                    target=folder/card/(side+'_mask.png');existing.save_mask(target,mask)
                    reports[card]['views'][side]=dict(**existing.prediction_report(prediction),
                        mask_pixels=int(mask.sum()),sanity_rejected_components=rejections)
                    overlay=np.asarray(image).copy();overlay[mask]=(overlay[mask]*.55+np.array([250,70,210])*.45).astype('uint8')
                    Image.fromarray(overlay).save(folder/card/(side+'_overlay.png'))
            for card,report in reports.items():
                report['artifacts']={s:dict(relative_path=str(Path(cid)/card/(s+'_mask.png')),
                    sha256=sha256(folder/card/(s+'_mask.png'))) for s in ['left','right']}
                write_json(folder/card/'report.json',report)
                rows.append(dict(case_id=cid,card=card,views=report['views']))
            if fingerprints(images)!=input_hash:raise ValueError('Input image changed')
            print(json.dumps(dict(case_id=cid,status='actual_mask_proposals_generated',seconds=time.perf_counter()-began)),flush=True)
    write_json(a.out/'summary.json',dict(rows=rows,runtime_seconds=time.perf_counter()-started,
        actual_model_proposals=len(rows),training_eligible=False,independent_accuracy_measured=False))


if __name__=='__main__':main()
