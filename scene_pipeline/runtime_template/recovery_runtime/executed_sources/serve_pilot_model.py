"""Offline model-only file queue; one model load, measured real inference.

Separate from reconstruction dependencies. Never execute model-produced code.
Image paths and hashes are checked before/after inference in both processes.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time

from reconstruction_tools import fingerprints
from run_compact_agent import MeteredGemma


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.rename(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--gemma-runner', type=Path, required=True)
    a=p.parse_args(); a.queue.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    try:
        import torch
        from transformers import pipeline, GenerationConfig
        torch.set_num_threads(8); torch.manual_seed(0)
        spec=importlib.util.spec_from_file_location('existing_gemma_runner', a.gemma_runner)
        backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)
        backend.disable_python_native_bmm(torch)
        model=MeteredGemma.__new__(MeteredGemma)
        model.backend=backend
        model.pipe=pipeline(task='image-text-to-text',model=a.model,device_map='cpu',dtype=torch.float32,
                            model_kwargs={'attn_implementation':'sdpa','low_cpu_mem_usage':True})
        backend.normalize_gemma4_image_feature_output(model.pipe.model)
        model.pipe.image_processor.max_soft_tokens=560
        model.generation=GenerationConfig.from_pretrained(a.model)
        model.generation.max_new_tokens=512; model.generation.do_sample=False
        model.generation.temperature=model.generation.top_k=model.generation.top_p=None
        manifest={'status':'ready','load_seconds':time.perf_counter()-started,
                  'device':'CPU','dtype':'float32','threads':8,'seed':0,'sampling':False,
                  'torch':torch.__version__,'model':a.model,'api_cost_usd':0,'compute_cost_usd':None,
                  'source_fingerprints':fingerprints({'service':__file__,'gemma_runner':a.gemma_runner})}
        write(a.queue/'service_manifest.json',manifest)
        print(json.dumps(manifest),flush=True)
        processed=set()
        while not (a.queue/'STOP').exists():
            files=[f for f in sorted(a.queue.glob('request_*.json'))
                   if not f.name.endswith('.response.json') and f not in processed]
            if not files:
                time.sleep(.25);continue
            f=files[0];processed.add(f);request=json.loads(f.read_text());start=time.perf_counter()
            response={'status':'failed','actual_model_used':False,'usage':None}
            try:
                messages=request['messages']
                images={str(i):item['url'] for i,item in enumerate(
                    x for m in messages for x in m['content'] if x['type']=='image')}
                for image in images.values():
                    path=Path(image).resolve()
                    if not any(path.is_relative_to(root) for root in
                               map(Path,('/workspace/inputs','/recordings','/prior','/checkpoint','/pilot'))):
                        raise ValueError('Image is outside declared data/output mounts')
                if fingerprints(images)!=request['image_hashes']:
                    raise ValueError('Request image provenance mismatch')
                raw=model(messages)
                f.with_suffix('.raw.txt').write_text(raw+'\n')
                response.update(actual_model_used=True,usage=model.last_usage)
                if fingerprints(images)!=request['image_hashes']:
                    raise ValueError('Image changed during inference')
                response['status']='completed'
            except Exception as error:
                response['error']=f'{type(error).__name__}: {error}'
            response['runtime_seconds']=time.perf_counter()-start
            write(f.with_suffix('.response.json'),response)
            print(json.dumps({'request':f.name,**response}),flush=True)
    except Exception as error:
        write(a.queue/'service_failure.json',{'error':f'{type(error).__name__}: {error}',
                                             'runtime_seconds':time.perf_counter()-started})
        raise


if __name__=='__main__':main()
