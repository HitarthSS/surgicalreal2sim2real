"""Execute both preregistered SAM3 cards on two frozen development pairs.

Only successful actual-model proposals enter the shared immutable cache. The
matched controller pilot is a separate step; validation is not scene accuracy.
"""
import argparse
import json
from pathlib import Path
import platform
import time

from complex_case_diagnostics import write_json
from complex_refiner_tools_v4 import SharedComplexTools
from reconstruction_tools import sha256
from sam3_proposal_cache_v4 import cache_key,record_proposal


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input-root','pipeline-tools','original-src','sam3-config','out','runtime-manifest'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--container-image-digest',required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    config=json.loads(a.sam3_config.read_text())
    if config.get('proposal_cache'):raise ValueError('Validation must execute actual SAM3, never reuse a cache')
    import numpy as np
    import torch
    import sam3.model_builder as builder
    import sam3.model.position_encoding as position
    if config['device']!='cpu' or torch.cuda.is_available():raise ValueError('This preregistered validation requires CPU-only inference')
    runtime={'container_image_digest':a.container_image_digest,'python':platform.python_version(),
             'numpy':np.__version__,'torch':torch.__version__,'device':'cpu','dtype':'float32',
             'position_cache':'lazy on input device; CUDA-only precompute disabled',
             'sam3_builder_sha256':sha256(builder.__file__),'position_encoding_sha256':sha256(position.__file__),
             'checkpoint_sha256':config['sha256'],'seed':0,'resolution':1008}
    write_json(a.runtime_manifest,runtime)
    started=time.perf_counter();rows=[]
    for case_id in ('snapshot_20261009_115511','snapshot_20261009_121654'):
        tools=SharedComplexTools(a.input_root/case_id/'case.json',a.pipeline_tools,a.original_src,sam3=config)
        for card,parameters in [('precision',{'prompt':'suture thread','confidence_threshold':.35}),
                                ('recall',{'prompt':'thin suture thread','confidence_threshold':.20})]:
            folder=a.out/case_id/card;before=time.perf_counter()
            row={'case_id':case_id,'card':card,'parameters':parameters,'actual_model_attempt':True}
            try:
                result=tools.resegment_thread(parameters,{},folder)
                key=cache_key(tools.inputs,config['sha256'],Path(__file__).with_name('refine_thread_sam3_v4.py'),
                              a.pipeline_tools/'segment_needle_thread_with_sam3.py',a.runtime_manifest,parameters)
                record_proposal(a.out,folder/'backend',key)
                row.update(status='actual_proposal_validated',result=result)
            except Exception as exc:row.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            row['runtime_wall_seconds']=time.perf_counter()-before;rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='result'}),flush=True)
    summary={'status':'validated' if all(r['status']=='actual_proposal_validated' for r in rows) else 'failed',
             'rows':rows,'runtime_wall_seconds':time.perf_counter()-started,'runtime':runtime,
             'actual_sam3_attempts':len(rows),'independent_accuracy_measured':False,
             'reconstruction_verification_pending':True,'physics_run':False,'training_eligible':False,
             'api_cost_usd':0,'compute_cost_usd':None}
    write_json(a.out/'summary.json',summary)
    if summary['status']!='validated':raise SystemExit(2)
    cached={**config,'proposal_cache':str(a.out),'producer_runtime_manifest':str(a.runtime_manifest)}
    write_json(a.out.parent/'sam3_config_cached.json',cached)


if __name__=='__main__':main()
