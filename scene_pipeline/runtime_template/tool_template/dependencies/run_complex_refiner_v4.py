"""Run one small development checkpoint; cache provenance, keep every outcome."""
import argparse
import json
from pathlib import Path
import re

from complex_case_diagnostics import write_json
from complex_refiner_v4 import ComplexRefiner,Budget
from complex_refiner_tools_v4 import SharedComplexTools
from run_multiscene_pilot import QueueModel


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('case','pipeline-tools','evidence','out'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--original-src',type=Path)
    p.add_argument('--bundle',type=Path)
    p.add_argument('--controller',choices=('agent','heuristic'),required=True)
    p.add_argument('--queue',type=Path)
    p.add_argument('--sam3-config',type=Path)
    p.add_argument('--runtime-manifest',type=Path)
    p.add_argument('--decisions',type=int,default=4)
    a=p.parse_args()
    if a.controller=='agent' and a.queue is None:raise ValueError('Actual-model queue required')
    case=json.loads(a.case.read_text());evidence=json.loads(a.evidence.read_text())
    if case.get('split')=='test':raise ValueError('Development refinement cannot tune a held-out test case')
    images=[('image.left',case['inputs']['left']),('image.right',case['inputs']['right'])]
    sam3=json.loads(a.sam3_config.read_text()) if a.sam3_config else None
    tools=SharedComplexTools(a.case,a.pipeline_tools,a.original_src,a.bundle,sam3,a.controller)
    protected={**tools.paths,'evidence':a.evidence,'executor_source':Path(__file__),
               'budget_protocol_source':Path(__file__).with_name('complex_refiner_v4.py')}
    if a.runtime_manifest:protected['runtime_manifest']=a.runtime_manifest
    if a.sam3_config:protected['sam3_config']=a.sam3_config
    refiner=ComplexRefiner(case['case_id'],evidence,images,protected,a.out,tools.registry(),
                           Budget(a.decisions,4,1),a.controller)
    infer=QueueModel(a.queue) if a.queue else None
    if infer:
        existing=[int(m.group(1)) for p in a.queue.iterdir()
                  if (m:=re.fullmatch(r'request_(\d+)\.json',p.name))]
        infer.index=max(existing,default=0) # Preserve earlier requests and responses.
    summary=refiner.run(infer)
    print(json.dumps({k:summary[k] for k in ('case_id','controller','status','tool_calls','model_calls_completed','error')}))


if __name__=='__main__':main()
