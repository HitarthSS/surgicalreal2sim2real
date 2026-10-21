"""Connect actual Sol to unchanged shared visibility repair tools and guards."""
import argparse
import json
from pathlib import Path
from openai_scene_backend import OpenAISceneModel
from run_visibility_refiner_v5 import run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('case','pipeline-tools','out','key-file'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    case=json.loads(a.case.read_text())
    if case.get('split') not in ('development','train','validation'):
        raise ValueError('Development checkpoint cannot tune test cases')
    a.out.mkdir(parents=True,exist_ok=False)
    model=OpenAISceneModel(a.out/'model_requests',a.key_file,max_calls=4)
    result=run(a.case,a.pipeline_tools,a.out/'agent','agent',infer_backend=model,
               model_manifest=model.manifest)
    baseline=run(a.case,a.pipeline_tools,a.out/'heuristic','heuristic')
    summary=dict(agent={k:result[k] for k in ('status','decision_count','actual_model_used','estimated_api_cost_usd')},
        heuristic={k:baseline[k] for k in ('status','decision_count','actual_model_used')},
        shared_tools_and_constraints=True,independent_accuracy_measured=False,
        training_eligible=False)
    (a.out/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
