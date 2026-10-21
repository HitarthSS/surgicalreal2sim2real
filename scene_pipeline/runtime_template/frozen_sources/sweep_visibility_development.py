"""Finite development-only heuristic sweep; not a test-set optimizer.

No selection by input fit alone is called an accuracy improvement. This tool
records all failures and requires a human review before freezing parameters.
"""
import argparse
import itertools
import json
from pathlib import Path
from run_visibility_refiner_v5 import run
from reconstruction_tools import fingerprints


def prepare(cases):
    cases=list(map(Path,cases))
    if not cases:raise ValueError('Development cases required')
    for path in cases:
        case=json.loads(path.read_text())
        if case.get('split')!='development':
            raise ValueError('Sweep only accepts development cases; train/validation/test excluded')
        if not case.get('session_id'):raise ValueError('Recording/session grouping required')
    return cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=Path,nargs='+',required=True)
    p.add_argument('--pipeline-tools',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();cases=prepare(a.cases);a.out.mkdir(parents=True,exist_ok=False)
    grid=[dict(initial_tolerance_px=tol,connection_preference=curve)
          for tol,curve in itertools.product((2,4),('score','scale_0.65','scale_1.0'))]
    registry=dict(grid=grid,cases=[json.loads(c.read_text()) for c in cases],
        case_sha256=fingerprints({str(i):c for i,c in enumerate(cases)}),
        status='preregistered',selection_requires_review=True,
        protected_limits_tuned=False,independent_accuracy_measured=False)
    (a.out/'preregistration.json').write_text(json.dumps(registry,indent=2)+'\n')
    rows=[]
    for i,config in enumerate(grid):
        for path in cases:
            case=json.loads(path.read_text());folder=a.out/f'config_{i:02d}'/case['case_id']
            try:
                result=run(path,a.pipeline_tools,folder,'heuristic',heuristic_config=config)
                row=dict(config_id=i,config=config,case_id=case['case_id'],session_id=case['session_id'],
                    status=result['status'],decision_count=result['decision_count'],
                    runtime_seconds=result['runtime_seconds'],candidate=result['candidate'])
            except Exception as error:
                row=dict(config_id=i,config=config,case_id=case['case_id'],session_id=case['session_id'],
                         status='failed',error=f'{type(error).__name__}: {error}')
            rows.append(row)
            (a.out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    # Intentionally no automatic best-by-mask-fit winner or parameter freeze.
    (a.out/'review_required.json').write_text(json.dumps(dict(status='awaiting_pilot_review',
        rows=len(rows),recommended_review='Inspect correctness/deferrals and failure cases before comparing cost/call count; independent accuracy references remain required.'),indent=2)+'\n')
    print(json.dumps({'status':'awaiting_pilot_review','rows':len(rows)}))


if __name__=='__main__':main()
