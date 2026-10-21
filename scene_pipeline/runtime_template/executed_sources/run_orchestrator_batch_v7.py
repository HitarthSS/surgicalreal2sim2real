"""Rerun frozen coded controllers, preserving every requested case/failure.

The model/repair protocol is unchanged from the prior v7 checkpoint. This
driver only prepares provenance-bound inputs and dispatches isolated processes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from complex_case_diagnostics import write_json
from reconstruction_tools import fingerprints,sha256


def prepare(original,out,cases):
    for short in cases:
        cid='snapshot_20261015_'+short;source=original/cid;folder=out/cid
        folder.mkdir(exist_ok=False)
        shutil.copy2(source/'case.json',folder/'case.json')
        needle=source/'common_needle/call_001/backend/shared_needle_arc.npz'
        shutil.copy2(needle,folder/'common_needle.npz')
        case=json.loads((folder/'case.json').read_text())
        if case['scene_priors'].get('single_thread') is not True:raise ValueError('User-declared single-thread prior absent')
        write_json(folder/'preprocessing_cache.json',dict(source_root=str(source),
            source_case_sha256=sha256(source/'case.json'),input_fingerprints=fingerprints(case['inputs']),
            needle_sha256=sha256(needle),
            reuse_reason='Same complete stereo images, masks, depth, calibration and common needle; controllers only rerun.',
            single_thread_prior='User confirmed all six physical scenes contain one thread.',split='development'))


def run_one(out,original,cache,short,controller,key_file):
    cid='snapshot_20261015_'+short;folder=out/cid;destination=folder/controller
    provenance=json.loads((folder/'preprocessing_cache.json').read_text())
    if fingerprints(json.loads((folder/'case.json').read_text())['inputs'])!=provenance['input_fingerprints']:
        raise ValueError('Cached preprocessing changed')
    command=[sys.executable,str(out/'executed_sources/run_scene_orchestrator_v7.py'),
        '--case',str(folder/'case.json'),'--pipeline',str(original/'pipeline-tools'),
        '--needle',str(folder/'common_needle.npz'),'--out',str(destination),
        '--controller',controller,'--cache',str(cache),'--max-model-calls','12','--max-tool-calls','10']
    if controller=='agent':command+=['--key-file',str(key_file)]
    environment=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    began=time.perf_counter()
    with (folder/(controller+'_stdout.log')).open('w') as stdout,(folder/(controller+'_stderr.log')).open('w') as stderr:
        process=subprocess.run(command,stdout=stdout,stderr=stderr,env=environment,check=False)
    row=dict(case_id=cid,controller=controller,exit_code=process.returncode,wall_seconds=time.perf_counter()-began,
        summary_path=str(destination/'summary.json'))
    if (destination/'summary.json').is_file():
        summary=json.loads((destination/'summary.json').read_text());candidate=summary.get('candidate');report=(candidate or {}).get('report',{})
        ledger=json.loads(Path(summary['endpoint_ledger_path']).read_text())
        history=json.loads(Path(summary['history_path']).read_text());attachment=summary.get('attachment') or {}
        accounting=summary.get('model_accounting') or {}
        row.update(status=summary['status'],geometry_stage=(candidate or {}).get('stage'),
            fragment_count=report.get('fragment_count'),numeric_checks_passed=report.get('numerical_checks_passed',False),
            geometry_failures=report.get('failures',[]),all_current_endpoints_assessed=summary['completion']['all_current_endpoints_assessed'],
            pending_endpoints=sum(x['status']=='pending' for x in ledger.values()),budget=summary['budget'],
            api_calls_completed=accounting.get('completed',0),actual_model_used=accounting.get('actual_model_used',False),
            estimated_api_cost_usd=accounting.get('estimated_api_cost_usd',0.),
            attachment_status=attachment.get('status'),attachment_source_gap_m=(attachment.get('metadata') or {}).get('source_endpoint_gap_m'),
            repair_operations=sum('tool' in event for event in history),
            errors=[{k:event[k] for k in ('role','status','error') if k in event} for event in history if event.get('status') in ('failed','invalid_plan','final_construction_failed')])
    else:
        row.update(status='process_failed_without_summary',stderr_path=str(folder/(controller+'_stderr.log')))
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--controller',choices=['agent','heuristic'],required=True)
    p.add_argument('--key-file',type=Path);p.add_argument('--workers',type=int,default=2)
    args=p.parse_args();out=args.out.resolve();original=args.original.resolve();cache=args.cache.resolve()
    reg=json.loads((out/'preregistration.json').read_text());cases=reg['cases']
    if not 1<=args.workers<=2:raise ValueError('At most two concurrent case processes')
    if args.controller=='heuristic':prepare(original,out,cases)
    elif any(not (out/('snapshot_20261015_'+s)/'case.json').exists() for s in cases):raise ValueError('Prepare heuristic/input run first')
    executed=out/'executed_sources';sources=fingerprints({p.name:p for p in executed.glob('*.py')})
    rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        tasks={executor.submit(run_one,out,original,cache,s,args.controller,args.key_file):s for s in cases}
        for task in as_completed(tasks):
            short=tasks[task]
            try:row=task.result()
            except Exception as error:row=dict(case_id='snapshot_20261015_'+short,controller=args.controller,status='driver_failed',error=str(error))
            rows.append(row);rows.sort(key=lambda r:cases.index(r['case_id'].split('_')[-1]))
            write_json(out/(args.controller+'_results.json'),rows)
            print(json.dumps(row),flush=True)
    if fingerprints({p.name:p for p in executed.glob('*.py')})!=sources:raise ValueError('Frozen runtime source changed')
    write_json(out/(args.controller+'_complete.json'),dict(cases=cases,count=len(rows),
        completed_utc=datetime.now(timezone.utc).isoformat(),source_fingerprints_verified=True))


if __name__=='__main__':main()
