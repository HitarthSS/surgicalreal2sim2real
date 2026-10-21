"""Frozen six-case development rerun; cached preprocessing is hash-bound."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time
from complex_case_diagnostics import write_json
from openai_scene_backend import OpenAISceneModel
from reconstruction_tools import fingerprints,sha256
from run_shared_refiner_v6 import run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--controller',choices=['heuristic','agent'],required=True)
    p.add_argument('--key-file',type=Path)
    a=p.parse_args();a.out=a.out.resolve();a.original=a.original.resolve()
    registry=json.loads((a.out/'preregistration.json').read_text())
    pipeline=a.original/'pipeline-tools';results_path=a.out/'batch_results.json'
    results=json.loads(results_path.read_text()) if results_path.exists() else []
    by_id={r['case_id']:r for r in results}
    # The old masks are a labeled manual option, and only for an exact match.
    manual=a.original.parent/'case5_newton_2026-10-12/inputs'
    registered=dict(masks={s:str(manual/(s+'_occluder.png')) for s in ['left','right']},
        image_sha256={s:sha256(manual/(s+'.png')) for s in ['left','right']},
        source='earlier assistant-traced visible instrument polygons; approximate development annotation')
    executed=Path(__file__).resolve().parent
    manifest=fingerprints({p.name:p for p in executed.glob('*.py')})
    write_json(a.out/('sources_'+a.controller+'.json'),manifest)
    for sid in registry['cases']:
        cid='snapshot_20261009_'+sid;old=a.original/cid;folder=a.out/cid
        if not folder.exists():
            folder.mkdir();shutil.copy2(old/'case.json',folder/'case.json')
            shutil.copy2(old/'needle_result.json',folder/'needle_result.json')
            shutil.copytree(old/'common_needle',folder/'common_needle')
            write_json(folder/'preprocessing_cache.json',dict(source_root=str(old),
                original_case_sha256=sha256(old/'case.json'),
                input_fingerprints=fingerprints(json.loads((old/'case.json').read_text())['inputs']),
                needle_result_sha256=sha256(old/'needle_result.json'),
                reuse_reason='Same original stereo, masks, depth, calibration and common needle; only controller changed.'))
        cache=json.loads((folder/'preprocessing_cache.json').read_text())
        if cache['original_case_sha256']!=sha256(old/'case.json') or cache['input_fingerprints']!=fingerprints(json.loads((folder/'case.json').read_text())['inputs']):
            raise ValueError('Preprocessing cache provenance mismatch')
        controllers=folder/'controllers';controllers.mkdir(exist_ok=True)
        model=OpenAISceneModel(controllers/'model_requests',a.key_file,max_calls=6) if a.controller=='agent' else None
        reg=registered if registered['image_sha256']=={s:sha256(json.loads((folder/'case.json').read_text())['inputs'][s]) for s in ['left','right']} else None
        began=time.perf_counter()
        summary=run(folder/'case.json',pipeline,controllers/a.controller,a.controller,model,
            a.out/'mask_cache',reg,max_decisions=6)
        row=by_id.setdefault(cid,dict(case_id=cid,split='development',single_thread_prior=True,
            needle_accepted=json.loads((folder/'needle_result.json').read_text())['accepted']))
        row[a.controller]={k:summary[k] for k in ['status','decision_count','actual_model_used','model_calls_attempted',
            'model_calls_completed','runtime_seconds','estimated_api_cost_usd','baseline_geometry_preserved',
            'terminal_review','manual_occluder_used']}
        row[a.controller]['actions']=[h['decision']['action_id'] for h in summary['history'] if 'decision' in h]
        row[a.controller]['failures']=(summary.get('candidate') or {}).get('report',{}).get('failures',[])
        row[a.controller]['initial_status']=summary['initial_reconstruction']['status']
        rows=[by_id['snapshot_20261009_'+s] for s in registry['cases'] if 'snapshot_20261009_'+s in by_id]
        write_json(results_path,rows)
        print(json.dumps(dict(case_id=cid,controller=a.controller,seconds=time.perf_counter()-began,result=row[a.controller])),flush=True)
    if fingerprints({p.name:p for p in executed.glob('*.py')})!=manifest:raise ValueError('Executed sources changed')
    write_json(a.out/(a.controller+'_complete.json'),dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        cases=registry['cases'],source_fingerprints_verified=True))


if __name__=='__main__':main()
