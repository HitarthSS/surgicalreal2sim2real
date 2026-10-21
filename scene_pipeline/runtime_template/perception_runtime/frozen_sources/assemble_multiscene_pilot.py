"""Strict, scene-specific handoff and attachment preflight for development pilot.

Rejected geometry never enters Newton. No missing tool fit is substituted.
Endpoint proximity is a proposed attachment, not proof of physical attachment.
"""
import argparse
import json
from pathlib import Path
import numpy as np

from newton_scene_adapter import assemble, sha256
from build_inspectable_example import export_obj
from needle_thread_connection_tool import attachment_state


def write(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','latest','mesh-helper'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--controllers',nargs='+',choices=('fixed','heuristic','agent'),required=True)
    a=p.parse_args();rows=[]
    for casepath in sorted((a.root/'cases').glob('*.json')):
        case=json.loads(casepath.read_text());cid=case['case_id']
        toolroot=a.latest/('hitarth_'+cid)
        tool=toolroot/'scene/combined_scene_online_tracker.obj'
        tracker=toolroot/'robot/online_tracker/tracker_result.npz'
        for controller in a.controllers:
            row={'case_id':cid,'controller':controller,'status':'not_assembled',
                 'missing_scene_specific_tool_fit':not (tool.is_file() and tracker.is_file()),
                 'independent_accuracy_measured':False,'training_eligible':False}
            outputs={t:a.root/'results'/cid/t/controller for t in ('thread','needle')}
            verdicts={t:json.loads((f/'pilot_result.json').read_text())
                      if (f/'pilot_result.json').exists() else None for t,f in outputs.items()}
            row['target_checks']={t:None if v is None else v['target_input_consistency_passed']
                                  for t,v in verdicts.items()}
            if not all(v and v['target_input_consistency_passed'] for v in verdicts.values()):
                row['reason']='Target reconstruction missing or rejected; do not import unchecked geometry.'
                rows.append(row);continue
            if row['missing_scene_specific_tool_fit']:
                row['reason']='Newest manual tool fit has not been completed for this scene.'
                rows.append(row);continue
            out=a.root/'assembly'/cid/controller;out.mkdir(parents=True,exist_ok=False)
            try:
                calls={t:sorted(outputs[t].glob('call_*/tool_result.json'))[-1] for t in outputs}
                inputs={'thread_samples':calls['thread'].parent/'backend/shared_thread_samples.npz',
                        'thread_result':calls['thread'],'needle_samples':calls['needle'].parent/'backend/shared_needle_arc.npz',
                        'needle_result':calls['needle'],'K':Path(case['inputs']['K']),
                        'tool_obj':tool,'tracker_result':tracker}
                contract={'scene_id':cid+'_'+controller+'_pilot','session_id':case['session_id'],
                          'split':'development','units':'m','frame':'rectified_left_camera',
                          'calibration_id':case['calibration_id'],'baseline_m':case['baseline_m'],
                          'inputs':{k:str(v) for k,v in inputs.items()},
                          'input_sha256':{k:sha256(v) for k,v in inputs.items()},
                          'needle':{'evidence':'shared_verified_free_radius_arc'},
                          'manual_interventions':case.get('manual_interventions','')+'; reuse same-case newest tool fit',
                          'tool_pose_evidence':'Fitted estimate, not independent ground truth'}
                g,report=assemble(contract,a.root)
                bundle=out/'scene_bundle.npz';np.savez_compressed(bundle,**g)
                write(out/'contract.json',contract);write(out/'handoff.json',report)
                export_obj(out/'scene_newton_m.obj',g,a.mesh_helper,.0005)
                state=attachment_state(g,bundle,{s:case['inputs'][s] for s in ('left','right')},case['baseline_m'])
                write(out/'attachment_preflight.json',state)
                eligible=[c for c in state['candidates'] if c['distance_m']<=state['max_connector_length_m']]
                row.update(status='assembled_not_physics_validated',bundle_sha256=sha256(bundle),
                    bundle=str(bundle),attachment_eligible_pair=eligible[0] if len(eligible)==1 and eligible[0]['endpoints_supported'] else None,
                    attachment_geometry_ambiguous=len(eligible)!=1,
                    closest_endpoint_gap_m=min(c['distance_m'] for c in state['candidates']),
                    complete_geometry_exported=True,physics_validated=False)
            except Exception as error:
                row.update(status='assembly_failed',error=f'{type(error).__name__}: {error}')
            write(out/'assembly_result.json',row);rows.append(row)
    write(a.root/('assembly_results_'+'_'.join(a.controllers)+'.json'),rows)
    for row in rows:print(json.dumps(row),flush=True)


if __name__=='__main__':main()
