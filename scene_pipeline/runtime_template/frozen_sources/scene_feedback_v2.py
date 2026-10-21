"""Live, artifact-bound assembly and attachment preflight for the refiner.

Reuses the existing Newton adapter and endpoint test. It cannot change geometry,
apply joints, replace missing tool fits or certify accuracy/physics/training.
"""
import json
from pathlib import Path
import numpy as np

from newton_scene_adapter import assemble,sha256
from needle_thread_connection_tool import attachment_state
from reconstruction_tools import fingerprints


class SceneFeedback:
    def __init__(self,session,needle_result_path,latest,out):
        self.session=session;self.needle_result_path=Path(needle_result_path)
        self.latest=Path(latest);self.out=Path(out);self.cache={};self.live_contexts=[]
        root=self.latest/('hitarth_'+session.case['case_id'])
        self.tool=root/'scene/combined_scene_online_tracker.obj'
        self.tracker=root/'robot/online_tracker/tracker_result.npz'
        event=self.needle_result_path.parent.parent/'events.jsonl'
        first=json.loads(event.read_text().splitlines()[0])
        if first['case_id']!=session.case['case_id'] or first['input_fingerprints']!=session.input_fingerprints:
            raise ValueError('Needle context does not share exact current case inputs')
        paths={'needle_result':self.needle_result_path,
               'needle_samples':self.needle_result_path.parent/'backend/shared_needle_arc.npz',
               'needle_events':event,'case':session.case_path,'adapter_source':Path(__file__).with_name('newton_scene_adapter.py'),
               'attachment_source':Path(__file__).with_name('needle_thread_connection_tool.py'),
               'feedback_source':Path(__file__)}
        if self.tool.exists():paths['tool_obj']=self.tool
        if self.tracker.exists():paths['tracker_result']=self.tracker
        # A failed needle context may not have a samples file. Keep its rejection.
        paths={k:p for k,p in paths.items() if p.exists()}
        self.paths=paths;self.expected=fingerprints(paths)

    def assert_unchanged(self):
        if any(not p.is_file() for p in self.paths.values()):
            raise ValueError('Scene-feedback context file removed or changed during inference')
        if fingerprints(self.paths)!=self.expected:
            raise ValueError('Scene-feedback context changed during inference')
        if ('tool_obj' in self.paths)!=self.tool.exists() or ('tracker_result' in self.paths)!=self.tracker.exists():
            raise ValueError('Scene-specific tool availability changed during inference')
        for paths,expected in self.live_contexts:
            if fingerprints(paths)!=expected:
                raise ValueError('Current assembly/attachment artifacts changed during inference')

    def __call__(self,result):
        self.assert_unchanged()
        verdict=json.loads(self.needle_result_path.read_text())
        missing=[]
        if not self.tool.is_file() or not self.tracker.is_file():missing.append('scene_specific_newest_tool_fit')
        if not (verdict.get('accepted') and verdict.get('verification',{}).get('accepted')):
            missing.append('accepted_needle_reconstruction')
        if not (result and result.get('accepted') and result.get('verification',{}).get('accepted')):
            missing.append('accepted_thread_reconstruction')
        feedback={'schema':'scene_preflight_v2','blocking_for_finish':True,'preflight_passed':False,
                  'assembly':{'status':'blocked','missing_prerequisites':missing},
                  'attachment':{'status':'not_assessed','max_connector_length_m':.004},
                  'physics_validated':False,'training_eligible':False,'independent_accuracy_measured':False,
                  '_provenance':{'fixed_context':self.expected}}
        if missing:return feedback
        current=self.session.out/result['call_id']/'tool_result.json'
        if json.loads(current.read_text())!=result:
            raise ValueError('Feedback result is not the current recorded tool result')
        key=sha256(current)
        if key in self.cache:return self.cache[key]
        folder=self.out/result['call_id'];folder.mkdir(parents=True,exist_ok=False)
        inputs={'thread_result':current,'thread_samples':current.parent/'backend/shared_thread_samples.npz',
                'needle_result':self.needle_result_path,'needle_samples':self.needle_result_path.parent/'backend/shared_needle_arc.npz',
                'K':Path(self.session.inputs['K']),'tool_obj':self.tool,'tracker_result':self.tracker}
        contract={'scene_id':self.session.case['case_id']+'_live_preflight',
                  'session_id':self.session.case['session_id'],'split':'development','units':'m',
                  'frame':'rectified_left_camera','calibration_id':self.session.case['calibration_id'],
                  'baseline_m':self.session.case['baseline_m'],'inputs':{k:str(v) for k,v in inputs.items()},
                  'input_sha256':{k:sha256(v) for k,v in inputs.items()},
                  'needle':{'evidence':'shared_verified_free_radius_arc'},
                  'manual_interventions':'Reuse same-case newest saved manual tool fit; no new selections',
                  'tool_pose_evidence':'Fitted estimate, not independent reference'}
        geometry,report=assemble(contract,self.out)
        bundle=folder/'scene_bundle.npz';np.savez_compressed(bundle,**geometry)
        state=attachment_state(geometry,bundle,{s:self.session.inputs[s] for s in ('left','right')},self.session.case['baseline_m'])
        eligible=[c for c in state['candidates'] if c['distance_m']<=state['max_connector_length_m']]
        valid=len(eligible)==1 and eligible[0]['endpoints_supported']
        closest=min(c['distance_m'] for c in state['candidates'])
        failures=[]
        if closest>state['max_connector_length_m']:failures.append('endpoint_gap_exceeds_4mm')
        elif len(eligible)!=1:failures.append('ambiguous_endpoint_pairs')
        elif not eligible[0]['endpoints_supported']:failures.append('attachment_endpoint_is_inferred')
        feedback.update(blocking_for_finish=not valid,preflight_passed=valid,
            assembly={'status':'geometry_assembled_not_physics_validated','missing_prerequisites':[],
                      'source_geometry_preserved':True,'input_consistency_only':True},
            attachment={'status':'eligible_short_gap_not_connected' if valid else 'rejected',
                        'closest_endpoint_gap_m':closest,'max_connector_length_m':.004,
                        'failures':failures,'candidates':state['candidates'],
                        'eligible_candidate_id':eligible[0]['id'] if valid else None,
                        'created':False,'physical_attachment_verified':False},
            _provenance={'fixed_context':self.expected,'current_contract':contract,
                         'bundle_sha256':sha256(bundle),'attachment_state':state})
        for name,value in [('contract.json',contract),('handoff.json',report),
                           ('attachment_preflight.json',state),('feedback.json',feedback)]:
            (folder/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
        self.cache[key]=feedback
        live=dict(inputs,bundle=bundle)
        for name in ('contract.json','handoff.json','attachment_preflight.json','feedback.json'):
            live[name]=folder/name
        self.live_contexts.append((live,fingerprints(live)))
        return feedback
