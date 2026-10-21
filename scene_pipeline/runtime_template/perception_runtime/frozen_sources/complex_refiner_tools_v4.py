"""Concrete shared tools for the v4 loop. Each result is a reversible proposal."""
import json
import hashlib
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from complex_case_diagnostics import (inspect_thread_masks,inspect_attachment,
    propose_stereo_thread_mask,mask_revision_audit,write_json)
from reconstruction_tools import ReconstructionSession,fingerprints,sha256
from sam3_proposal_cache_v4 import cache_key,load_proposal


def mask_content_id(masks):
    """Identify binary pixels, independent of PNG compression or metadata."""
    h=hashlib.sha256()
    for side in ('left','right'):
        pixels=cv2.imread(str(masks[side]),cv2.IMREAD_GRAYSCALE)
        if pixels is None:raise ValueError('Unreadable '+side+' mask')
        binary=np.ascontiguousarray(pixels>0,dtype=np.uint8)
        h.update(json.dumps({'side':side,'shape':binary.shape},sort_keys=True).encode())
        h.update(binary.tobytes())
    return h.hexdigest()


class SharedComplexTools:
    def __init__(self,case_path,pipeline_tools,original_src=None,bundle=None,sam3=None,controller='agent'):
        self.case_path=Path(case_path);self.case=json.loads(self.case_path.read_text());self.inputs=self.case['inputs']
        self.pipeline=Path(pipeline_tools);self.original=original_src;self.bundle=bundle
        self.sam3=sam3;self.controller=controller
        self.paths={**self.inputs,'case':self.case_path,'toolkit_source':__file__,
                    'diagnostic_source':Path(__file__).with_name('complex_case_diagnostics.py'),
                    'shared_interface':Path(__file__).with_name('reconstruction_tools.py'),
                    'shared_verifier':Path(__file__).with_name('reconstruction_verification.py')}
        for p in self.pipeline.glob('*.py'):self.paths['pipeline/'+p.name]=p
        if original_src:
            for p in Path(original_src).rglob('*.py'):self.paths['original/'+str(p.relative_to(original_src))]=p
        if bundle:self.paths['scene_bundle']=str(bundle)
        if sam3:
            self.paths['sam3_checkpoint']=str(sam3['checkpoint'])
            self.paths['sam3_runner']=Path(__file__).with_name('refine_thread_sam3_v4.py')
            if sha256(sam3['checkpoint'])!=sam3['sha256']:raise ValueError('Unexpected SAM3 checkpoint')
            if sam3.get('proposal_cache'):
                self.paths['sam3_cache_source']=Path(__file__).with_name('sam3_proposal_cache_v4.py')
                self.paths['sam3_producer_runtime']=sam3['producer_runtime_manifest']
                for p in Path(sam3['proposal_cache']).rglob('*'):
                    if p.is_file():self.paths['sam3_cache/'+str(p.relative_to(sam3['proposal_cache']))]=str(p)
        self.expected=fingerprints(self.paths)

    def assert_unchanged(self):
        if fingerprints(self.paths)!=self.expected:raise ValueError('Shared tool inputs/sources changed')

    def registry(self):
        result={'inspect_fragments':self.inspect_fragments,'stereo_mask':self.stereo_mask,
                'reconstruct_thread':self.reconstruct_thread}
        if self.bundle:result['inspect_attachment']=self.inspect_attachment
        if self.sam3:result['resegment_thread']=self.resegment_thread
        return result

    def inspect_fragments(self,parameters,candidate,out):
        self.assert_unchanged()
        if parameters:raise ValueError('Inspection has no adjustable parameters')
        revised=bool(candidate.get('mask_revision'))
        inputs=json.loads(Path(candidate['case_path']).read_text())['inputs'] if revised else self.inputs
        result=inspect_thread_masks(inputs,self.pipeline,out)
        return {'status':'observed','inspection':{'candidate_mask' if revised else 'mask':result},'geometry_modified':False}

    def inspect_attachment(self,parameters,candidate,out):
        self.assert_unchanged()
        if parameters:raise ValueError('Inspection has no adjustable parameters')
        result=inspect_attachment(self.bundle,self.inputs,self.case['baseline_m'],out)
        return {'status':'observed','inspection':{'attachment':result},'geometry_modified':False}

    def candidate(self,masks,previous,report,out):
        content=mask_content_id(masks)
        if previous.get('mask_content_id')==content:
            # Preserve the already verified state and its provenance. A
            # different parameter card is not a different scene hypothesis
            # when it produces exactly the same pixels in both views.
            return json.loads(json.dumps(previous))
        case=json.loads(json.dumps(self.case));case['inputs'].update({
            'thread_left_mask':masks['left'],'thread_right_mask':masks['right']})
        path=Path(out)/'derived_case.json';write_json(path,case)
        protected={'case':str(path.resolve()),'left_mask':masks['left'],'right_mask':masks['right']}
        return {'revision':previous.get('revision',0)+1,'mask_revision':True,'mask_content_id':content,
                'case_path':str(path.resolve()),'parameters':report.get('parameters'),
                'mask_audit':report['audit'],'needs_visual_review':True,
                'protected_candidate_paths':protected,'candidate_fingerprints':fingerprints(protected),
                'independent_accuracy_measured':False,'training_eligible':False}

    def stereo_mask(self,parameters,candidate,out):
        self.assert_unchanged()
        result=propose_stereo_thread_mask(self.inputs,self.case['baseline_m'],parameters,self.pipeline,out)
        proposed=self.candidate(result['proposed_masks'],candidate,result,out)
        duplicate=bool(candidate.get('mask_content_id')==proposed['mask_content_id'])
        inputs=json.loads(Path(proposed['case_path']).read_text())['inputs']
        inspection=inspect_thread_masks(inputs,self.pipeline,Path(out)/'candidate_inspection')
        return {'status':'duplicate_proposal_no_state_change' if duplicate else 'proposal_only',
                'candidate':proposed,'inspection':{'candidate_mask':inspection},
                'audit':result['audit'],'duplicate_mask_content':duplicate,'geometry_modified':False}

    def resegment_thread(self,parameters,candidate,out):
        self.assert_unchanged()
        if set(parameters)!= {'prompt','confidence_threshold'} or parameters['prompt'] not in ('suture thread','thin suture thread'):
            raise ValueError('Invalid segmentation request')
        threshold=parameters['confidence_threshold']
        if isinstance(threshold,bool) or threshold not in (.20,.35):raise ValueError('Invalid segmentation threshold')
        out=Path(out);out.mkdir(parents=True,exist_ok=False);output=out/'backend'
        cached=bool(self.sam3.get('proposal_cache'))
        if cached:
            key=cache_key(self.inputs,self.sam3['sha256'],Path(__file__).with_name('refine_thread_sam3_v4.py'),
                          self.pipeline/'segment_needle_thread_with_sam3.py',
                          self.sam3['producer_runtime_manifest'],parameters)
            masks,report,entry=load_proposal(self.sam3['proposal_cache'],key)
            write_json(out/'cache_reuse.json',{'cache_hit':True,'exact_provenance':entry,
                'logical_tool_attempt':True,'new_model_inference':False,
                'producer_cost_accounted_as_common_preprocessing':True})
            write_json(out/'backend_report.json',report)
        else:
            masks,report=self.execute_sam3(parameters,out,output)
        audit=mask_revision_audit(self.inputs,masks,self.pipeline)
        result={'parameters':parameters,'audit':audit};proposed=self.candidate(masks,candidate,result,out)
        duplicate=bool(candidate.get('mask_content_id')==proposed['mask_content_id'])
        inspection_inputs=json.loads(Path(proposed['case_path']).read_text())['inputs']
        inspection=inspect_thread_masks(inspection_inputs,self.pipeline,out/'candidate_inspection')
        self.assert_unchanged()
        return {'status':'duplicate_proposal_no_state_change' if duplicate else 'proposal_only',
                'candidate':proposed,'inspection':{'candidate_mask':inspection},'audit':audit,
                'duplicate_mask_content':duplicate,'cache_hit':cached,'logical_tool_attempt':True,
                'new_model_inference':not cached,
                'backend_report':report,'geometry_modified':False}

    def execute_sam3(self,parameters,out,output):
        """One actual-model subprocess; caller logs/audits the resulting proposal."""
        threshold=parameters['confidence_threshold']
        command=[sys.executable,str(Path(__file__).with_name('refine_thread_sam3_v4.py'))]
        for name,value in {'left':self.inputs['left'],'right':self.inputs['right'],'pipeline-tools':self.pipeline,
                           'checkpoint':self.sam3['checkpoint'],'expected-checkpoint-sha256':self.sam3['sha256'],
                           'out':output,'prompt':parameters['prompt'],'confidence-threshold':threshold,
                           'device':self.sam3.get('device','cpu')}.items():command+=['--'+name,str(value)]
        write_json(out/'command.json',{'argv':command,'parameters':parameters})
        with (out/'stdout.txt').open('w') as stdout,(out/'stderr.txt').open('w') as stderr:
            completed=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=600,check=False)
        report=json.loads((output/'report.json').read_text()) if (output/'report.json').exists() else {}
        if completed.returncode or report.get('status')!='proposal_generated':
            raise RuntimeError(report.get('error','SAM3 backend failed; inspect stderr'))
        masks={s:str((output/(s+'_thread_mask.png')).resolve()) for s in ('left','right')}
        return masks,report

    def reconstruct_thread(self,parameters,candidate,out):
        self.assert_unchanged()
        if not candidate.get('mask_revision'):raise ValueError('No proposed masks to verify')
        if set(parameters)!={'method','parameters'} or parameters['method'] not in ('thread_qp','thread_occlusion'):
            raise ValueError('Invalid reconstruction request')
        session=ReconstructionSession(candidate['case_path'],out,self.controller,'complex_refiner_v4_1',
                                      self.pipeline,max_calls=1,original_src=self.original)
        result=session.call({'tool':parameters['method'],'parameters':parameters['parameters'],
                             'reason':'Verify the logged mask revision with the unchanged shared reconstruction tools and checks.'})
        proposed={**candidate,'last_reconstruction':{'result_path':str((Path(out)/result['call_id']/'tool_result.json').resolve()),
                  'accepted':result['accepted'],'verification':result.get('verification')}}
        protected=dict(candidate['protected_candidate_paths'])
        for p in (Path(out)/result['call_id']).rglob('*'):
            if p.is_file():protected['reconstruction/'+str(p.relative_to(out))]=str(p.resolve())
        proposed['protected_candidate_paths']=protected
        proposed['candidate_fingerprints']=fingerprints(protected)
        self.assert_unchanged()
        stderr=Path(out)/result['call_id']/'backend_stderr.txt'
        text=stderr.read_text() if stderr.exists() else ''
        infrastructure=next((line.strip() for line in text.splitlines()
                             if line.startswith(('ModuleNotFoundError:','ImportError:'))),None)
        return {'status':result['status'],'candidate':proposed,'shared_tool_result':result,
                'candidate_consistency_passed':result['accepted'],'mask_revision_needs_review':True,
                'infrastructure_failure':infrastructure,
                'independent_accuracy_measured':False,'physics_validated':False,'training_eligible':False}
