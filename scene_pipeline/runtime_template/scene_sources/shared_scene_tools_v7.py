"""Common reconstruction/assembly operations for hierarchical controllers."""
from pathlib import Path
import cv2
import numpy as np
from compact_agent_protocol import digest
from complex_case_diagnostics import write_json
from run_shared_refiner_v6 import SharedRepairToolsV6
from thread_pair_assembly_v7 import connect_fragment_pair
from occluder_polygon_tool_v7 import propose_occluder_polygons


class SharedSceneToolsV7(SharedRepairToolsV6):
    def __init__(self,*args,needle_path=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.needle_path=Path(needle_path).resolve() if needle_path else None
        self.geometry_version=None
        for name in ['shared_scene_tools_v7.py','thread_pair_assembly_v7.py','occluder_polygon_tool_v7.py',
                     'thread_gap_inventory_v7.py','needle_attachment_inventory_v7.py','run_scene_orchestrator_v7.py','thread_object_evidence_v8.py','evidence_transition.py']:
            self.paths['v7/'+name]=Path(__file__).with_name(name)
        if self.needle_path:self.paths['unchanged_needle']=self.needle_path
        from reconstruction_tools import fingerprints
        self.expected=fingerprints(self.paths)

    def audit_geometry(self,proposal):
        from thread_object_evidence_v8 import evaluate_thread_object_evidence
        from reconstruction_tools import fingerprints
        audit=evaluate_thread_object_evidence(self.original,self.pipeline,proposal['fragment_path'],self.needle_path,identity_receipt=self.case['inputs'].get('object_identity_receipt'))
        report=proposal['report'];report['object_evidence_audit']=audit
        report['original_visible_mask_coverage']={s:v['raw_original_mask_coverage'] for s,v in audit['views'].items()}
        report['object_aware_original_mask_coverage']={s:v['object_aware_thread_coverage'] for s,v in audit['views'].items()}
        report['mask_revision_audit']=self.last_audit
        failures=set(report['failures'])-{'original_visible_support_lost','no_unambiguous_thread_evidence'}
        failures.update(audit['failures']);report['failures']=sorted(failures);report['numerical_checks_passed']=not failures
        report['warnings']=sorted(set(report.get('warnings',[])+audit['warnings']))
        proposal['stage']='connected' if not failures and report['fragment_count']==1 else 'fragments'
        write_json(proposal['protected_paths']['report'],report);proposal['fingerprints']=fingerprints(proposal['protected_paths'])
        return proposal

    def action_key(self,card):
        key=super().action_key(card)
        if card['operation']=='occluder_polygons':
            from reconstruction_tools import sha256
            evidence={k:sha256(v) for k,v in self.case['inputs'].items() if k.endswith('_mask') or k.endswith('_occluder')}
            return digest([key,evidence,self.geometry_version,card.get('repair_opportunity',0)])
        # Fragment indices can be reused after a merge. The same proposal ID
        # now addresses different geometry and must not be mistaken for a retry.
        if card['operation']=='inferred_connection':
            from reconstruction_tools import sha256
            evidence={k:sha256(v) for k,v in self.case['inputs'].items() if k.endswith('_mask') or k.endswith('_occluder')}
            return digest([key,self.geometry_version,evidence])
        return digest([key,self.geometry_version]) if card['operation']=='smooth_connection' else key

    def menu(self,candidate,attempted):
        actions,rows=super().menu(candidate,attempted)
        if candidate and candidate['stage']=='connected':return actions,rows
        if candidate and candidate['report']['numerical_checks_passed']:
            from thread_visibility_repairs_v5 import join_proposals,load_parts,in_view,project,load_inputs
            parts=load_parts(candidate['fragment_path'])
            proposals=join_proposals(self.case,self.pipeline,parts)
            _,_,masks,_,_,K,baseline=load_inputs(self.case,self.pipeline)
            boundary=[]
            for row in proposals:
                xy0=project(row['_curve'],K);xy1=project(row['_curve'],K,baseline)
                outside=((~in_view(xy0,masks['left'].shape)) & (~in_view(xy1,masks['right'].shape)))[5:-5]
                failures=set(row['failures']) | ({'unobserved_out_of_frame_connection'} if outside.any() else set())
                row['_bounded_failures']=failures
                if failures=={'unobserved_out_of_frame_connection'}:boundary.append(row)
            boundary_pairs={tuple(sorted(row['endpoints'])) for row in boundary}
            boundary_ids={row['proposal_id'] for row in boundary}
            for row in proposals:
                mode=None
                if len(parts)==2 and len(boundary_pairs)==1 and row['proposal_id'] in boundary_ids:mode='boundary'
                elif not row['_bounded_failures']:
                    competitors=[p for p in proposals if not p['_bounded_failures'] and set(p['endpoints'])!=set(row['endpoints']) and set(p['endpoints'])&set(row['endpoints'])]
                    if competitors and min(p['score'] for p in competitors)-row['score']<.1:mode='topology'
                if mode:
                    card=dict(operation='inferred_connection',parameters=dict(proposal_id=row['proposal_id'],inference_mode=mode),
                        effect='Propose a bounded '+mode+' connection from known endpoints; hidden curve/topology remains an explicit hypothesis. All physical and visible-background checks still apply.')
                    if self.action_key(card) not in attempted:actions['hypothesis.'+mode+'.'+row['proposal_id']]=card
        # A checked partial reconstruction can still lack a continuation that
        # the other camera sees. Do not hide view-switching tools in that state.
        for reference,condition in self.reference_preconditions.items():
            if not condition['eligible']:continue
            for tolerance in [2.,4.]:
                card=dict(operation='cross_view_recovery',parameters=dict(reference=reference,stereo_match_px=tolerance),
                    effect='Reobserve the complete ordered paths in the other view; preserve observed geometry and label unsupported depth.')
                if self.action_key(card) not in attempted:actions[f'v{self.version}.cross.{reference}.{int(tolerance)}px']=card
        card=dict(operation='occluder_polygons',parameters={},parameter_schema={
            'polygons':{'left':'0–2 visible-tool polygons; each 3–16 [x,y] vertices','right':'same'},
            'coordinates':'original full-frame pixels; inside image; tool must enter image boundary'},
            effect='Propose approximate visible tool masks from both images; exclude observed thread/needle; inspect feedback and rerun gap checks.')
        for opportunity in range(2):
            card['repair_opportunity']=opportunity
            if self.action_key(card) not in attempted:
                actions['occluder.polygons']=dict(card)
                break
        return actions,rows

    def execute_repair(self,aid,card,candidate,out,selection_evidence=None):
        self.assert_unchanged(candidate)
        if card['operation'] in ('smooth_connection','inferred_connection'):
            if not candidate or not candidate['report']['numerical_checks_passed']:raise ValueError('Checked visible fragments required')
            report=connect_fragment_pair(self.case,self.pipeline,candidate['fragment_path'],card['parameters']['proposal_id'],out,
                inference_mode=card['parameters'].get('inference_mode'),selection_evidence=selection_evidence)
            lineage=list(candidate['report'].get('hypothesis_lineage',[]))
            if report['numerical_checks_passed'] and report.get('inference_mode'):
                lineage.append(dict(mode=report['inference_mode'],proposal_id=card['parameters']['proposal_id'],
                    uncertainty_warnings=report['uncertainty_warnings'],selection_evidence=selection_evidence))
            report['hypothesis_lineage']=lineage
            report['connected_hypothesis_only']=bool(lineage)
            report['uncertainty_warnings']=sorted(set(report['uncertainty_warnings']+[
                warning for prior in lineage for warning in prior['uncertainty_warnings']]))
            proposal=self.audit_geometry(self.package(Path(out)/'fragments.npz',report,out))
            return proposal,dict(pair_assembly=True,unselected_fragments_preserved=report['unselected_fragments_preserved'])
        if card['operation']=='occluder_polygons':
            report=propose_occluder_polygons(self.case,card['parameters'],out,candidate)
            from thread_visibility_repairs_v5 import load_inputs
            _,_,_,old_masks,*_=load_inputs(self.case,self.pipeline)
            same=all(np.array_equal(cv2.imread(path,0)>0,old_masks[side]) for side,path in report['masks'].items())
            if same:
                report['no_op']=True;write_json(Path(out)/'report.json',report)
                return candidate,report
            self.case['inputs'].update({s+'_occluder':p for s,p in report['masks'].items()})
            self.version+=1;self.working_path=Path(out)/'derived_case.json';write_json(self.working_path,self.case)
            self.register_derived({str(p.resolve()):str(p.resolve()) for p in Path(out).glob('*') if p.is_file()})
            return candidate,report
        return super().execute_repair(aid,card,candidate,out)


def heuristic_polygons(tools):
    """Same polygon API; fixed heuristic obtains outlines from cached perception.

    No fabricated mask is substituted when SAM3 is empty. Earlier manual masks
    remain their separate, explicitly manual operation.
    """
    polygons={}
    masks,_=tools.read_cache('occluder.sam3.grasper')
    for side,path in masks.items():
        mask=cv2.imread(path,0);contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        values=[]
        for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:2]:
            epsilon=1.
            points=cv2.approxPolyDP(contour,epsilon,True)
            while len(points)>16:epsilon+=1.;points=cv2.approxPolyDP(contour,epsilon,True)
            if len(points)>=3:values.append(points.reshape(-1,2).tolist())
        polygons[side]=values
    return {'polygons':polygons}
