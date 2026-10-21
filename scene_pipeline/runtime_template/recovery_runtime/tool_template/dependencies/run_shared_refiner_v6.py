"""Reconstruct first, then use matched bounded tools to repair explicit failures.

The original controller remains available. All revisions are separate artifacts;
defer never discards a checked initial reconstruction. No needle changes,
physical-limit tuning, attachment, policy or simulation approval is performed.
"""
from collections import Counter
import json
from pathlib import Path
import time
import cv2
import numpy as np
from compact_agent_protocol import digest,encoded
from complex_case_diagnostics import inspect_thread_masks,mask_revision_audit,propose_stereo_thread_mask,write_json
from complex_refiner_tools_v4 import mask_content_id
from complex_refiner_v4 import validate_decision
from needle_thread_connection_tool import decode_attachment_response
from reconstruction_tools import fingerprints,sha256
from run_visibility_refiner_v5 import VisibilityRepairTools,messages_for as visibility_messages
from thread_visibility_repairs_v5 import (RepairLimits,load_inputs,load_parts,coverage,
    lift_ordered_reference_paths,distances)
from thread_topology_repairs_v6 import topology_proposals


def conservative_occluder(mask,needle,thread):
    """Visible foreground cannot itself license an inferred hidden connection.

    This disjointness constraint is not semantic validation of the remaining
    mask: it is still a visually inspectable hypothesis, not measured occlusion.
    """
    if mask.shape!=needle.shape or mask.shape!=thread.shape:raise ValueError('Mask dimensions differ')
    proposal=(mask>0)&~(needle>0)&~(thread>0)
    return proposal,dict(raw_pixels=int((mask>0).sum()),proposed_pixels=int(proposal.sum()),
        needle_overlap_excluded=int(((mask>0)&(needle>0)).sum()),
        visible_thread_overlap_excluded=int(((mask>0)&(thread>0)).sum()),
        semantic_correctness_validated=False,observed_foreground_cannot_license_hidden_geometry=True)


class SharedRepairToolsV6(VisibilityRepairTools):
    def __init__(self,case_path,pipeline,cache=None,registered_occluder=None):
        super().__init__(case_path,pipeline)
        self.original=json.loads(encoded(self.case));self.version=0;self.working_path=self.case_path
        self.cache=Path(cache).resolve() if cache else None;self.registered=registered_occluder
        self.dynamic={};self.dynamic_hash={};self.routed={};self.topology={};self.last_audit=None
        self.automatic_keys=set()
        self.paths['controller_v6']=Path(__file__)
        self.paths['topology_v6']=Path(__file__).with_name('thread_topology_repairs_v6.py')
        if cache:
            for p in self.cache.rglob('*'):
                if p.is_file():self.paths['cache/'+str(p.relative_to(self.cache))]=p
            self.paths['mask_producer']=Path(__file__).with_name('produce_shared_mask_repairs_v6.py')
        if registered_occluder:
            for k,v in registered_occluder['masks'].items():self.paths['registered_occluder/'+k]=v
            if registered_occluder['image_sha256']!={s:sha256(self.case['inputs'][s]) for s in ['left','right']}:
                raise ValueError('Registered manual occluder belongs to another image pair')
        self.expected=fingerprints(self.paths)

    def assert_unchanged(self,candidate=None):
        super().assert_unchanged(candidate)
        if hasattr(self,'dynamic') and fingerprints(self.dynamic)!=self.dynamic_hash:
            raise ValueError('Derived masks/case changed outside a logged tool')

    def register_derived(self,paths):
        self.dynamic.update(paths);self.dynamic_hash=fingerprints(self.dynamic)

    def refresh(self):
        base,_,masks,*_=load_inputs(self.case,self.pipeline);self.reference_preconditions={}
        for side,mask in masks.items():
            _,g,_,count,_=base.skeleton_graph(mask,0.,30)
            ends=sum(g.degree[n]==1 for n in g);branches=sum(g.degree[n]>2 for n in g)
            eligible=(ends==2*count and branches==0) or side in self.routed
            self.reference_preconditions[side]=dict(eligible=eligible,components=count,endpoints=ends,
                junction_pixels=branches,explicit_path_hypothesis=side in self.routed,
                reason='ordered projected paths available' if eligible else 'topology repair required')

    def read_cache(self,card):
        if not self.cache:raise ValueError('Actual model cache unavailable')
        folder=self.cache/self.case['case_id']/card;p=folder/'report.json'
        report=json.loads(p.read_text());runtime=self.cache/'runtime.json'
        producer=Path(__file__).with_name('produce_shared_mask_repairs_v6.py')
        cards={'segment.precision':dict(target='thread',prompt='suture thread',confidence_threshold=.35),
            'segment.recall':dict(target='thread',prompt='thin suture thread',confidence_threshold=.20),
            'occluder.sam3.instrument':dict(target='occluder',prompt='robotic surgical instrument',confidence_threshold=.20),
            'occluder.sam3.grasper':dict(target='occluder',prompt='surgical grasper',confidence_threshold=.20)}
        if (report.get('actual_model_producer') is not True or report.get('status')!='proposal_generated'
            or card not in cards or report.get('parameters')!=cards[card]
            or set(report.get('artifacts',{}))!={'left','right'}
            or report.get('full_frame') is not True or report.get('image_crop') is not None
            or report['runtime_sha256']!=sha256(runtime) or report['producer_sha256']!=sha256(producer)
            or report['helper_sha256']!=sha256(self.pipeline/'segment_needle_thread_with_sam3.py')
            or report['checkpoint_sha256']!=json.loads(runtime.read_text())['checkpoint_sha256']
            or any(report['input_sha256'][s]['sha256']!=sha256(self.original['inputs'][s]) for s in ['left','right'])):
            raise ValueError('Actual mask proposal provenance mismatch')
        masks={}
        for side,item in report['artifacts'].items():
            path=(self.cache/item['relative_path']).resolve()
            expected=(folder/(side+'_mask.png')).resolve()
            if path!=expected or not path.is_relative_to(self.cache) or sha256(path)!=item['sha256']:
                raise ValueError('Mask cache path/hash mismatch')
            masks[side]=str(path)
        return masks,report

    def lift(self,reference,tolerance,candidate,out):
        if reference not in self.routed:
            card=dict(operation='cross_view_recovery',parameters=dict(reference=reference,stereo_match_px=tolerance))
            proposal=super().execute(card,candidate,out)
        else:
            row=self.routed[reference]
            report=lift_ordered_reference_paths(self.case,self.pipeline,reference,out,
                row['ordered_paths_rc'],row['fragment_count'],row['discarded_short_components'],RepairLimits(stereo_match_px=tolerance))
            report.update(topology_hypothesis={k:v for k,v in row.items() if k!='ordered_paths_rc'},
                crossing_depth_order_recovered=False,topology_requires_visual_review=True)
            path=Path(out)/'fragments.npz'
            proposal=self.package(path,report,out)
        return self.audit_geometry(proposal)

    def package(self,path,report,out):
        report_path=Path(out)/'report.json';write_json(report_path,report)
        paths={'fragments':str(path.resolve()),'report':str(report_path.resolve())}
        images={s:str((Path(out)/(s+'_proposal.png')).resolve()) for s in ['left','right']}
        paths.update({'overlay.'+s:p for s,p in images.items()})
        return dict(stage='connected' if report['numerical_checks_passed'] and report['fragment_count']==1 else 'fragments',
            fragment_path=str(path.resolve()),report=report,images=images,
            protected_paths=paths,fingerprints=fingerprints(paths))

    def audit_geometry(self,proposal):
        # Verification must still account for immutable original visible masks;
        # removing evidence cannot make coverage improve by definition.
        parts=load_parts(proposal['fragment_path']);_,_,original_masks,_,_,K,baseline=load_inputs(self.original,self.pipeline)
        original_coverage={}
        from skimage.morphology import skeletonize
        from complex_case_diagnostics import project
        for side,mask in original_masks.items():
            canvas=np.zeros(mask.shape,'uint8')
            for part in parts:
                xy=project(part['points_m'],K,baseline if side=='right' else 0.)
                cv2.polylines(canvas,[np.rint(xy).astype('int32')],False,1,1)
            distances=cv2.distanceTransform(1-canvas,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            sk=skeletonize(mask);original_coverage[side]=float((distances[sk]<=8.).mean()) if sk.any() else 0.
        report=proposal['report'];report['original_visible_mask_coverage']=original_coverage
        report['mask_revision_audit']=self.last_audit
        if min(original_coverage.values())<.85:
            report['failures']=sorted(set(report['failures']+['original_visible_support_lost']))
            report['numerical_checks_passed']=False;proposal['stage']='fragments'
        write_json(proposal['protected_paths']['report'],report)
        proposal['fingerprints']=fingerprints(proposal['protected_paths'])
        return proposal

    def reconstruct_first(self,out):
        self.refresh()
        ranking=sorted(self.reference_preconditions,key=lambda s:(
            self.reference_preconditions[s]['junction_pixels'],s))
        side=next((s for s in ranking if self.reference_preconditions[s]['eligible']),None)
        if side is None:
            # Branches prevent a spline, but must not prevent observing/lifting
            # visible support before deciding which topology repair to request.
            out=Path(out);out.mkdir(parents=True,exist_ok=False)
            base,_,masks,_,depth,K,baseline=load_inputs(self.case,self.pipeline)
            _,graph,*_=base.skeleton_graph(masks['left'],0.,30)
            rc=np.asarray(sorted(graph),float).reshape(-1,2)
            z,_=base.sample_depth(depth,masks['left'],rc,3,.03,.5)
            good=np.isfinite(z);points=base.backproject(rc[good],z[good],K)
            from complex_case_diagnostics import project
            supported=distances(masks['right'],project(points,K,baseline))<=2. if len(points) else np.zeros(0,bool)
            path=out/'observed_support_unordered.npz'
            np.savez_compressed(path,points_m=points,reference_xy=rc[good,::-1],
                stereo_mask_supported=supported,geometry_inferred=np.zeros(len(points),bool))
            self.register_derived({str(path.resolve()):str(path.resolve())})
            return None,dict(status='visible_support_lifted_topology_unresolved',required_attempt_completed=True,
                ordered_path_available=False,point_count=len(points),stereo_supported_points=int(supported.sum()),
                diagnostic_artifact=str(path.resolve()),diagnostic_sha256=sha256(path),
                numerical_checks_passed=False,independent_accuracy_measured=False,training_eligible=False),None
        card=dict(operation='cross_view_recovery',parameters=dict(reference=side,stereo_match_px=2.))
        try:
            candidate=self.lift(side,2.,None,out)
            return candidate,dict(status='visible_reconstruction_executed',required_attempt_completed=True,
                reference=side,report=candidate['report']),card
        except Exception as error:
            return None,dict(status='visible_reconstruction_failed',required_attempt_completed=True,
                reference=side,error=f'{type(error).__name__}: {error}'),card

    def action_key(self,card):
        op=card['operation']
        version=0
        if op in ['cross_view_recovery','smooth_connection','topology_paths']:
            version={'masks':mask_content_id({s:self.case['inputs']['thread_'+s+'_mask'] for s in ['left','right']})}
            if op=='cross_view_recovery':
                row=self.routed.get(card['parameters']['reference'])
                version['paths']=digest([p.tolist() for p in row['ordered_paths_rc']]) if row else None
            if op=='smooth_connection':
                version['occluders']={s:sha256(self.case['inputs'][s+'_occluder'])
                    for s in ['left','right'] if s+'_occluder' in self.case['inputs']}
        return digest(dict(operation=op,parameters=card['parameters'],version=version))

    def menu(self,candidate,attempted):
        self.refresh();self.topology={}
        actions={'defer':dict(operation='defer',parameters={},effect='Keep observed/checked baseline and preserve ambiguity.')}
        if candidate and candidate['stage']=='connected':
            actions['keep']=dict(operation='keep',parameters={},effect='Retain the checked observed reconstruction; no gap repair is required.')
            return actions,[]
        actions_core,connections=super().menu(candidate,set())
        for aid,card in actions_core.items():
            if aid!='defer' and self.action_key(card) not in attempted:actions['v'+str(self.version)+'.'+aid]=card
        base,_,masks,*_=load_inputs(self.case,self.pipeline)
        for side,condition in self.reference_preconditions.items():
            if condition['junction_pixels']==0 or side in self.routed:continue
            seen=set()
            for spur in [2,6]:
                rows,_=topology_proposals(masks[side],self.pipeline,spur)
                for row in rows:
                    content=digest([p.tolist() for p in row['ordered_paths_rc']])
                    if content in seen:continue
                    seen.add(content)
                    params=dict(reference=side,max_spur_px=spur,proposal_id=row['proposal_id'],stereo_match_px=2.)
                    card=dict(operation='topology_paths',parameters=params,
                        effect='Preserve visible crossing strands as separate ordered-path hypotheses; audit short spurs and verify stereo/depth.')
                    aid=f'v{self.version}.topology.{side}.spur{spur}.{row["proposal_id"]}'
                    if self.action_key(card) not in attempted:
                        actions[aid]=card;self.topology[aid]=row
        for aid in ['segment.precision','segment.recall','occluder.sam3.instrument','occluder.sam3.grasper']:
            try:
                _,report=self.read_cache(aid)
                if aid.startswith('occluder') and not any(v['mask_pixels'] for v in report['views'].values()):continue
                card=dict(operation='occluder_mask' if aid.startswith('occluder') else 'resegment_thread',
                    parameters={'card':aid},effect='Actual cached SAM3 proposal; immutable needle and images; audited, then verified.')
                if self.action_key(card) not in attempted:actions[aid]=card
            except (FileNotFoundError,ValueError):pass
        for aid,params in [('stereo.trim',dict(mode='trim',tolerance_px=2,recovery_band_px=4)),
                          ('stereo.recover',dict(mode='recover',tolerance_px=2,recovery_band_px=6))]:
            card=dict(operation='stereo_mask',parameters=params,effect='Bounded stereo-supported mask proposal; preserve invalid-depth pixels and verify original coverage.')
            if self.action_key(card) not in attempted:actions[aid]=card
        if self.registered:
            card=dict(operation='occluder_mask',parameters={'card':'registered.manual'},
                effect='Reuse earlier assistant-traced tool masks for this exact stereo pair. Manual input; no autonomous segmentation claim.')
            if self.action_key(card) not in attempted:actions['occluder.registered.manual']=card
        return actions,connections

    def execute_repair(self,aid,card,candidate,out):
        self.assert_unchanged(candidate);out=Path(out);op=card['operation'];p=card['parameters']
        audit=None
        if op=='cross_view_recovery':return self.lift(p['reference'],p['stereo_match_px'],candidate,out),None
        if op=='smooth_connection':return self.audit_geometry(super().execute(card,candidate,out)),None
        if op=='topology_paths':
            row=self.topology[aid];self.routed[p['reference']]=row
            self.automatic_keys.add(self.action_key(dict(operation='cross_view_recovery',
                parameters=dict(reference=p['reference'],stereo_match_px=p['stereo_match_px']))))
            return self.lift(p['reference'],p['stereo_match_px'],candidate,out),dict(topology={k:v for k,v in row.items() if k!='ordered_paths_rc'})
        out.mkdir(parents=True,exist_ok=False)
        if op=='occluder_mask':
            if p['card']=='registered.manual':
                masks=self.registered['masks'];audit=dict(source='earlier assistant manual polygons',manual_input=True)
            else:masks,report=self.read_cache(p['card']);audit=dict(source='actual SAM3',actual_model_producer=True,cache_report=report)
            audit['views']={};audit['feedback_images']={}
            for side,path in masks.items():
                mask=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                if mask is None or mask.shape!=cv2.imread(self.original['inputs'][side]).shape[:2]:raise ValueError('Invalid occlusion mask')
                needle=cv2.imread(self.original['inputs']['needle_'+side+'_mask'],0)
                thread=cv2.imread(self.case['inputs']['thread_'+side+'_mask'],0)
                mask,metrics=conservative_occluder(mask,needle,thread);audit['views'][side]=metrics
                target=out/(side+'_occluder.png');cv2.imwrite(str(target),mask.astype('uint8')*255)
                self.case['inputs'][side+'_occluder']=str(target.resolve())
                # One full-frame feedback image per side, preserving thread
                # proposal lines and making actual mask scope visible to VLM.
                image=cv2.imread(candidate['images'][side] if candidate else self.original['inputs'][side])
                image[mask]=(image[mask]*.65+np.array([190,65,220])*.35).astype('uint8')
                cv2.putText(image,'Magenta: occluder hypothesis, semantic review required',(8,18),
                    cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1,cv2.LINE_AA)
                overlay=out/(side+'_occluder_feedback.png');cv2.imwrite(str(overlay),image)
                audit['feedback_images'][side]=str(overlay.resolve())
            self.version+=1;self.working_path=out/'derived_case.json';write_json(self.working_path,self.case)
            paths={str(p.resolve()):str(p.resolve()) for p in out.glob('*') if p.is_file()}
            self.register_derived(paths);write_json(out/'repair_audit.json',audit)
            return candidate,audit
        if op=='resegment_thread':
            masks,report=self.read_cache(p['card']);masks=dict(masks)
            for side,path in masks.items():
                mask=cv2.imread(path,0)>0
                needle=cv2.imread(self.original['inputs']['needle_'+side+'_mask'],0)>0
                mask&=~needle;target=out/(side+'_thread_mask.png');cv2.imwrite(str(target),mask.astype('uint8')*255)
                masks[side]=str(target.resolve())
            audit=dict(actual_model_producer=True,cache_report=report,frozen_needle_pixels_excluded=True)
        elif op=='stereo_mask':
            result=propose_stereo_thread_mask(self.original['inputs'],self.original['baseline_m'],p,self.pipeline,out/'backend')
            masks=result['proposed_masks'];audit=dict(stereo_mask_proposal=result)
        else:raise ValueError('Unknown repair operation')
        audit['mask_audit']=mask_revision_audit(self.original['inputs'],masks,self.pipeline)
        current={s:self.case['inputs']['thread_'+s+'_mask'] for s in ['left','right']}
        if mask_content_id(masks)==mask_content_id(current):
            audit['duplicate_mask_content']=True;write_json(out/'repair_audit.json',audit);return candidate,audit
        self.case['inputs'].update({'thread_'+s+'_mask':masks[s] for s in ['left','right']})
        self.routed={};self.version+=1;self.last_audit=audit['mask_audit']
        self.working_path=out/'derived_case.json';write_json(self.working_path,self.case)
        self.register_derived({str(p.resolve()):str(p.resolve()) for p in [self.working_path,*map(Path,masks.values())]})
        candidate,reconstruction,automatic=self.reconstruct_first(out/'verification')
        if automatic:self.automatic_keys.add(self.action_key(automatic))
        audit['reconstruction']=reconstruction;write_json(out/'repair_audit.json',audit)
        return candidate,audit


def heuristic_action(state):
    actions=state['actions']
    if 'keep' in actions:return 'keep'
    connections=[(aid,c) for aid,c in actions.items() if c['operation']=='smooth_connection']
    if connections:
        scores={p['proposal_id']:p['score'] for p in state['connection_candidates']}
        return min(connections,key=lambda item:scores[item[1]['parameters']['proposal_id']])[0]
    candidate=state.get('candidate') or {};failures=candidate.get('failures',[])
    cross=[(aid,c) for aid,c in actions.items() if c['operation']=='cross_view_recovery']
    if cross:
        def rank(item):
            p=item[1]['parameters'];view=state['reference_preconditions'][p['reference']]
            current=candidate.get('reference_view')
            return (0 if p['reference']==current and p['stereo_match_px']==4. and failures else 1,
                view['junction_pixels'],p['stereo_match_px'],p['reference'])
        return min(cross,key=rank)[0]
    rejected=Counter(f for c in state['connection_candidates'] for f in c['failures'])
    if rejected and any('visible_background_contradiction' in f for f in rejected):
        for aid in ['occluder.sam3.grasper','occluder.sam3.instrument','occluder.registered.manual']:
            if aid in actions:return aid
    topology=state.get('topology_candidates',[])
    eligible=[r for r in topology if r['action_id'] in actions]
    if eligible:return min(eligible,key=lambda r:(r['score'],r['pruned_source_pixels'],r['reference']))['action_id']
    for aid in ['segment.precision','segment.recall','stereo.recover','stereo.trim',
                'occluder.sam3.grasper','occluder.sam3.instrument','occluder.registered.manual','defer']:
        if aid in actions:return aid
    return 'defer'


def messages_for(state,images):
    compact=json.loads(encoded(state))
    if compact.get('candidate'):
        topology=compact['candidate'].get('topology_hypothesis')
        if topology:
            audit=topology.pop('audit',{})
            topology['pruned_source_pixels']=audit.get('pruned_source_pixels',0)
            topology.pop('ports',None)
    messages=visibility_messages(compact,images)
    prefix=('RECONSTRUCT FIRST, THEN ASSESS REPAIRS. The required initial visible reconstruction has already run. '
        'No gap/no repair needed means KEEP a checked reconstruction, not discard it. An image-boundary exit does not '
        'require completing an invisible thread. Choose keep when the connected proposal is supported; defer preserves '
        'the best checked baseline. For failures choose a matching operation: topology paths for projected crossings/tiny '
        'spurs, SAM3 or stereo mask repair for segmentation/correspondence, occluder mask for a join blocked behind a tool. '
        'A registered manual mask remains manual input. All sides have identical tools/budgets and frozen physical gates. '
        'Crossing routing and hidden connections are hypotheses, not recovered depth or independent truth.\n')
    messages[0]['content'][0]['text']=prefix+messages[0]['content'][0]['text']
    return messages


def run(case_path,pipeline,out,controller,infer_backend=None,cache=None,registered_occluder=None,max_decisions=6):
    if controller not in ['heuristic','agent'] or type(max_decisions) is not int or not 1<=max_decisions<=6:raise ValueError('Invalid controller/budget')
    if controller=='agent' and infer_backend is None:raise ValueError('Actual model required')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    tools=SharedRepairToolsV6(case_path,pipeline,cache,registered_occluder)
    start=time.perf_counter()
    candidate,initial,initial_card=tools.reconstruct_first(out/'initial_reconstruction')
    write_json(out/'initial_reconstruction.json',initial)
    attempted=set()
    if initial_card:attempted.add(tools.action_key(initial_card))
    best=candidate if candidate and candidate['stage']=='connected' else None
    partial=candidate if candidate and candidate['report']['numerical_checks_passed'] else None
    initial_checked=best is not None
    evidence={'mask':inspect_thread_masks(tools.case['inputs'],pipeline,out/'initial_inspection')}
    images=[('image.'+s,tools.case['inputs'][s]) for s in ['left','right']]
    history=[];status='budget_exhausted';review='pending';completed=0
    for iteration in range(max_decisions):
        tools.assert_unchanged(candidate);actions,connections=tools.menu(candidate,attempted)
        topology=[dict(action_id=aid,reference=actions[aid]['parameters']['reference'],score=r['score'],
            fragment_count=r['fragment_count'],pruned_source_pixels=r['audit']['pruned_source_pixels'],
            original_skeleton_coverage_2px=r['original_skeleton_coverage_2px'],ports=r['ports'],pairing_hypothesis=r['pairing_hypothesis'])
            for aid,r in tools.topology.items()]
        state=dict(protocol='shared_refiner_v6',target='scene_refinement',case_id=tools.case['case_id'],
            evidence=evidence,candidate=candidate['report'] if candidate else None,
            reference_preconditions=tools.reference_preconditions,scene_priors=tools.case.get('scene_priors',{}),
            initial_reconstruction_completed=True,checked_baseline_preserved=bool(best),
            connection_candidates=connections,topology_candidates=topology,actions=actions,
            remaining={'decisions':max_decisions-iteration},evidence_ids=['diagnosis',*[eid for eid,_ in images]],
            last_tool_feedback=history[-1].get('feedback') if history else None)
        state['scene_version']=digest(dict(state=state,inputs=tools.expected,attempted=sorted(attempted)))
        write_json(out/f'state_{iteration:03d}.json',state)
        if len(actions)==1:status='deferred_no_supported_action';review='unresolved';break
        before=time.perf_counter()
        try:
            if controller=='agent':
                messages=messages_for(state,images);write_json(out/f'input_{iteration:03d}.json',{'messages':messages})
                raw=infer_backend(messages);completed+=1;(out/f'raw_{iteration:03d}.txt').write_text(raw+'\n')
                decoded,_=decode_attachment_response(raw);decision=validate_decision(decoded,state)
            else:
                decision=validate_decision(dict(scene_version=state['scene_version'],target=state['target'],
                    action_id=heuristic_action(state),evidence_ids=['diagnosis'],reason='Frozen shared-menu heuristic; retain checked reconstruction, then prioritize bounded repairs by failure mechanism.'),state)
        except Exception as error:
            event=dict(iteration=iteration,status='invalid_or_failed_inference',error=f'{type(error).__name__}: {error}')
            history.append(event);write_json(out/f'event_{iteration:03d}.json',event);continue
        aid=decision['action_id'];card=actions[aid];event=dict(iteration=iteration,decision=decision,resolved_action=card,
            usage=getattr(infer_backend,'last_usage',None) if controller=='agent' else None)
        if aid in ['defer','keep']:
            status='deferred' if aid=='defer' else 'connected_proposal_needs_review';review=aid
            event['status']=status;history.append(event);write_json(out/f'event_{iteration:03d}.json',event);break
        attempted.add(tools.action_key(card))
        try:
            candidate,audit=tools.execute_repair(aid,card,candidate,out/f'tool_{iteration+1:03d}')
            attempted.update(tools.automatic_keys)
            event.update(status='tool_executed',repair_audit=audit,report=candidate['report'] if candidate else None)
            if candidate and candidate['stage']=='connected':best=candidate
            if candidate and candidate['report']['numerical_checks_passed']:partial=candidate
            evidence={'mask':inspect_thread_masks(tools.case['inputs'],pipeline,out/f'inspection_{iteration:03d}')}
            paths=(audit or {}).get('feedback_images') or (candidate['images'] if candidate else {s:evidence['mask']['views'][s]['overlay'] for s in ['left','right']})
            images=[('overlay.'+s,paths[s]) for s in ['left','right']]
            event['feedback']=dict(operation=card['operation'],numerical_checks_passed=bool(candidate and candidate['report']['numerical_checks_passed']),
                failures=candidate['report'].get('failures',[]) if candidate else ['no_orderable_visible_path'],
                mask_audit=tools.last_audit,occluder_mask_audit=(audit or {}).get('views') if card['operation']=='occluder_mask' else None,
                duplicate_mask_content=bool(audit and audit.get('duplicate_mask_content')))
        except Exception as error:event.update(status='tool_failed',error=f'{type(error).__name__}: {error}',feedback={'error':str(error)})
        attempted.update(tools.automatic_keys)
        event['runtime_seconds']=time.perf_counter()-before;history.append(event);write_json(out/f'event_{iteration:03d}.json',event)
    tools.assert_unchanged(candidate)
    returned=best or partial or candidate
    if best:status='connected_proposal_needs_review'
    summary=dict(protocol='shared_refiner_v6',controller=controller,status=status,history=history,
        initial_reconstruction=initial,baseline_geometry_preserved=initial_checked,
        checked_geometry_preserved=bool(best),terminal_review=review,
        checked_fragment_geometry_preserved=bool(partial),
        actual_model_used=bool(completed),model_calls_completed=completed,
        model_calls_attempted=infer_backend.index if controller=='agent' else 0,
        decision_count=len(history),candidate=returned,working_case=tools.case,
        provenance=tools.expected,runtime_seconds=time.perf_counter()-start,
        estimated_api_cost_usd=getattr(infer_backend,'estimated_api_cost_usd',0.) if controller=='agent' else 0.,
        model_manifest=getattr(infer_backend,'manifest',None),same_tools_and_limits=True,
        repair_budget=max_decisions,required_initial_attempts=1,independent_accuracy_measured=False,
        shared_reconstruction_accepted=False,physics_validated=False,training_eligible=False,
        manual_occluder_available=bool(registered_occluder),manual_occluder_used=any(
            h.get('resolved_action',{}).get('parameters',{}).get('card')=='registered.manual' for h in history))
    write_json(out/'summary.json',summary);return summary
