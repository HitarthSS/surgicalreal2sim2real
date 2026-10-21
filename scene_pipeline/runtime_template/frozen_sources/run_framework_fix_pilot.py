"""Three-case development rerun with live scene feedback and unchanged tools."""
import argparse
import json
from pathlib import Path
import time

from compact_agent_protocol import accepted_target,signature
from reconstruction_tools import ReconstructionSession,heuristic_request
from run_multiscene_pilot import QueueModel
from run_compact_agent_v2 import run_compact_loop
from scene_feedback_v2 import SceneFeedback


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','pipeline-tools','original-src','latest','gemma-runner'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--stage',choices=('controls','agent'),required=True)
    a=p.parse_args();r=a.root;rows=[];queue=QueueModel(r/'results/model_queue')
    for casepath in sorted((r/'cases').glob('*.json')):
        case=json.loads(casepath.read_text());cid=case['case_id']
        context=r/'results'/cid/'needle_context'
        if a.stage=='controls':
            needle=ReconstructionSession(casepath,context,'fixed','unchanged_common_needle_context',
                                         a.pipeline_tools,max_calls=1,original_src=a.original_src)
            needle_result=needle.call({'tool':'needle_arc','parameters':{},
                'reason':'Identical free-radius needle context for all controllers, not model refinement.'})
            write(context/'context_result.json',needle_result)
        controllers=('fixed','heuristic') if a.stage=='controls' else ('agent',)
        for controller in controllers:
            start=time.perf_counter();folder=r/'results'/cid/'thread'/controller
            session=ReconstructionSession(casepath,folder,controller,'scene_feedback_v2' if controller=='agent'
                else 'unchanged_initial_rule_common_baseline',a.pipeline_tools,max_calls=2,original_src=a.original_src)
            result=session.call({'tool':'thread_qp','parameters':{},'reason':'Unchanged default production baseline before refinement.'})
            provider=SceneFeedback(session,context/'call_001/tool_result.json',a.latest,r/'scene_feedback'/cid/controller)
            if controller=='heuristic' and not accepted_target(result,'thread'):
                request=heuristic_request(session.observe())
                if signature({'operation':request['tool'],'parameters':request['parameters']})!=signature(
                    {'operation':result['tool'],'parameters':result['effective_parameters']}):result=session.call(request)
            summary={'status':result['status'],'actual_model_used':False,'model_calls_completed':0,
                     'invalid_decisions':0,'last_tool_result':result,'scene_feedback':provider(result)}
            if controller=='agent':
                summary=run_compact_loop(session,queue,{'target':'thread','name':'google/gemma-4-E4B-it',
                    'revision':'ee0ef6023621cff504d758262d4e04895a5af4a2','device':'CPU','dtype':'float32',
                    'sampling':False,'gemma_runner':str(a.gemma_runner)},result,scene_feedback_provider=provider)
            feedback=summary['scene_feedback']
            row={'case_id':cid,'controller':controller,'status':summary['status'],
                 'target_input_consistency_passed':accepted_target(summary['last_tool_result'],'thread'),
                 'assembly_preflight_passed':feedback['preflight_passed'],
                 'finish_blocked_by_scene':feedback['blocking_for_finish'],
                 'attachment':feedback['attachment'],'assembly':feedback['assembly'],
                 'tool_calls':session.index,'model_calls_completed':summary['model_calls_completed'],
                 'invalid_decisions':summary['invalid_decisions'],'actual_model_used':summary['actual_model_used'],
                 'runtime_seconds_total':time.perf_counter()-start,'error':summary.get('error'),
                 'independent_accuracy_measured':False,'physics_validated':False,'training_eligible':False}
            write(folder/'pilot_result.json',row);rows.append(row);write(r/f'results_{a.stage}.json',rows)
            print(json.dumps(row),flush=True)
    if a.stage=='agent':(queue.root/'STOP').touch()


if __name__=='__main__':main()
