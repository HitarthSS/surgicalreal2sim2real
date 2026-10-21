"""Summarize all pilot outcomes without dropping failures or claiming accuracy."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();r=a.root
    prereg=json.loads((r/'preregistration.json').read_text())
    rows=sum((json.loads((r/'results'/name).read_text()) for name in
              ('results_fixed_heuristic.json','results_agent.json')),[])
    counts={c:{t:sum(x['target_input_consistency_passed'] for x in rows if x['controller']==c and x['target']==t)
               for t in ('thread','needle')} for c in ('fixed','heuristic','agent')}
    matrix=[]
    for cid in prereg['case_ids']:
        item={'case_id':cid}
        for controller in counts:
            row=next(x for x in rows if x['case_id']==cid and x['controller']==controller and x['target']=='thread')
            item[controller]='pass' if row['target_input_consistency_passed'] else 'reject'
            item[controller+'_status']=row['status']
            item[controller+'_error']=row.get('error')
        matrix.append(item)
    responses=[json.loads(p.read_text()) for p in sorted((r/'results/model_queue').glob('*.response.json'))]
    agent_summaries=[json.loads(p.read_text()) for p in sorted((r/'results').glob('*/thread/agent/compact_summary.json'))]
    model=dict(actual_calls=len(responses),model_load_count=1,
        inference_seconds=sum(x['runtime_seconds'] for x in responses),
        input_tokens_total=sum(x['usage']['input_tokens_total'] for x in responses),
        image_placeholder_tokens=sum(x['usage']['image_placeholder_tokens'] for x in responses),
        output_tokens=sum(x['usage']['output_tokens'] for x in responses),
        invalid_evidence_id_cases=sum(x.get('error')=='Cite one to three available evidence IDs' for x in agent_summaries),
        exhausted_evidence_budget_cases=sum(x.get('error')=='Evidence-read budget exhausted' for x in agent_summaries),
        valid_defer_cases=sum(x['status']=='deferred' for x in agent_summaries),
        new_reconstruction_operations_executed=sum(x['tool_calls']-1 for x in agent_summaries),
        api_cost_usd=0,compute_cost_usd=None,device='CPU',dtype='float32')
    parity=[]
    for cid in prereg['case_ids']:
        for target,artifact in (('thread','shared_thread_samples.npz'),('needle','shared_needle_arc.npz')):
            files={c:sorted((r/'results'/cid/target/c).glob('call_*/backend/'+artifact)) for c in counts}
            for controller in ('heuristic','agent'):
                if not files['fixed'] or not files[controller]:continue
                if next(x for x in rows if x['case_id']==cid and x['controller']==controller and x['target']==target)['target_input_consistency_passed']:
                    with np.load(files['fixed'][-1]) as base,np.load(files[controller][-1]) as other:
                        p0,p1=base['points_m'],other['points_m']
                        parity.append(dict(case_id=cid,target=target,controller=controller,
                            compared_to='fixed',same_shape=p0.shape==p1.shape,
                            points_bit_identical=np.array_equal(p0,p1),
                            max_absolute_point_difference_m=float(np.max(np.abs(p0-p1))) if p0.shape==p1.shape else None))
    summary=dict(case_count=6,selection=prereg['selection'],counts=counts,thread_matrix=matrix,
        model=model,geometry_comparisons=parity,
        assembly=json.loads((r/'assembly_results_fixed_heuristic.json').read_text())+
                 json.loads((r/'assembly_results_agent.json').read_text()),
        physics=json.loads((r/'physics_stage_summary.json').read_text()),
        independent_accuracy_measured=False,policy_demonstrations_validated=0,training_eligible=False)
    (r/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    with (r/'thread_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(matrix[0]));writer.writeheader();writer.writerows(matrix)
    print(json.dumps({'counts':counts,'model':model},indent=2))


if __name__=='__main__':main()
