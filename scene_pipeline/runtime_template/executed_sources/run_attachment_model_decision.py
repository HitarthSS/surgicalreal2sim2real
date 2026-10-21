"""One cached VLM decision; retain invalid output without a heuristic fallback."""
import argparse
import json
from pathlib import Path
import time

from needle_thread_connection_tool import validate_attachment_decision, decode_attachment_response
from newton_scene_adapter import sha256
from run_compact_agent import MeteredGemma


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--messages', type=Path, required=True)
    p.add_argument('--image-root', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--gemma-runner', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    state=json.loads(a.state.read_text()); messages=json.loads(a.messages.read_text())
    for m in messages:
        for item in m['content']:
            if item['type']=='image': item['url']=str(a.image_root/Path(item['url']).name)
    inputs={str(p):sha256(p) for p in (a.state,a.messages)}
    for m in messages:
        for item in m['content']:
            if item['type']=='image':inputs[item['url']]=sha256(item['url'])
    (a.out/'input.json').write_text(json.dumps(dict(messages=messages,input_sha256=inputs,
        model=a.model,seed=0,sampling=False,max_calls=1,retries=0,image_tokens_per_view=560,max_new_tokens=256),indent=2)+'\n')
    started=time.perf_counter(); report=dict(actual_model_used=False,status='started',api_cost_usd=0,compute_cost_usd=None)
    try:
        import torch
        torch.manual_seed(0)
        model=MeteredGemma(a.model,a.gemma_runner)
        model.generation.max_new_tokens=256
        load_time=time.perf_counter()-started
        raw=model(messages)
        report.update(actual_model_used=True,load_seconds=load_time,usage=model.last_usage)
        (a.out/'raw_response.txt').write_text(raw+'\n')
        if any(sha256(p)!=h for p,h in inputs.items()):raise ValueError('Model inputs changed')
        decoded,transport=decode_attachment_response(raw)
        decision,candidate=validate_attachment_decision(decoded,state)
        (a.out/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
        report.update(status='valid_decision',decision=decision,candidate=candidate,transport=transport)
    except Exception as exc:
        report.update(status='model_or_decision_failed',error=f'{type(exc).__name__}: {exc}')
        import traceback;traceback.print_exc()
    report.update(runtime_seconds=time.perf_counter()-started,source_sha256=sha256(__file__))
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
