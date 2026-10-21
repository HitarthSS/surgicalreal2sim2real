"""Development pilot using existing shared tools, verifiers and compact protocol.

Model inference runs in a separate existing container because its runtime lacks
the production QP dependency. File transport changes neither prompts nor tools.
"""
import argparse
import json
from pathlib import Path
import time

from compact_agent_protocol import accepted_target, signature
from reconstruction_tools import ReconstructionSession, heuristic_request, fingerprints
import run_compact_agent
from needle_thread_connection_tool import decode_attachment_response


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class QueueModel:
    def __init__(self, root, timeout=600):
        self.root, self.timeout = root, timeout
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = 0
        self.last_usage = None

    def __call__(self, messages):
        self.index += 1
        stem = self.root / f'request_{self.index:06d}'
        images = {str(i): item['url'] for i, item in enumerate(
            x for m in messages for x in m['content'] if x['type'] == 'image')}
        request = {'messages': messages, 'image_hashes': fingerprints(images)}
        temporary = stem.with_suffix('.tmp')
        write(temporary, request)
        temporary.rename(stem.with_suffix('.json'))
        start = time.monotonic()
        response_path = stem.with_suffix('.response.json')
        while not response_path.exists():
            failure = self.root / 'service_failure.json'
            if failure.exists():
                raise RuntimeError('Inference service failed: ' + failure.read_text())
            if time.monotonic() - start > self.timeout:
                raise TimeoutError('Inference queue wait exceeded recorded CPU budget')
            time.sleep(.25)
        response = json.loads(response_path.read_text())
        if fingerprints(images) != request['image_hashes']:
            raise ValueError('Inference image changed')
        if response['status'] != 'completed':
            raise RuntimeError(response['error'])
        self.last_usage = response['usage']
        return stem.with_suffix('.raw.txt').read_text()


def run(args):
    cases = sorted(args.cases.glob('*.json'))
    registry = []
    queue = QueueModel(args.out / 'model_queue')
    # Same strict, whole-document transport decoder already tested for attachment.
    # This does not change any semantic validator, evidence, operation or gate.
    run_compact_agent.decode_transport = decode_attachment_response
    for case in cases:
        for target in ('thread', 'needle'):
            for controller in args.controllers:
                folder = args.out / json.loads(case.read_text())['case_id'] / target / controller
                if folder.exists():
                    raise FileExistsError('Refusing to replace pilot output: ' + str(folder))
                session = None
                start = time.perf_counter()
                try:
                    session = ReconstructionSession(case, folder, controller,
                        'compact_gemma_cpu_fp32_v1' if controller == 'agent' else 'initial_rule_common_baseline_v1',
                        args.pipeline_tools, max_calls=2, original_src=args.original_src)
                    result = session.call({'tool': 'thread_qp' if target == 'thread' else 'needle_arc',
                        'parameters': {}, 'reason': 'Identical default baseline before any controller repair.'})
                    summary = {'status': result['status'], 'tool_calls': session.index,
                               'last_tool_result': result, 'actual_model_used': False,
                               'model_calls_completed': 0}
                    if controller == 'heuristic' and target == 'thread' and not accepted_target(result, target):
                        request = heuristic_request(session.observe())
                        old = signature({'operation': result['tool'], 'parameters': result['effective_parameters']})
                        from reconstruction_tools import parameters_for
                        new = signature({'operation': request['tool'],
                                         'parameters': parameters_for(request['tool'], request['parameters'])})
                        if old != new:
                            result = session.call(request)
                            summary.update(status=result['status'], tool_calls=session.index, last_tool_result=result)
                        else:
                            summary['status'] = 'rejected_no_distinct_heuristic_operation'
                    if controller == 'agent':
                        summary = run_compact_agent.run_compact_loop(session, queue,
                            {'target': target, 'name': 'google/gemma-4-E4B-it',
                             'revision': 'ee0ef6023621cff504d758262d4e04895a5af4a2',
                             'sampling': False, 'device': 'CPU', 'dtype': 'float32',
                             'gemma_runner': str(args.gemma_runner)}, result)
                    row = {'case_id': session.case['case_id'], 'session_id': session.case['session_id'],
                        'target': target, 'controller': controller, 'status': summary['status'],
                        'target_input_consistency_passed': accepted_target(summary['last_tool_result'], target),
                        'runtime_seconds_total': time.perf_counter()-start,
                        'tool_calls': session.index, 'actual_model_used': summary['actual_model_used'],
                        'model_calls_completed': summary.get('model_calls_completed', 0),
                        'error': summary.get('error'), 'output': str(folder),
                        'independent_accuracy_measured': False, 'training_eligible': False}
                    write(folder / 'pilot_result.json', row)
                except Exception as error:
                    row = {'case_id': json.loads(case.read_text())['case_id'], 'target': target,
                        'controller': controller, 'status': 'pilot_infrastructure_failed',
                        'error': f'{type(error).__name__}: {error}',
                        'target_input_consistency_passed': False,
                        'runtime_seconds_total': time.perf_counter()-start,
                        'independent_accuracy_measured': False, 'training_eligible': False}
                registry.append(row)
                write(args.out / ('results_' + '_'.join(args.controllers) + '.json'), registry)
                print(json.dumps(row), flush=True)
    if 'agent' in args.controllers:
        (queue.root / 'STOP').touch()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cases', 'out', 'pipeline-tools', 'original-src', 'gemma-runner'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--controllers', nargs='+', choices=('fixed', 'heuristic', 'agent'), required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
