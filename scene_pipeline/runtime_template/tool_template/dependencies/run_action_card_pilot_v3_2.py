"""Matched development comparison: fixed, original rule, common-menu heuristic, VLM.

Common preprocessing and verifiers are unchanged. No case-specific controller
logic, favorable-case exclusion, test data, physics claim or training.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from compact_agent_protocol import accepted_target, signature
from compact_agent_protocol_v3_2 import (make_state, image_evidence, heuristic_action,
                                      validate_decision, resolve_action)
from reconstruction_tools import ReconstructionSession, heuristic_request, fingerprints
from run_multiscene_pilot import QueueModel
from run_bounded_agent_pilot import assert_unchanged
from run_compact_agent_v3_2 import run_compact_loop
from scene_feedback_v2 import SceneFeedback
from needle_thread_connection_tool import decode_attachment_response, validate_attachment_decision


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def attach_scene(root, cid, label, session, provider, result, queue):
    """Existing shared assembly tool; one separate topology decision for the VLM."""
    start = time.perf_counter(); out = root / 'attachments' / cid / label; out.mkdir(parents=True)
    bundle = provider.out / result['call_id'] / 'scene_bundle.npz'
    source_paths = {n: Path(__file__).with_name(n) for n in
        ('build_needle_thread_connection.py', 'needle_thread_connection_tool.py', 'newton_scene_adapter.py')}
    source_hashes = fingerprints(source_paths)
    command = [sys.executable, str(source_paths['build_needle_thread_connection.py']), '--bundle', str(bundle),
        '--left', session.inputs['left'], '--right', session.inputs['right'], '--baseline-m', str(session.case['baseline_m']), '--out', str(out)]
    summary = {'controller': label, 'case_id': cid, 'actual_model_used': False, 'max_topology_model_calls': 1,
               'physics_validated': False, 'training_eligible': False, 'sources': source_hashes}
    try:
        provider.assert_unchanged()
        preparation = subprocess.run(command, capture_output=True, text=True, check=True)
        (out / 'prepare_stdout.txt').write_text(preparation.stdout)
        state = json.loads((out / 'state.json').read_text())
        if label == 'agent':
            messages = json.loads((out / 'messages.json').read_text())
            protected = {k: out / k for k in ('state.json', 'messages.json', 'left_endpoints.png', 'right_endpoints.png')}
            protected['source_bundle'] = bundle; expected = fingerprints(protected)
            raw = queue(messages); summary.update(actual_model_used=True, usage=queue.last_usage)
            (out / 'raw_response.txt').write_text(raw + '\n')
            provider.assert_unchanged(); assert_unchanged(session, source_paths, source_hashes)
            if fingerprints(protected) != expected: raise ValueError('Attachment evidence changed during inference')
            decoded, transport = decode_attachment_response(raw)
            decision, _ = validate_attachment_decision(decoded, state); summary['transport'] = transport
        else:
            eligible = [c for c in state['candidates'] if c['distance_m'] <= state['max_connector_length_m'] and c['endpoints_supported']]
            if len(eligible) != 1: raise ValueError('No unique supported heuristic attachment')
            decision = {'scene_version': state['scene_version'], 'target': 'attachment', 'operation': 'attach_needle_thread',
                        'parameters': {'candidate_id': eligible[0]['id']}, 'evidence_ids': ['stereo', 'endpoints'],
                        'reason': 'Common geometric rule: one supported endpoint pair within the unchanged connector bound.'}
            validate_attachment_decision(decision, state)
        write(out / 'decision.json', decision)
        execution = subprocess.run(command + ['--decision', str(out / 'decision.json'), '--controller',
            'agent' if label == 'agent' else 'heuristic'], capture_output=True, text=True, check=True)
        (out / 'execute_stdout.txt').write_text(execution.stdout)
        report = json.loads((out / 'connection_report.json').read_text())
        summary.update(status=report['status'], decision=decision, connection_report=str(out / 'connection_report.json'),
                       derived_bundle=str(out / 'connected_scene_bundle.npz') if report['attachment'] else None)
    except Exception as error:
        summary.update(status='attachment_failed', error=f'{type(error).__name__}: {error}')
    summary['runtime_seconds'] = time.perf_counter() - start
    write(out / 'attachment_summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('root', 'pipeline-tools', 'original-src', 'latest', 'gemma-runner'):
        p.add_argument('--' + n, type=Path, required=True)
    p.add_argument('--stage', choices=('controls', 'agent'), required=True)
    a = p.parse_args(); root = a.root; rows = []; queue = QueueModel(root / 'results/model_queue')
    for casepath in sorted((root / 'cases').glob('*.json')):
        case = json.loads(casepath.read_text()); cid = case['case_id']
        context = root / 'results' / cid / 'needle_context'
        if a.stage == 'controls':
            needle = ReconstructionSession(casepath, context, 'fixed', 'common_needle_context',
                a.pipeline_tools, max_calls=1, original_src=a.original_src)
            needle.call({'tool': 'needle_arc', 'parameters': {}, 'reason': 'Identical existing free-radius needle context, not model refinement.'})
        labels = ('fixed', 'initial_rule', 'menu_heuristic') if a.stage == 'controls' else ('agent',)
        for label in labels:
            started = time.perf_counter(); folder = root / 'results' / cid / 'thread' / label
            family = label if label in ('fixed', 'agent') else 'heuristic'
            session = ReconstructionSession(casepath, folder, family, label + '_actioncards_v3_2',
                a.pipeline_tools, max_calls=2, original_src=a.original_src)
            result = session.call({'tool': 'thread_qp', 'parameters': {}, 'reason': 'Identical default production baseline before controller refinement.'})
            provider = SceneFeedback(session, context / 'call_001/tool_result.json', a.latest,
                root / 'scene_feedback' / cid / label)
            decisions = []
            if label == 'initial_rule' and not accepted_target(result, 'thread'):
                req = heuristic_request(session.observe())
                if signature({'operation': req['tool'], 'parameters': req['parameters']}) != signature(
                        {'operation': result['tool'], 'parameters': result['effective_parameters']}):
                    result = session.call(req)
            if label == 'menu_heuristic':
                attempted = [{'operation': result['tool'], 'parameters': result['effective_parameters'],
                              'status': result['status']}]
                for revision in range(2):
                    feedback = provider(result)
                    state, catalog = make_state(session.observe(), {'inputs': session.input_fingerprints,
                        'sources': session.source_fingerprints}, 'thread',
                        {'model_decisions': 2 - revision, 'tool_calls': session.max_calls - session.index, 'evidence_reads': 1},
                        result, revision, attempted, image_evidence(session, 'thread', result), feedback)
                    decision = validate_decision(heuristic_action(state), state, catalog)
                    resolved = resolve_action(decision, state)
                    decisions.append({'state': state, 'decision': decision, 'resolved_action': resolved})
                    if resolved['tool'] in ('finish', 'defer'):
                        break
                    result = session.call(resolved)
                    attempted.append({'operation': result['tool'], 'parameters': result['effective_parameters'], 'status': result['status']})
                write(folder / 'heuristic_decisions.json', decisions)
            feedback = provider(result)
            summary = {'status': result['status'], 'last_tool_result': result, 'scene_feedback': feedback,
                       'model_calls_completed': 0, 'invalid_decisions': 0, 'actual_model_used': False}
            if label == 'menu_heuristic':
                summary['status'] = 'control_finished' if feedback['preflight_passed'] else 'control_deferred'
            if label == 'agent':
                summary = run_compact_loop(session, queue, {'target': 'thread', 'name': 'google/gemma-4-E4B-it',
                    'revision': 'ee0ef6023621cff504d758262d4e04895a5af4a2', 'device': 'CPU', 'dtype': 'float32',
                    'sampling': False, 'gemma_runner': str(a.gemma_runner)}, result, scene_feedback_provider=provider)
                result = summary['last_tool_result']; feedback = summary['scene_feedback']
            row = {'case_id': cid, 'session_id': case['session_id'], 'controller': label, 'status': summary['status'],
                   'target_input_consistency_passed': accepted_target(result, 'thread'),
                   'assembly_preflight_passed': feedback['preflight_passed'],
                   'finish_blocked_by_scene': feedback['blocking_for_finish'],
                   'assembly': feedback['assembly'], 'attachment': feedback['attachment'],
                   'tool_calls': session.index, 'model_calls_completed': summary['model_calls_completed'],
                   'invalid_decisions': summary['invalid_decisions'], 'actual_model_used': summary['actual_model_used'],
                   'runtime_seconds_total': time.perf_counter() - started, 'error': summary.get('error'),
                   'independent_accuracy_measured': False, 'physics_validated': False, 'training_eligible': False}
            if feedback['preflight_passed'] and label != 'fixed':
                row['attachment_execution'] = attach_scene(root, cid, label, session, provider, result, queue)
            write(folder / 'pilot_result.json', row); rows.append(row)
            write(root / f'results_{a.stage}.json', rows); print(json.dumps(row), flush=True)
    if a.stage == 'agent':
        (queue.root / 'STOP').touch()


if __name__ == '__main__':
    main()
