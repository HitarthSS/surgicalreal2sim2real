"""Compare prompt representations using saved development evidence; no inference."""
import argparse
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from compact_agent_protocol import make_state, messages_for
from reconstruction_tools import sha256
from run_bounded_agent_pilot import initial_prompt, feedback_content
from run_model_tool_checkpoint import build_messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[3]
    research = project / 'research/ismr_2027'
    args.out.mkdir(parents=True, exist_ok=False)
    cases = [('frame_000787', research / 'checkpoint_2026-10-11/example/thread',
              research / 'development_2026-10-09/inputs/frame_000787', 'left.jpg', 'right.jpg')]
    for case in ('snapshot_20261009_103902', 'snapshot_20261009_115511', 'snapshot_20261009_121654'):
        cases.append((case, research / 'new_data_2026-10-09/qp_ordering_trace_checkpoint' / case / 'fixed',
                      research / 'new_data_2026-10-09/recordings' / case, 'left_rect.png', 'right_rect.png'))
    rows = []
    for case_id, source, images, left_name, right_name in cases:
        out = args.out / case_id; out.mkdir()
        original = json.loads((source / 'events.jsonl').read_text().splitlines()[0])
        result_path = source / 'call_001/tool_result.json'
        result = json.loads(result_path.read_text())
        observation = original['observation']
        for side, name in (('left', left_name), ('right', right_name)):
            shutil.copyfile(images / name, out / (side + '.png' if name.endswith('.png') else side + '.jpg'))
        image_evidence = []
        observation['images'] = {}
        for side, name in (('left', left_name), ('right', right_name)):
            filename = side + ('.png' if name.endswith('.png') else '.jpg')
            # Portable review payload with explicit container paths; raw sources separately hashed.
            path = '/workspace/replay/' + case_id + '/' + filename
            observation['images'][side] = path
            image_evidence.append(('Original rectified ' + side + '; origin (0,0).', path))
        observation['available_tools'] = [schema['name'] for schema in original.get('tool_schemas', [])]
        if not observation['available_tools']:
            observation['available_tools'] = ['thread_occlusion', 'thread_distance', 'needle_arc', 'needle_mask_points']
        provenance = {"source_result_sha256": sha256(result_path),
                      "original_event_sha256": sha256(source / 'events.jsonl'),
                      "calibration_status": 'prior_scale_physical_units_unverified'}
        state, catalog = make_state(observation, provenance, 'thread',
                                   {'model_decisions': 1, 'tool_calls': 1, 'evidence_reads': 1}, result, 1)
        legacy = build_messages(observation, initial_prompt(observation))
        legacy.append({'role': 'assistant', 'content': [{'type': 'text', 'text': json.dumps({
            'tool': result['tool'], 'parameters': result['effective_parameters'],
            'reason': 'Recorded development operation; not a new model decision.'})}]})
        feedback = feedback_content(SimpleNamespace(out=source), result)
        for part in feedback:
            if part['type'] == 'image':
                path = Path(part['url'])
                shutil.copyfile(path, out / path.name)
                part['url'] = '/workspace/replay/' + case_id + '/' + path.name
                image_evidence.append((path.name + '; estimated source support.', part['url']))
        legacy.append({'role': 'user', 'content': feedback})
        compact = messages_for(state, image_evidence)
        for filename, value in (('state.json', state), ('evidence_catalog.json', catalog),
                                ('legacy_messages.json', legacy), ('compact_messages.json', compact)):
            (out / filename).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
        def text_bytes(messages):
            return sum(len(part['text'].encode()) for msg in messages for part in msg['content'] if part['type']=='text')
        rows.append({'case_id': case_id, 'source_result_sha256': sha256(result_path),
                     'full_candidate_records': len(result.get('backend_report', {}).get('candidates', [])),
                     'catalog_candidate_records': sum(key.startswith('candidate.') for key in catalog),
                     'failures_preserved': state['current_result']['failures']==(result.get('verification') or {}).get('failures', []),
                     'legacy_text_bytes': text_bytes(legacy), 'compact_text_bytes': text_bytes(compact),
                     'validity': state['validity'], 'actual_model_used': False})
    report = {'kind': 'saved_evidence_prompt_representation_replay', 'rows': rows,
              'token_savings_measured': False, 'quality_comparison_measured': False,
              'note': 'Byte counts are not tokens or evidence of equivalent model quality. Images identical in each pair.'}
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
