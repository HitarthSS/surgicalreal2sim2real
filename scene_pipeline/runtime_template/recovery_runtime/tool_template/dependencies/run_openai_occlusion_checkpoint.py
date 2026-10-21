"""One API request on the frozen case 5 image and unchanged occlusion prompt."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='gpt-6-astra')
    p.add_argument('--key-file', type=Path)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    prior = root / 'occlusion_assessment_2026-10-12'
    old = json.loads((prior / 'request_manifest.json').read_text())
    image = Path(old['image'])
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    if digest != old['image_sha256']:
        raise ValueError('Image changed since Gemma assessment')
    prompt = (prior / 'prompt.txt').read_text()
    a.out.mkdir(parents=True, exist_ok=True)
    record = {'status': 'prepared_not_run', 'actual_model_used': False,
              'model_requested': a.model, 'image': str(image), 'image_sha256': digest,
              'prompt': prompt, 'image_detail': 'original', 'reasoning_effort': 'low',
              'max_output_tokens': 4096, 'retry_limit': 0, 'store': False,
              'geometry_changed': False, 'independent_accuracy_measured': False,
              'budget_note': 'API output cap includes reasoning tokens; not equivalent to local 512 generated-token cap.'}
    manifest = a.out / 'request_manifest.json'
    if (a.out / 'api_response.json').exists():
        raise ValueError('Refusing to overwrite completed API response')
    manifest.write_text(json.dumps(record, indent=2) + '\n')
    if a.prepare_only:
        print(json.dumps({'status': record['status'], 'image_sha256': digest}))
        return
    key = os.environ.get('OPENAI_API_KEY')
    if not key and a.key_file:
        key = a.key_file.read_text().strip()
    if not key:
        raise SystemExit('OpenAI API key is not configured; no API request made.')
    encoded = base64.b64encode(image.read_bytes()).decode('ascii')
    start = time.perf_counter()
    try:
        payload = dict(model=a.model, store=False, reasoning={'effort': 'low'},
            max_output_tokens=4096, input=[{'role': 'user', 'content': [
                {'type': 'input_image', 'image_url': 'data:image/png;base64,' + encoded,
                 'detail': 'original'},
                {'type': 'input_text', 'text': prompt},
            ]}],
        )
        request = urllib.request.Request('https://api.openai.com/v1/responses',
            data=json.dumps(payload).encode(),
            headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'},
            method='POST')
        with urllib.request.urlopen(request, timeout=180) as transport:
            response = json.load(transport)
        (a.out / 'api_response.json').write_text(json.dumps(response, indent=2) + '\n')
        output_text = '\n'.join(part['text'] for item in response.get('output', [])
            if item.get('type') == 'message' for part in item.get('content', [])
            if part.get('type') == 'output_text')
        (a.out / 'raw_response.txt').write_text(output_text + '\n')
        record.update(status=response.get('status'), actual_model_used=True,
                      model_returned=response.get('model'), response_id=response.get('id'),
                      usage=response.get('usage'),
                      runtime_seconds=time.perf_counter() - start,
                      api_cost_usd=None)
        if hashlib.sha256(image.read_bytes()).hexdigest() != digest:
            record['status'] = 'failed_post_inference_provenance'
        print(output_text)
    except Exception as error:
        record.update(status='failed', error_type=type(error).__name__,
                      http_status=getattr(error, 'status_code', getattr(error, 'code', None)),
                      runtime_seconds=time.perf_counter() - start)
        raise
    finally:
        manifest.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
