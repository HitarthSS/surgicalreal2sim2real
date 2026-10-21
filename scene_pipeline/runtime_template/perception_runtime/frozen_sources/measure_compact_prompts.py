"""CPU-only processor tokenization of matched image/text prompts; no model load."""
import argparse
import json
from pathlib import Path
import time


def main():
    from transformers import AutoConfig, AutoProcessor
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.image_processor.max_soft_tokens = 560
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    image_id = getattr(config, 'image_token_id', getattr(config, 'image_token_index', None))
    rows = []
    for case in sorted(p for p in args.replay.iterdir() if p.is_dir()):
        row = {'case_id': case.name}
        for mode in ('legacy', 'compact'):
            messages = json.loads((case / (mode + '_messages.json')).read_text())
            start = time.perf_counter()
            tokens = processor.apply_chat_template(messages, tokenize=True, return_dict=True,
                return_tensors='pt', add_generation_prompt=True, enable_thinking=False,
                processor_kwargs={'truncation': False})
            ids = tokens['input_ids']
            image_tokens = None if image_id is None else int((ids == image_id).sum())
            row[mode] = {'input_tokens_total': int(ids.numel()), 'image_placeholder_tokens': image_tokens,
                         'non_image_input_tokens': None if image_tokens is None else int(ids.numel())-image_tokens,
                         'image_count': sum(p['type']=='image' for m in messages for p in m['content']),
                         'processor_runtime_seconds': time.perf_counter()-start}
        rows.append(row)
    report = {'kind': 'actual_processor_prompt_tokenization_only', 'rows': rows,
              'image_token_id': image_id, 'max_soft_tokens': 560, 'model_weights_loaded': False,
              'model_generation_performed': False, 'quality_equivalence_measured': False,
              'gpu_used': False, 'cached_tokens': None, 'api_cost_usd': 0., 'compute_cost_usd': None}
    (args.out / 'token_counts.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
