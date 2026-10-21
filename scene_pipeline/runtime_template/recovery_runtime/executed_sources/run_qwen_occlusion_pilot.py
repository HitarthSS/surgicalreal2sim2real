"""Read-only VLM image assessment; preserves raw outputs and measured token usage.

No reconstruction, physics, or model-generated code is executed. Each request
contains frozen messages and image hashes. Failures remain in the registry.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    requests = json.loads(args.requests.read_text())
    # Validate all original images before model loading or inference.
    for request in requests:
        for image in request["images"]:
            if digest(image["path"]) != image["sha256"]:
                raise ValueError("Original image provenance mismatch")
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    torch.set_num_threads(8)
    torch.manual_seed(0)
    started = time.perf_counter()
    manifest = {
        "model": str(args.model), "model_revision": args.model.name,
        "device": "cpu", "dtype": "float32", "threads": 8, "seed": 0,
        "do_sample": False, "max_new_tokens": args.max_new_tokens,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "api_cost_usd": 0, "compute_cost_usd": None,
        "source_sha256": digest(__file__), "requests_sha256": digest(args.requests),
        "independent_accuracy_measured": False, "geometry_changed": False,
    }
    write(args.out / "model_manifest.json", manifest)
    try:
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model, dtype=torch.float32, device_map="cpu",
            attn_implementation="sdpa", local_files_only=True,
        ).eval()
        manifest["load_seconds"] = time.perf_counter() - started
        write(args.out / "model_manifest.json", manifest)
        print(json.dumps({"status": "model_ready", **manifest}), flush=True)
        for request in requests:
            folder = args.out / request["request_id"]
            if folder.exists():
                raise ValueError("Refusing to overwrite existing inference")
            folder.mkdir()
            write(folder / "input.json", request)
            result = {"status": "failed", "actual_model_used": False,
                      "request_id": request["request_id"]}
            began = time.perf_counter()
            try:
                for image in request["images"]:
                    if digest(image["path"]) != image["sha256"]:
                        raise ValueError("Image changed before inference")
                inputs = processor.apply_chat_template(
                    request["messages"], tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt",
                )
                with torch.inference_mode():
                    output = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens,
                        do_sample=False, temperature=None, top_p=None, top_k=None,
                    )
                input_count = inputs["input_ids"].shape[-1]
                generated = output[0, input_count:]
                raw = processor.decode(generated, skip_special_tokens=True)
                (folder / "raw_response.txt").write_text(raw + "\n")
                result.update(
                    actual_model_used=True, input_tokens_total=int(input_count),
                    output_tokens=int(generated.numel()),
                    usage_accounting="actual_generation_tensors",
                    generation_hit_limit=int(generated.numel()) >= args.max_new_tokens,
                )
                for image in request["images"]:
                    if digest(image["path"]) != image["sha256"]:
                        raise ValueError("Image changed during inference")
                result["status"] = "completed"
            except Exception as error:
                result["error"] = f"{type(error).__name__}: {error}"
            result["runtime_seconds"] = time.perf_counter() - began
            write(folder / "result.json", result)
            print(json.dumps(result), flush=True)
    except Exception as error:
        write(args.out / "service_failure.json", {
            "error": f"{type(error).__name__}: {error}",
            "runtime_seconds": time.perf_counter() - started,
        })
        raise


if __name__ == "__main__":
    main()
