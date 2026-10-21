#!/usr/bin/env python3
"""One real vision-model decision through the unchanged reconstruction interface.

No retry, heuristic fallback, training, or physical actuation. The existing Gemma
runner supplies inference only; its contact-specific prompt/parser are not used.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import random
import time

from reconstruction_tools import ReconstructionSession, fingerprints, sha256, strict_json, tool_schemas


def make_prompt(observation):
    return (
        "Choose exactly ONE next reconstruction operation for this surgical stereo case. "
        "The task is thread reconstruction for a future thread grasp and 5 mm pull experiment. "
        "Inspect the original left and right images and the supplied numerical observation. "
        "Masks and depth are predictions, not ground truth. No previous tool outcomes are supplied. "
        "Choose an available tool and bounded parameters; omitted parameters use the listed defaults. "
        "You cannot change calibration, acceptance checks, paths, commands, or budgets. "
        "Hidden geometry is inferred and uncertain. Do not claim independent accuracy or a full needle pose. "
        "If no operation is justified, use tool=defer with empty parameters. "
        "Return ONLY a JSON object containing exactly tool (string), parameters (object), "
        "and reason (a concise observable justification, 1-500 characters). "
        "Do not return markdown, additional keys, or a reasoning chain.\n"
        "OBSERVATION:\n" + json.dumps(observation, sort_keys=True) +
        "\nAVAILABLE TOOL SCHEMAS:\n" + json.dumps(tool_schemas(observation.get("available_tools")), sort_keys=True)
    )


def build_messages(observation, prompt):
    content = []
    for side in ("left", "right"):
        content += [{"type": "text", "text": f"Original rectified {side} camera image."},
                    {"type": "image", "url": observation["images"][side]}]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def decode_transport(raw):
    """Remove one known terminal Gemma EOT token, preserving the JSON payload.

    Confirmed special token id 106 in the pinned cached tokenizer, and documented
    at https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4 . This is
    transport decoding, not arbitrary JSON extraction or semantic repair.
    """
    text = raw.rstrip()
    suffix = "<turn|>"
    removed = text.endswith(suffix)
    return (text[:-len(suffix)] if removed else raw), {"removed_terminal_eot": removed,
                                                     "token": suffix if removed else None}


def run_decision(session, infer, model_manifest, model_sources):
    """Persist the real output before executing it; failure never invokes a fallback."""
    out = session.out
    observation = session.observe()
    prompt = make_prompt(observation)
    messages = build_messages(observation, prompt)
    sources = fingerprints(model_sources)
    manifest = {**model_manifest, "source_fingerprints": sources, "prompt": prompt,
                "messages": messages, "observation": observation,
                "input_fingerprints": session.input_fingerprints,
                "tool_schemas": tool_schemas(observation.get("available_tools")), "max_model_calls": 1, "retries": 0,
                "case_selection": "Preselected known development example frame 787, not a held-out benchmark.",
                "api_cost_usd": 0., "compute_cost_usd": None}
    (out / "model_input.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "prompt.txt").write_text(prompt + "\n")
    summary = {"kind": "actual_model_first_tool_checkpoint", "llm_invoked": False,
               "status": "started", "model": model_manifest,
               "independent_accuracy_measured": False, "physics_run": False,
               "api_cost_usd": 0., "compute_cost_usd": None, "retries": 0}
    started = time.perf_counter()
    model_finished = False
    try:
        raw, backend_final = infer(messages)
        model_finished = True
        summary.update(llm_invoked=True, model_runtime_seconds=time.perf_counter() - started)
        (out / "raw_response.txt").write_text(raw + "\n")
        (out / "backend_final_response.txt").write_text(backend_final + "\n")
        session._event("model_response", raw_response=raw, model=model_manifest,
                       runtime_seconds=summary["model_runtime_seconds"], retries=0,
                       api_cost_usd=0., compute_cost_usd=None)
        if fingerprints(model_sources) != sources:
            raise ValueError("Model adapter source changed during inference")
        decoded, transport = decode_transport(raw)
        (out / "decoded_request.txt").write_text(decoded + "\n")
        session._event("model_transport_decoded", **transport, decoded_request=decoded)
        summary["transport_decoding"] = transport
        # No JSON extraction, fence repair, semantic edits or retries.
        try:
            decision = strict_json(decoded)
        except ValueError:
            decision = None
        if isinstance(decision, dict) and decision.get("tool") == "defer":
            if (set(decision) != {"tool", "parameters", "reason"} or decision["parameters"] != {}
                    or not isinstance(decision["reason"], str) or not 1 <= len(decision["reason"].strip()) <= 500):
                raise ValueError("Malformed defer decision")
            summary.update(status="deferred", decision=decision, tool_result=None)
            session._event("model_deferred", decision=decision)
        else:
            # The identical shared validator handles malformed/unsupported model requests.
            result = session.call(decoded)
            summary.update(status=result["status"], decision=decision, tool_result=result)
        if decision is not None:
            (out / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    except Exception as error:
        summary.update(status="model_or_adapter_failed", error=str(error), llm_invoked=model_finished)
        if not model_finished:
            summary["model_runtime_seconds"] = time.perf_counter() - started
        session._event("model_or_adapter_failed", error=str(error), model=model_manifest,
                       generation_completed=model_finished)
    summary["total_runtime_seconds"] = time.perf_counter() - started
    summary["limitations"] = ["One development case, no agent superiority claim.",
                              "Input consistency checks are not independent accuracy.",
                              "Hidden thread joins remain inferred; no dynamics or policy validation."]
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pipeline-tools", type=Path, required=True)
    parser.add_argument("--gemma-runner", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Read-only cached snapshot path")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--environment-label", required=True)
    args = parser.parse_args()
    import torch
    import transformers
    import numpy as np
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Checkpoint requires exactly one explicitly allocated GPU")
    spec = importlib.util.spec_from_file_location("existing_gemma_runner", args.gemma_runner)
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    model_path = Path(args.model)
    model_files = {}
    for path in sorted(model_path.iterdir()):
        if path.is_file():
            model_files[path.name] = {"resolved_path": str(path.resolve()), "bytes": path.stat().st_size,
                                     "sha256": sha256(path) if path.stat().st_size < 20_000_000 else None,
                                     "weight_hash_computed": path.stat().st_size < 20_000_000}
    manifest = {"backend": "local_transformers_gemma", "model_id": "google/gemma-4-E4B-it",
                "cached_snapshot": args.model, "revision": args.model_revision,
                "model_files": model_files, "environment": args.environment_label,
                "torch": torch.__version__, "transformers": transformers.__version__,
                "seed": 0, "do_sample": False, "enable_thinking": False,
                "image_tokens_per_view": 560, "max_new_tokens": 512,
                "gpu_count": torch.cuda.device_count(), "gpu_name": torch.cuda.get_device_name(0),
                "utc": datetime.now(timezone.utc).isoformat()}
    session = ReconstructionSession(args.case, args.out, "agent", "google/gemma-4-E4B-it@" + args.model_revision,
                                    args.pipeline_tools, max_calls=1, timeout_seconds=120)
    summary = run_decision(session, lambda messages: backend.run_model(args.model, messages, 560, 512),
                           manifest, {"model_controller": Path(__file__), "gemma_runner": args.gemma_runner})
    print(json.dumps({key: summary.get(key) for key in ("status", "decision", "error", "model_runtime_seconds", "total_runtime_seconds")}), flush=True)
    if summary["status"] not in {"accepted", "deferred"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
