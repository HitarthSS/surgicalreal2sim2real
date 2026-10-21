#!/usr/bin/env python3
"""Three development cases, one shared model load, at most two decisions per case.

All controllers retain identical reconstruction tools and checks. Model feedback
can propose a bounded operation, finish an accepted thread, or defer. No training.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import random
import time

from reconstruction_tools import (ReconstructionSession, fingerprints, heuristic_request,
                                  parameters_for, sha256, strict_json, tool_schemas)
from run_model_tool_checkpoint import build_messages, decode_transport, make_prompt

FRAMES = (832, 787, 192)
PROMPT_VERSION = "grasp_hold_feedback_v2_backend_errors"


def initial_prompt(observation, target="thread"):
    # The previous checkpoint prompt remains preserved; this pilot records a new
    # version reflecting the user's updated provisional task, applied to every case.
    prompt = make_prompt(observation).replace(
        "future thread grasp and 5 mm pull experiment",
        "future single-arm thread grasp, small verification lift, and two-second hold experiment"
    ) + (
        "\nYour reason must cite concrete visible or numerical evidence relevant to the choice. "
        "Justify any changed parameter using that evidence. If available tools cannot resolve "
        "the ambiguity, defer rather than inventing certainty."
    )
    if target == "needle":
        prompt = prompt.replace("The task is thread reconstruction", "The task is needle arc reconstruction")
        prompt += ("\nThe requested output is the existing free-radius needle arc fit, using needle_arc. "
                   "needle_mask_points can inspect source support but cannot finish this target. "
                   "No fixed size, estimator switch, forced hidden endpoint completion, or fallback is allowed. "
                   "Occluded failures may remain failures or be deferred. A fit is an estimate, not full measured geometry.")
    return prompt


def thread_accepted(result):
    return bool(result and result.get("accepted") and result.get("tool", "").startswith("thread_"))


def target_accepted(result, target):
    if target == "thread":
        return thread_accepted(result)
    if target == "needle":
        return bool(result and result.get("accepted") and result.get("tool") == "needle_arc")
    raise ValueError("Unknown reconstruction target")


def control_decision(decision, last_result, target="thread"):
    """Stop controls cannot turn a failed/non-thread tool into accepted geometry."""
    if not isinstance(decision, dict) or decision.get("tool") not in {"defer", "finish"}:
        return None
    if (set(decision) != {"tool", "parameters", "reason"} or decision["parameters"] != {}
            or not isinstance(decision["reason"], str) or not 1 <= len(decision["reason"].strip()) <= 500):
        raise ValueError("Malformed stop decision")
    if decision["tool"] == "finish" and not target_accepted(last_result, target):
        raise ValueError("finish requires a shared-verified accepted " + target + " result")
    return "accepted" if decision["tool"] == "finish" else "deferred"


def feedback_content(session, result, target="thread"):
    report = result.get("backend_report", {})
    diagnostic = {
        "status": result["status"], "accepted": result["accepted"],
        "tool": result.get("tool"), "effective_parameters": result.get("effective_parameters"),
        "error": result.get("error"), "verification": result.get("verification"),
        "backend_status": report.get("status"), "backend_error": report.get("error"),
        "backend_failures": report.get("failures"), "joins": report.get("joins"),
        "backend_diagnostics": result.get("backend_diagnostics"),
        "candidates": report.get("candidates"),
        "thread_task_completed": thread_accepted(result),
        "requested_target": target, "requested_target_completed": target_accepted(result, target),
    }
    text = (
        "Review the actual tool result and any supplied diagnostic overlays. Green left-overlay "
        "segments have visible support; orange segments are inferred. Right-overlay green/red "
        "indicates predicted-mask agreement/mismatch, with hidden samples excluded from visible-only checks. "
        "These checks are input consistency, not independent accuracy or physics validation. "
        "You have ONE remaining decision and at most ONE additional reconstruction call. "
        "If an accepted thread result is adequate, return tool=finish, parameters={}, and a concise "
        "evidence-based reason. Otherwise select a justified changed operation/parameter or defer. "
        "Do not repeat an identical operation/effective parameters, lower checks, fabricate new observations, "
        "or assert that a failed result passed. Use the same three-field JSON format.\n"
        "ACTUAL TOOL RESULT:\n" + json.dumps(diagnostic, sort_keys=True)
    )
    content = [{"type": "text", "text": text}]
    if target == "needle":
        content[0]["text"] = text.replace("accepted thread result", "accepted needle arc result")
    backend = session.out / result["call_id"] / "backend"
    overlays = (("shared_needle_left_overlay.png", "Left estimated needle arc overlay."),
                ("shared_needle_right_overlay.png", "Right needle arc mask diagnostic.")) if target == "needle" else (
                    ("shared_left_overlay.png", "Left estimated geometry overlay."),
                    ("right_validation_overlay.png", "Right predicted-mask diagnostic."))
    for name, label in overlays:
        path = backend / name
        if path.is_file():
            content.extend([{"type": "text", "text": label}, {"type": "image", "url": str(path)}])
    return content


def assert_unchanged(session, model_sources, expected_sources):
    if fingerprints(model_sources) != expected_sources:
        raise ValueError("Controller/model adapter source changed")
    if fingerprints({**session.inputs, "case_manifest": str(session.case_path)}) != session.input_fingerprints:
        raise ValueError("Case inputs changed")
    if fingerprints(session.source_paths) != session.source_fingerprints:
        raise ValueError("Shared reconstruction source changed")


def run_loop(session, infer, model_manifest, model_sources, target="thread"):
    target_accepted(None, target)
    sources = fingerprints(model_sources)
    observation = session.observe()
    messages = build_messages(observation, initial_prompt(observation, target))
    rounds, last_result, seen = [], None, set()
    attempted, completed = 0, 0
    summary = {"kind": "bounded_actual_agent_development_pilot", "prompt_version": PROMPT_VERSION,
               "model": model_manifest, "max_model_calls": 2, "max_tool_calls": 2,
               "retries": 0, "rounds": rounds, "api_cost_usd": 0., "compute_cost_usd": None,
               "independent_accuracy_measured": False, "physics_run": False, "requested_target": target}
    if target == "needle":
        summary["prompt_version"] = "existing_free_radius_needle_feedback_v1"
    start = time.perf_counter()
    status = "started"
    try:
        for index in (1, 2):
            assert_unchanged(session, model_sources, sources)
            folder = session.out / f"model_round_{index:03d}"
            folder.mkdir()
            image_paths = {f"image_{i}_{j}": part["url"] for i, message in enumerate(messages)
                           for j, part in enumerate(message["content"]) if part["type"] == "image"}
            image_hashes = fingerprints(image_paths)
            manifest = {"messages": messages, "model": model_manifest, "source_fingerprints": sources,
                        "image_fingerprints": image_hashes, "input_fingerprints": session.input_fingerprints,
                        "observation": observation, "tool_schemas": tool_schemas(observation.get("available_tools")),
                        "prompt_version": PROMPT_VERSION, "round": index}
            (folder / "input.json").write_text(json.dumps(manifest, indent=2) + "\n")
            before = time.perf_counter()
            attempted += 1
            raw = infer(messages)
            completed += 1
            runtime = time.perf_counter() - before
            (folder / "raw_response.txt").write_text(raw + "\n")
            assert_unchanged(session, model_sources, sources)
            if fingerprints(image_paths) != image_hashes:
                raise ValueError("An observed overlay/image changed during model inference")
            decoded, transport = decode_transport(raw)
            (folder / "decoded_request.txt").write_text(decoded + "\n")
            try:
                decision = strict_json(decoded)
            except ValueError:
                decision = None
            row = {"round": index, "model_runtime_seconds": runtime, "decision": decision,
                   "transport": transport, "tool_result": None}
            rounds.append(row)
            session._event("model_decision", **row, raw_response=raw, decoded_request=decoded,
                           model=model_manifest, observed_image_fingerprints=image_hashes)
            control = control_decision(decision, last_result, target)
            if control:
                status = control
                session._event("agent_stopped", status=status, decision=decision)
                break
            signature = None
            if isinstance(decision, dict) and isinstance(decision.get("tool"), str):
                try:
                    effective = parameters_for(decision["tool"], decision.get("parameters"))
                    # JSON integers and floats can denote the same effective setting.
                    signature = json.dumps([decision["tool"], {k: float(v) for k, v in effective.items()}], sort_keys=True)
                except ValueError:
                    pass  # Shared interface logs the invalid request and consumes its budget.
            if signature is not None and signature in seen:
                status = "stopped_duplicate_request"
                session._event("agent_stopped", status=status, decision=decision,
                               reason="Identical operation/effective parameters already attempted; no rerun.")
                break
            if signature is not None:
                seen.add(signature)
            last_result = session.call(decoded)
            row["tool_result"] = last_result
            (folder / "round_result.json").write_text(json.dumps(row, indent=2) + "\n")
            status = "accepted_at_budget" if target_accepted(last_result, target) else "failed_at_budget"
            if index == 1:
                messages = [*messages, {"role": "assistant", "content": [{"type": "text", "text": decoded}]},
                            {"role": "user", "content": feedback_content(session, last_result, target)}]
    except Exception as error:
        status = "model_or_controller_failed"
        summary["error"] = str(error)
        session._event("agent_failed", error=str(error))
    summary.update(status=status, last_tool_result=last_result,
                   scene_accepted=status in {"accepted", "accepted_at_budget"} and thread_accepted(last_result),
                   requested_target_accepted=status in {"accepted", "accepted_at_budget"} and target_accepted(last_result, target),
                   model_calls_attempted=attempted, model_calls_completed=completed,
                   decisions_logged=len(rounds), tool_calls=session.index,
                   runtime_seconds=time.perf_counter() - start)
    (session.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


class CachedGemma:
    """Load once, reuse the project's existing feature patch and response decoder."""
    def __init__(self, model, runner):
        import torch
        from transformers import GenerationConfig, pipeline
        spec = importlib.util.spec_from_file_location("existing_gemma_runner", runner)
        self.backend = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.backend)
        self.backend.disable_python_native_bmm(torch)
        self.pipe = pipeline(task="image-text-to-text", model=model, device_map="auto", dtype="auto",
                             model_kwargs={"attn_implementation": "sdpa", "low_cpu_mem_usage": True})
        self.backend.normalize_gemma4_image_feature_output(self.pipe.model)
        self.pipe.image_processor.max_soft_tokens = 560
        self.generation = GenerationConfig.from_pretrained(model)
        self.generation.max_new_tokens = 512
        self.generation.do_sample = False
        self.generation.temperature = self.generation.top_k = self.generation.top_p = None

    def __call__(self, messages):
        output = self.pipe(text=messages, return_full_text=False, enable_thinking=False,
                           processor_kwargs={"truncation": False},
                           generate_kwargs={"generation_config": self.generation})
        return self.backend.pipeline_response_text(output)


def preflight_cases(cases, out, pipeline_tools, target):
    """Validate every case and shared source without loading or invoking a model."""
    rows = []
    for index, case in enumerate(cases):
        session = ReconstructionSession(case, Path(out) / f'case_{index:03d}',
                                        'fixed', 'preflight_no_model', pipeline_tools,
                                        max_calls=2)
        required = {target + '_left_mask', target + '_right_mask'}
        if not required <= set(session.inputs):
            raise ValueError('Missing stereo masks for requested target: ' + target)
        rows.append({'case_id': session.case['case_id'],
                     'input_fingerprints': session.input_fingerprints,
                     'source_fingerprints': session.source_fingerprints,
                     'tool_calls': 0, 'model_loaded': False})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pipeline-tools", type=Path, required=True)
    parser.add_argument("--gemma-runner", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--environment-label", required=True)
    parser.add_argument("--target", choices=("thread", "needle"), default="thread")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    preflight = preflight_cases([args.cases / f"frame_{frame:06d}.json" for frame in FRAMES],
                                args.out / 'preflight', args.pipeline_tools, args.target)
    (args.out / 'preflight.json').write_text(json.dumps(preflight, indent=2) + '\n')
    if args.preflight_only:
        print(json.dumps({'preflight_passed': True, 'case_count': len(preflight),
                          'model_loaded': False}), flush=True)
        return
    import torch
    import numpy as np
    import transformers
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Allocate exactly one idle GPU")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    model_manifest = {"model_id": "google/gemma-4-E4B-it", "revision": args.model_revision,
                      "seed": 0, "do_sample": False, "enable_thinking": False,
                      "image_tokens_per_view": 560, "max_new_tokens": 512,
                      "environment": args.environment_label, "torch": torch.__version__,
                      "transformers": transformers.__version__, "gpu_name": torch.cuda.get_device_name(0)}
    files = {}
    for path in sorted(Path(args.model).iterdir()):
        if path.is_file():
            files[path.name] = {"bytes": path.stat().st_size, "resolved_path": str(path.resolve()),
                                "sha256": sha256(path) if path.stat().st_size < 20_000_000 else None}
    model_manifest["cached_model_files"] = files
    start = time.perf_counter()
    infer = CachedGemma(args.model, args.gemma_runner)
    load_seconds = time.perf_counter() - start
    rows = []
    for frame in FRAMES:
        case = args.cases / f"frame_{frame:06d}.json"
        results = {}
        for controller in ("fixed", "heuristic"):
            session = ReconstructionSession(case, args.out / f"frame_{frame:06d}" / controller,
                                            controller, "distance_defaults_reference" if controller == "fixed" else "initial_untuned_rule",
                                            args.pipeline_tools, max_calls=2)
            request = ({"tool": "needle_arc", "parameters": {}, "reason": "Existing free-radius defaults reference."}
                       if args.target == "needle" else
                       {"tool": "thread_distance", "parameters": {}, "reason": "Fixed distance defaults reference."}
                       if controller == "fixed" else heuristic_request(session.observe()))
            result = session.call(request)
            results[controller] = {"status": result["status"], "accepted": result["accepted"],
                                   "request": request, "verification": result["verification"],
                                   "tool_calls": 1, "runtime_seconds": result["runtime_seconds"]}
        agent = ReconstructionSession(case_path=case, out=args.out / f"frame_{frame:06d}" / "agent",
                                      controller="agent", controller_backend="google/gemma-4-E4B-it@" + args.model_revision,
                                      pipeline_tools=args.pipeline_tools, max_calls=2)
        result = run_loop(agent, infer, model_manifest, {
            "loop": Path(__file__), "transport_adapter": Path(__file__).with_name("run_model_tool_checkpoint.py"),
            "gemma_runner": args.gemma_runner}, target=args.target)
        results["agent"] = result
        row = {"frame": frame, "results": results}
        rows.append(row)
        print(json.dumps({"frame": frame, "agent_status": result["status"],
                          "accepted": result["scene_accepted"], "decisions": [r["decision"] for r in result["rounds"]]}), flush=True)
    summary = {"kind": "three_case_development_feedback_pilot", "rows": rows,
               "prompt_version": PROMPT_VERSION, "model_load_seconds": load_seconds,
               "total_runtime_seconds": time.perf_counter() - start, "utc": datetime.now(timezone.utc).isoformat(),
               "independent_accuracy_measured": False, "physics_run": False, "api_cost_usd": 0., "compute_cost_usd": None,
               "limitations": ["Three known cases from one development recording; no held-out accuracy.",
                               "Agent gets up to two decisions; fixed/initial heuristic each use one call. Report used budgets; no superiority claim.",
                               "Distance reference is not the production QP-spline baseline.",
                               "All stereo/mask checks use predicted inputs; no independent geometry or physics validation."]}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
