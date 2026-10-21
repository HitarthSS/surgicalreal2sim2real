"""Isolated compact controller using the unchanged shared reconstruction session."""
import argparse
import json
from pathlib import Path
import time

from compact_agent_protocol import (PROTOCOL, accepted_target, make_state, messages_for,
                                    shared_request, signature, token_usage, validate_decision)
from reconstruction_tools import ReconstructionSession, fingerprints
from run_bounded_agent_pilot import CachedGemma, assert_unchanged
from run_model_tool_checkpoint import decode_transport


class MeteredGemma(CachedGemma):
    """Observe real generate tensors once; never re-tokenize for usage estimates."""
    def __call__(self, messages):
        model = self.pipe.model
        original = model.generate
        usages = []
        def observed(*args, **kwargs):
            ids = kwargs.get("input_ids", args[0] if args else None)
            output = original(*args, **kwargs)
            sequence = getattr(output, "sequences", output)
            image_id = getattr(model.config, "image_token_id", None)
            if image_id is None:
                image_id = getattr(model.config, "image_token_index", None)
            usages.append(token_usage(ids, sequence, image_id))
            return output
        model.generate = observed
        try:
            response = super().__call__(messages)
        finally:
            model.generate = original
        if len(usages) != 1:
            raise ValueError("Expected exactly one measured generation")
        self.last_usage = usages[0]
        return response


def run_compact_loop(session, infer, model_manifest, initial_result=None, inference_kind="actual_model"):
    """Two model decisions total; common baseline consumes the shared tool budget.

    ``initial_result`` must be the result just executed in this session, never an
    unrelated cached verdict. The CLI validates/executes the baseline before
    loading the model. Ambiguous failed cases remain failures at exhaustion.
    """
    if inference_kind not in {"actual_model", "scripted_contract_test"}:
        raise ValueError("Unknown inference kind")
    if initial_result is not None:
        path = session.out / initial_result["call_id"] / "tool_result.json"
        if not path.is_file() or json.loads(path.read_text()) != initial_result:
            raise ValueError("Initial result is not the recorded result of this session")
    target = model_manifest["target"]
    source_paths = {"protocol": Path(__file__).with_name("compact_agent_protocol.py"),
                    "runner": Path(__file__),
                    "existing_model_adapter": Path(__file__).with_name("run_bounded_agent_pilot.py"),
                    "transport_adapter": Path(__file__).with_name("run_model_tool_checkpoint.py")}
    if model_manifest.get("gemma_runner"):
        source_paths["gemma_runner"] = Path(model_manifest["gemma_runner"])
    expected = fingerprints(source_paths)
    provenance = {"inputs": session.input_fingerprints, "sources": session.source_fingerprints,
                  "controller_sources": expected, "case": session.case}
    result, rounds, seen, attempted = initial_result, [], set(), []
    if result and result.get("tool"):
        seen.add(signature({"operation": result["tool"], "parameters": result["effective_parameters"]}))
        attempted.append({"operation": result["tool"], "parameters": result["effective_parameters"],
                          "status": result["status"], "kind": "common_baseline"})
    start, evidence_reads, requested = time.perf_counter(), 0, None
    summary = {"protocol": PROTOCOL, "model": model_manifest, "rounds": rounds,
               "max_model_calls": 2, "max_total_tool_calls": session.max_calls, "max_evidence_reads": 1,
               "model_calls_attempted": 0, "model_calls_completed": 0,
               "api_cost_usd": 0., "compute_cost_usd": None, "cached_model_tokens": None,
               "common_baseline_included_in_tool_budget": initial_result is not None,
               "status": "started", "actual_model_used": False, "inference_kind": inference_kind}
    try:
        if accepted_target(result, target):
            summary["status"] = "routine_target_pass_no_model"
        else:
            for index in range(2):
                assert_unchanged(session, source_paths, expected)
                state, catalog = make_state(session.observe(), provenance, target,
                    {"model_decisions": 2-index, "tool_calls": session.max_calls-session.index,
                     "evidence_reads": 1-evidence_reads}, result, index, attempted)
                images = [(f"Original rectified {side} image; pixel origin (0,0).", session.inputs[side])
                          for side in ("left", "right")]
                if result:
                    names = ("shared_needle_left_overlay.png", "shared_needle_right_overlay.png") if target == "needle" else (
                        "shared_left_overlay.png", "right_validation_overlay.png")
                    for name in names:
                        p = session.out / result["call_id"] / "backend" / name
                        if p.exists():
                            images.append((name + "; source support, not independent accuracy.", p))
                messages = messages_for(state, images, requested)
                folder = session.out / f"compact_round_{index+1:03d}"
                folder.mkdir()
                image_hashes = fingerprints({str(i): path for i, (_, path) in enumerate(images)})
                for filename, value in (("state.json", state), ("evidence_catalog.json", catalog),
                                        ("input.json", {"messages": messages, "image_hashes": image_hashes,
                                                        "provenance": provenance, "model": model_manifest})):
                    (folder / filename).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
                before = time.perf_counter()
                summary["model_calls_attempted"] += 1
                raw = infer(messages)
                summary["actual_model_used"] = inference_kind == "actual_model"
                summary["model_calls_completed"] += 1
                (folder / "raw_response.txt").write_text(raw + "\n")
                assert_unchanged(session, source_paths, expected)
                if fingerprints({str(i): path for i, (_, path) in enumerate(images)}) != image_hashes:
                    raise ValueError("Observed image changed during inference")
                decoded, transport = decode_transport(raw)
                row = {"round": index+1, "runtime_seconds": time.perf_counter()-before,
                       "usage": getattr(infer, "last_usage", None), "transport": transport,
                       "scene_version": state["scene_version"]}
                rounds.append(row)
                decision = validate_decision(decoded, state, catalog)
                row["decision"] = decision
                session._event("compact_model_decision", **row)
                op = decision["operation"]
                if op in {"finish", "defer"}:
                    summary["status"] = "target_finished" if op == "finish" else "deferred"
                elif op == "inspect_evidence":
                    evidence_reads += 1
                    requested = {key: catalog[key] for key in decision["evidence_ids"]}
                    row["evidence_returned"] = list(requested)
                    summary["status"] = "evidence_requested_at_budget"
                else:
                    sig = signature(decision)
                    if sig in seen:
                        summary["status"] = "stopped_duplicate_request"
                    else:
                        seen.add(sig)
                        result = session.call(shared_request(decision))
                        attempted.append({"operation": op, "parameters": result.get("effective_parameters"),
                                          "status": result["status"], "kind": "model_selected"})
                        row["tool_result"] = result
                        requested = None
                        summary["status"] = "target_passed_at_budget" if accepted_target(result, target) else "failed_at_budget"
                (folder / "result.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
                if op in {"finish", "defer"} or summary["status"] == "stopped_duplicate_request":
                    break
    except Exception as error:
        summary.update(status="controller_or_model_failed", error=str(error))
        session._event("compact_agent_failed", error=str(error))
    state, catalog = make_state(session.observe(), provenance, target,
        {"model_decisions": 2-summary["model_calls_attempted"], "tool_calls": session.max_calls-session.index,
         "evidence_reads": 1-evidence_reads}, result, len(rounds), attempted)
    summary.update(validity=state["validity"], last_tool_result=result, tool_calls=session.index,
                   runtime_seconds=time.perf_counter()-start, evidence_reads=evidence_reads)
    (session.out / "compact_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case", "out", "pipeline-tools", "gemma-runner"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--original-src", type=Path)
    parser.add_argument("--target", choices=("thread", "needle"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    args = parser.parse_args()
    session = ReconstructionSession(args.case, args.out, "agent", "compact_gemma_v1", args.pipeline_tools,
                                    original_src=args.original_src, max_calls=2)
    if args.target == "thread" and args.original_src is None:
        raise ValueError("Production thread baseline requires original-src")
    common = session.call({"tool": "thread_qp" if args.target == "thread" else "needle_arc",
                           "parameters": {}, "reason": "Common default baseline before any model refinement."})
    # Laziness avoids GPU/model loading for routine verified passes.
    class LazyModel:
        last_usage = None
        def __call__(self, messages):
            if not hasattr(self, "model"):
                self.model = MeteredGemma(args.model, args.gemma_runner)
            response = self.model(messages)
            self.last_usage = self.model.last_usage
            return response
    summary = run_compact_loop(session, LazyModel(), {"target": args.target, "name": args.model,
                              "revision": args.model_revision, "sampling": False,
                              "gemma_runner": str(args.gemma_runner)}, common)
    print(json.dumps({"status": summary["status"], "validity": summary["validity"],
                      "model_calls": summary["model_calls_completed"]}))


if __name__ == "__main__":
    main()
