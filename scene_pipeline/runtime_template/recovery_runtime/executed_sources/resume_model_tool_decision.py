#!/usr/bin/env python3
"""Execute the first real model response after a transport-only adapter repair.

No model inference occurs here. Original inference artifacts remain unchanged;
this stage verifies identical inputs/tools and records its link to that call.
"""
import argparse
import json
from pathlib import Path
import time

from reconstruction_tools import ReconstructionSession, sha256, strict_json, tool_schemas
from run_model_tool_checkpoint import decode_transport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-session", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pipeline-tools", type=Path, required=True)
    args = parser.parse_args()
    prior = args.original_session
    original = strict_json((prior / "model_input.json").read_text())
    old_summary = strict_json((prior / "summary.json").read_text())
    if not old_summary.get("llm_invoked") or old_summary["status"] != "rejected":
        raise ValueError("Expected preserved real-model transport rejection")
    session = ReconstructionSession(args.case, args.out, "agent",
                                    "google/gemma-4-E4B-it@" + original["revision"] + ":transport_resume",
                                    args.pipeline_tools, max_calls=1)
    # Paths in copied case/container mounts are identical, as are all content hashes.
    if session.input_fingerprints != original["input_fingerprints"]:
        raise ValueError("Original model input provenance no longer matches")
    old_event = strict_json((prior / "events.jsonl").read_text().splitlines()[0])
    if session.source_fingerprints != old_event["source_fingerprints"]:
        raise ValueError("Shared tool source changed since the real model call")
    if tool_schemas() != original["tool_schemas"] or session.observe() != original["observation"]:
        raise ValueError("Observation or tool definitions changed")
    raw = (prior / "raw_response.txt").read_text().removesuffix("\n")
    decoded, transport = decode_transport(raw)
    provenance = {"kind": "transport_resume_of_actual_model_decision", "new_model_calls": 0,
                  "llm_invoked_this_stage": False, "actual_model_generation_completed": True,
                  "original_session": str(prior), "model_id": original["model_id"],
                  "revision": original["revision"], "transport_decoding": transport,
                  "original_raw_sha256": sha256(prior / "raw_response.txt"),
                  "original_model_input_sha256": sha256(prior / "model_input.json"),
                  "original_model_runtime_seconds": old_summary["model_runtime_seconds"],
                  "adapter_sha256": sha256(Path(__file__).with_name("run_model_tool_checkpoint.py")),
                  "resume_source_sha256": sha256(__file__),
                  "api_cost_usd": 0., "compute_cost_usd": None,
                  "independent_accuracy_measured": False, "physics_run": False}
    (session.out / "raw_response.txt").write_text(raw + "\n")
    (session.out / "decoded_request.txt").write_text(decoded + "\n")
    session._event("actual_model_decision_resumed", **provenance, decoded_request=decoded)
    start = time.perf_counter()
    result = session.call(decoded)
    summary = {**provenance, "status": result["status"], "tool_result": result,
               "execution_runtime_seconds": time.perf_counter() - start,
               "decision": strict_json(decoded)}
    (session.out / "decision.json").write_text(json.dumps(summary["decision"], indent=2) + "\n")
    (session.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "decision": summary["decision"],
                      "verification": result["verification"]}), flush=True)
    if not result["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
