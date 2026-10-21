#!/usr/bin/env python3
"""Small integration smoke test on saved development frames; no LLM or physics."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from reconstruction_tools import ReconstructionSession, heuristic_request, LIMITS, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--K", type=Path, required=True)
    parser.add_argument("--pipeline-tools", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--environment-label", required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cases = args.out / "cases"
    cases.mkdir()
    rows, parity = [], []
    start = time.perf_counter()
    for frame in (64, 192, 787, 832, 1088):
        run = args.input_root / f"hitarth_frame_{frame:06d}"
        case = {
            "case_id": f"frame_{frame:06d}", "session_id": "dvrk_20260902_130146",
            "split": "development", "calibration_id": "saved_stereo_K_baseline_20261006",
            "baseline_m": .006820275085,
            "manual_interventions": "Reuses saved SAM3 masks and reviewed tool mask; no new selections. Manual time unavailable.",
            "inputs": {
                "left": str(run / "frames/episode_0000/colors/left_image_000000.jpg"),
                "right": str(run / "frames/episode_0000/colors/right_image_000000.jpg"),
                "depth_npy": str(run / "stereo/fast_foundationstereo/depth_meter.npy"),
                "K": str(args.K),
                "thread_left_mask": str(run / "thread/sam3_text/left_thread_mask.png"),
                "thread_right_mask": str(run / "thread/sam3_text/right_thread_mask.png"),
                "needle_left_mask": str(run / "thread/sam3_text/left_needle_mask.png"),
                "needle_right_mask": str(run / "thread/sam3_text/right_needle_mask.png"),
                "left_occluder": str(run / "robot/sam3_tool/tool_mask.png"),
            },
        }
        case_path = cases / f"frame_{frame:06d}.json"
        case_path.write_text(json.dumps(case, indent=2) + "\n")
        frame_results = {}
        for controller in ("fixed", "heuristic", "agent"):
            backend = "recorded_request_not_llm" if controller == "agent" else "fixed_defaults" if controller == "fixed" else "initial_untuned_rule"
            session = ReconstructionSession(case_path, args.out / f"frame_{frame:06d}" / controller,
                                            controller, backend, args.pipeline_tools, max_calls=4)
            selection = {"tool": "thread_distance", "parameters": {}, "reason": "Fixed defaults reference."}
            if controller != "fixed":
                selection = heuristic_request(session.observe())
            if controller == "agent":
                selection = {**selection, "reason": "Recorded request replay to verify shared adapter; no model decision."}
                result = session.call(json.dumps(selection))
            else:
                result = session.call(selection)
            row = {"frame": frame, "controller": controller, "backend": backend,
                   "operation": result.get("tool"), "status": result["status"], "accepted": result["accepted"],
                   "verification": result["verification"], "error": result.get("error"),
                   "runtime_seconds": result["runtime_seconds"], "result": str(session.out / "call_001/tool_result.json"),
                   "kind": "thread_interface_smoke_test"}
            rows.append(row)
            frame_results[controller] = (session, result)
            print(json.dumps(row), flush=True)
            if frame == 787:
                needle = session.call({"tool": "needle_mask_points", "parameters": {},
                                       "reason": "Smoke-test existing needle point extraction, without estimating full needle pose."})
                needle_row = {"frame": frame, "controller": controller, "backend": backend,
                              "operation": "needle_mask_points", "status": needle["status"], "accepted": needle["accepted"],
                              "verification": needle["verification"], "error": needle.get("error"),
                              "runtime_seconds": needle["runtime_seconds"],
                              "result": str(session.out / "call_002/tool_result.json"), "kind": "needle_interface_smoke_test"}
                rows.append(needle_row)
                print(json.dumps(needle_row), flush=True)
        h_session, h_result = frame_results["heuristic"]
        a_session, a_result = frame_results["agent"]
        same_geometry = None
        h_npz = h_session.out / "call_001/backend/thread_samples.npz"
        a_npz = a_session.out / "call_001/backend/thread_samples.npz"
        if h_npz.exists() and a_npz.exists():
            with np.load(h_npz, allow_pickle=False) as h, np.load(a_npz, allow_pickle=False) as a:
                same_geometry = np.array_equal(h["points_m"], a["points_m"])
        parity.append({"frame": frame, "same_acceptance": h_result["accepted"] == a_result["accepted"],
                       "same_verification": h_result["verification"] == a_result["verification"],
                       "same_geometry": same_geometry})
    summary = {
        "kind": "shared_tool_interface_smoke_test", "independent_accuracy_measured": False,
        "llm_invoked": False, "heuristic_is_tuned": False, "physics_run": False,
        "environment": args.environment_label, "shared_limits": LIMITS,
        "pilot_script_sha256": sha256(__file__), "runtime_seconds": time.perf_counter() - start,
        "rows": rows, "adapter_parity": parity,
        "limitations": ["Five frames from one development recording, not held-out evaluation.",
                        "Agent calls replay heuristic requests; not an agent-performance comparison.",
                        "Acceptance checks measure input consistency, not independently measured accuracy.",
                        "No complete surgical scene or dynamics validation in this smoke test."],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
