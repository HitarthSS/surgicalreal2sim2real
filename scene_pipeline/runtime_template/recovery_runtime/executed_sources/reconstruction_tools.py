#!/usr/bin/env python3
"""Bounded, logged reconstruction operations shared by heuristic and agent callers."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import time
from types import MappingProxyType

from reconstruction_verification import inspect_case, verify_needle, verify_thread, verify_native_qp
from needle_arc_verification import verify_needle_arc

def default_pipeline_tools(code_path):
    """Find the checkout when available; allow explicit mounts at any depth."""
    path = Path(code_path).resolve()
    for parent in path.parents:
        candidate = parent / 'integrated_surgical_scene_pipeline' / 'tools'
        if candidate.is_dir():
            return candidate
    return path.parent / 'pipeline_tools'


DEFAULT_TOOLS = default_pipeline_tools(__file__)
LIMITS = {
    "thread_samples": 200, "needle_samples": 96, "segment_min_skeleton_px": 30,
    "min_depth_m": .03, "max_depth_m": .5, "max_join_gap_m": .04,
    "max_depth_relative_difference": .25, "right_tolerance_px": 8.,
    "min_visible_coverage": .85, "min_valid_visible_depth": .7,
    "min_visible_right_fraction": .5,
}
TOOLS = {
    "thread_qp": {
        "description": "Production SAM3/FFS original ordering and QP spline, unchanged. Shared checks reject missing fragments or undocumented hidden joins; native confidence is not measured accuracy.",
        "script": "thread_qp_tool.py", "target": "thread",
        "defaults": {"connection_radius_px": 80.},
        "bounds": {"connection_radius_px": ("number", 1., 240.)},
        "flags": {"connection_radius_px": "--segment-connection-radius-px"},
    },
    "thread_distance": {
        "description": "Existing distance-only connector and stereo backprojection; no fallback.",
        "script": "reconstruct_thread_from_files.py", "target": "thread",
        "defaults": {"connection_radius_px": 80.},
        "bounds": {"connection_radius_px": ("number", 0., 240.)},
        "flags": {"connection_radius_px": "--segment-connection-radius-px"},
    },
    "thread_occlusion": {
        "description": "Existing occlusion/stereo/tangent-aware joining; rejects unresolved fragments.",
        "script": "reconstruct_thread_occlusion_aware.py", "target": "thread",
        "defaults": {"max_gap_px": 240., "max_gap_m": .04, "min_occluder_fraction": .55,
                     "min_direction_cosine": .25, "ambiguity_margin": .1},
        "bounds": {"max_gap_px": ("number", 1., 240.), "max_gap_m": ("number", .001, .04),
                   "min_occluder_fraction": ("number", .55, 1.),
                   "min_direction_cosine": ("number", .25, 1.),
                   "ambiguity_margin": ("number", .1, .5)},
        "flags": {"max_gap_px": "--max-gap-px", "max_gap_m": "--max-gap-m",
                  "min_occluder_fraction": "--min-occluder-fraction",
                  "min_direction_cosine": "--min-direction-cosine", "ambiguity_margin": "--ambiguity-margin"},
    },
    "needle_mask_points": {
        "description": "Existing stereo needle-mask backprojection to estimated 3D points; not pose/completion.",
        "script": "reconstruct_needle_from_masks.py", "target": "needle",
        "defaults": {"depth_sample_radius_px": 3},
        "bounds": {"depth_sample_radius_px": ("integer", 1, 5)},
        "flags": {"depth_sample_radius_px": "--depth-sample-radius"},
    },
    "needle_arc": {
        "description": "Existing stereo point extraction followed by existing free-radius RANSAC arc fit. No size prior, estimator switch, endpoint extension or fallback; failed checks reject the fit.",
        "script": "needle_arc_tool.py", "target": "needle",
        "defaults": {"depth_sample_radius_px": 3, "ransac_threshold_m": .002,
                     "ransac_min_inlier_fraction": 2/3},
        "bounds": {"depth_sample_radius_px": ("integer", 1, 5),
                   "ransac_threshold_m": ("number", .0003, .002),
                   "ransac_min_inlier_fraction": ("number", 2/3, 1.)},
        "flags": {"depth_sample_radius_px": "--depth-sample-radius",
                  "ransac_threshold_m": "--ransac-threshold-m",
                  "ransac_min_inlier_fraction": "--ransac-min-inlier-fraction"},
    },
}
CASE_KEYS = {"case_id", "session_id", "split", "calibration_id", "baseline_m", "inputs",
             "manual_interventions"}
INPUT_KEYS = {"left", "right", "depth_npy", "K", "thread_left_mask", "thread_right_mask",
              "needle_left_mask", "needle_right_mask", "left_occluder", "right_occluder"}
CORE_INPUTS = {"left", "right", "depth_npy", "K"}


def strict_json(text):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("Duplicate JSON key: " + key)
            out[key] = value
        return out
    def constant(value):
        raise ValueError("Nonfinite JSON value: " + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprints(paths):
    return {name: {"path": str(path), "sha256": sha256(path), "bytes": Path(path).stat().st_size}
            for name, path in paths.items()}


def scrub(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "invalid_nonfinite_value"
    if isinstance(value, dict):
        return {str(k): scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(v) for v in value]
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


def native_qp_diagnostics(stdout):
    """Read only the existing structured stop message, never execute log text.

    Distances are the rounded values reported by native ordering. This is a
    diagnostic of its decisions, not measured geometry or a join authorization.
    """
    pattern = (r"^cold assembly: stopping with (\d+) segment\(s\) \((\d+) keypoints\) "
               r"unjoined — nearest gap ([0-9]+(?:\.[0-9]+)?)px > "
               r"([0-9]+(?:\.[0-9]+)?)px cap; kept a chain of (\d+)$")
    stops = []
    for line in stdout.splitlines():
        match = re.fullmatch(pattern, line)
        if match:
            segments, points, gap, cap, kept = match.groups()
            stops.append(dict(unjoined_segment_count=int(segments), unjoined_keypoint_count=int(points),
                              nearest_gap_px_reported=float(gap), join_cap_px_reported=float(cap),
                              retained_keypoint_count=int(kept)))
    return dict(native_ordering_stops=stops, stop_diagnostic_available=bool(stops),
                origin='native_stdout_recognized_stop_messages', values_are_native_rounded_reports=True,
                join_provenance_available=False, independent_accuracy_measured=False)


def tool_schemas(names=None):
    result = []
    for name, tool in TOOLS.items():
        if names is not None and name not in names:
            continue
        properties = {key: {"type": kind, "minimum": low, "maximum": high, "default": tool["defaults"][key]}
                      for key, (kind, low, high) in tool["bounds"].items()}
        result.append({"name": name, "description": tool["description"],
                       "parameters": {"type": "object", "properties": properties, "additionalProperties": False}})
    return result


def parameters_for(name, supplied):
    if name not in TOOLS:
        raise ValueError("Unknown tool")
    if not isinstance(supplied, dict):
        raise ValueError("parameters must be an object")
    tool = TOOLS[name]
    extra = set(supplied) - set(tool["defaults"])
    if extra:
        raise ValueError("Unknown or protected parameters: " + ", ".join(sorted(extra)))
    result = dict(tool["defaults"])
    for key, value in supplied.items():
        kind, low, high = tool["bounds"][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(key + " must be finite numeric data")
        if kind == "integer" and not isinstance(value, int):
            raise ValueError(key + " must be an integer")
        if not low <= value <= high:
            raise ValueError(key + " outside allowed bounds")
        result[key] = value
    return result


class ReconstructionSession:
    """Controller labels affect provenance only, never tools, limits or checks."""
    def __init__(self, case_path, out, controller, controller_backend="unspecified",
                 pipeline_tools=DEFAULT_TOOLS, max_calls=4, timeout_seconds=120, purpose="development",
                 original_src=None):
        if controller not in {"fixed", "heuristic", "agent"}:
            raise ValueError("Unknown controller")
        if purpose not in {"development", "evaluation"}:
            raise ValueError("Unknown purpose")
        if isinstance(max_calls, bool) or not isinstance(max_calls, int) or not 1 <= max_calls <= 16:
            raise ValueError("max_calls must be an integer from 1 to 16")
        if not math.isfinite(timeout_seconds) or not 1 <= timeout_seconds <= 600:
            raise ValueError("timeout must be between 1 and 600 seconds")
        self.case_path = Path(case_path).resolve()
        self.case = strict_json(self.case_path.read_text())
        if not isinstance(self.case, dict) or set(self.case) - CASE_KEYS:
            raise ValueError("Unknown case fields")
        for key in ("case_id", "session_id", "calibration_id"):
            if not isinstance(self.case.get(key), str) or not self.case[key].strip():
                raise ValueError("Missing " + key)
        split = self.case.get("split")
        if split not in {"development", "train", "validation", "test"}:
            raise ValueError("Missing or invalid session split")
        if split == "test" and purpose != "evaluation":
            raise ValueError("Held-out test cases are forbidden in development mode")
        baseline = self.case.get("baseline_m")
        if isinstance(baseline, bool) or not isinstance(baseline, (int, float)) or not math.isfinite(baseline) or not .001 <= baseline <= .1:
            raise ValueError("Invalid stereo baseline")
        provided = self.case.get("inputs", {})
        if not isinstance(provided, dict) or set(provided) - INPUT_KEYS or not CORE_INPUTS <= set(provided):
            raise ValueError("Invalid case inputs")
        self.inputs = {}
        for key, value in provided.items():
            if not isinstance(value, str) or not value:
                raise ValueError("Input paths must be nonempty strings")
            path = Path(value)
            path = (self.case_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            self.inputs[key] = str(path)
        for prefix in ("thread", "needle"):
            keys = {prefix + "_left_mask", prefix + "_right_mask"}
            if set(self.inputs) & keys and not keys <= set(self.inputs):
                raise ValueError("Both stereo masks are required for " + prefix)
        if not {"thread_left_mask", "needle_left_mask"} & set(self.inputs):
            raise ValueError("No reconstructable object masks supplied")
        self.pipeline_tools = Path(pipeline_tools).resolve()
        self.original_src = Path(original_src).resolve() if original_src is not None else None
        self.source_paths = {
            "interface": Path(__file__).resolve(),
            "verification": Path(__file__).with_name("reconstruction_verification.py").resolve(),
            "base_thread": self.pipeline_tools / "reconstruct_thread_from_files.py",
            "occlusion_helper": self.pipeline_tools / "occlusion_aware_thread_joining.py",
            **{name: (Path(__file__).with_name(tool["script"]) if name in {"needle_arc", "thread_qp"}
                       else self.pipeline_tools / tool["script"]) for name, tool in TOOLS.items()},
            "existing_needle_arc_fitter": self.pipeline_tools / "make_needle_arc_npz_from_points3d.py",
            "needle_arc_verification": Path(__file__).with_name("needle_arc_verification.py"),
            "native_qp_runner": self.pipeline_tools / "reconstruct_thread_sam3_original.py",
        }
        if self.original_src is not None:
            package = self.original_src / "thread_reconstruction"
            if not package.is_dir():
                raise ValueError("Original QP package is missing")
            for source in sorted(package.rglob("*.py")):
                self.source_paths["original_qp/" + str(source.relative_to(package))] = source
        self.input_fingerprints = fingerprints({**self.inputs, "case_manifest": str(self.case_path)})
        self.source_fingerprints = fingerprints(self.source_paths)
        self.observation = inspect_case({**self.inputs, "baseline_m": baseline}, self.pipeline_tools)
        self.observation["available_tools"] = [name for name in TOOLS
                                                if name != "thread_qp" or self.original_src is not None]
        self.controller = {"kind": controller, "backend": controller_backend}
        self.max_calls, self.timeout_seconds = max_calls, timeout_seconds
        self.index, self.last_parameters = 0, {}
        self.limits = MappingProxyType(dict(LIMITS))
        self.out = Path(out).resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.log_path = self.out / "events.jsonl"
        self._event("session_started", case=self.case, observation=self.observation,
                    input_fingerprints=self.input_fingerprints, source_fingerprints=self.source_fingerprints,
                    shared_limits=dict(self.limits), max_calls=max_calls, timeout_seconds=timeout_seconds,
                    purpose=purpose, python_version=sys.version,
                    tool_schemas=tool_schemas(self.observation["available_tools"]),
                    cache_used=False)

    def _event(self, event, **fields):
        record = {"event": event, "utc": datetime.now(timezone.utc).isoformat(),
                  "controller": self.controller, "case_id": self.case["case_id"], **fields}
        with self.log_path.open("a") as stream:
            stream.write(json.dumps(scrub(record), allow_nan=False, sort_keys=True) + "\n")

    def observe(self):
        return strict_json(json.dumps(self.observation))

    def _command(self, name, parameters, out):
        tool = TOOLS[name]
        prefix = tool["target"]
        for key in (prefix + "_left_mask", prefix + "_right_mask"):
            if key not in self.inputs:
                raise ValueError("Missing object masks for " + name)
        command = [sys.executable, str(self.source_paths[name])]
        if name in {"needle_arc", "thread_qp"}:
            command += ["--pipeline-tools", str(self.pipeline_tools)]
        if name == "thread_qp":
            if self.original_src is None:
                raise ValueError("Production QP tool requires explicit original_src; no fallback")
            command += ["--original-src", str(self.original_src),
                        "--min-depth-m", str(self.limits["min_depth_m"]),
                        "--max-depth-m", str(self.limits["max_depth_m"]),
                        "--min-valid-depth-fraction", str(self.limits["min_valid_visible_depth"])]
        for key, flag in (("left", "--left"), ("right", "--right"), ("depth_npy", "--depth-npy"),
                          ("K", "--K"), (prefix + "_left_mask", "--left-mask"),
                          (prefix + "_right_mask", "--right-mask")):
            command += [flag, self.inputs[key]]
        command += ["--baseline-m", str(self.case["baseline_m"])]
        command += ["--out-dir" if prefix == "needle" else "--out", str(out)]
        command += ["--samples", str(self.limits["needle_samples"] if prefix == "needle" else self.limits["thread_samples"])]
        if prefix == "thread" and name != "thread_qp":
            command += ["--segment-min-skeleton-px", str(self.limits["segment_min_skeleton_px"])]
        if name == "thread_occlusion":
            if not self.observation.get("left_occluder_pixels"):
                raise ValueError("Occlusion tool requires a supplied nonempty occluder mask")
            command += ["--left-occluder", self.inputs["left_occluder"]]
            if "right_occluder" in self.inputs:
                command += ["--right-occluder", self.inputs["right_occluder"]]
            command += ["--min-visible-coverage", str(self.limits["min_visible_coverage"]),
                        "--min-valid-visible-depth", str(self.limits["min_valid_visible_depth"]),
                        "--min-visible-right-fraction", str(self.limits["min_visible_right_fraction"])]
        for key, value in parameters.items():
            command += [tool["flags"][key], str(value)]
        return command

    def call(self, request):
        """Consumes one attempt even when rejected; model cannot provide file paths/thresholds."""
        self.index += 1
        call_id = f"call_{self.index:03d}"
        folder = self.out / call_id
        folder.mkdir()
        result = {"call_id": call_id, "status": "rejected", "accepted": False,
                  "verification": None, "api_cost_usd": 0., "compute_cost_usd": None,
                  "independent_accuracy_measured": False}
        start = time.perf_counter()
        self._event("tool_requested", call_id=call_id, request=request, observation=self.observation)
        try:
            if isinstance(request, str):
                request = agent_request(request)
            if self.index > self.max_calls:
                raise ValueError("Shared call budget exhausted")
            if not isinstance(request, dict) or set(request) != {"tool", "parameters", "reason"}:
                raise ValueError("Request must contain exactly tool, parameters and reason")
            reason = request["reason"]
            if not isinstance(reason, str) or not reason.strip() or len(reason) > 500:
                raise ValueError("A concise reason of 1–500 characters is required")
            name = request["tool"]
            if not isinstance(name, str):
                raise ValueError("tool must be a string")
            parameters = parameters_for(name, request["parameters"])
            previous = self.last_parameters.get(name, TOOLS[name]["defaults"])
            changes = {k: {"from": previous[k], "to": v} for k, v in parameters.items() if previous[k] != v}
            command = self._command(name, parameters, folder / "backend")
            if fingerprints({**self.inputs, "case_manifest": str(self.case_path)}) != self.input_fingerprints:
                raise ValueError("Inputs changed since session start")
            if fingerprints(self.source_paths) != self.source_fingerprints:
                raise ValueError("Source changed since session start")
            self.last_parameters[name] = dict(parameters)
            self._event("tool_started", call_id=call_id, operation=name, effective_parameters=parameters,
                        parameter_changes=changes, concise_justification=reason, command=command,
                        shared_limits=dict(self.limits))
            result.update(status="running", tool=name, effective_parameters=parameters, parameter_changes=changes)
            with (folder / "backend_stdout.txt").open("w") as stdout, (folder / "backend_stderr.txt").open("w") as stderr:
                completed = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=self.timeout_seconds,
                                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, shell=False)
            result["backend_exit_code"] = completed.returncode
            report_name = "mask_backprojection_report.json" if name == "needle_mask_points" else "quality_report.json"
            report_path = folder / "backend" / report_name
            backend_report = strict_json(report_path.read_text()) if report_path.exists() else {}
            result.update(backend_exit_code=completed.returncode, backend_status=backend_report.get("status"),
                          backend_report=backend_report)
            if name == "thread_qp":
                result["backend_diagnostics"] = native_qp_diagnostics((folder / "backend_stdout.txt").read_text())
                trace = folder / "backend" / "qp_ordering_trace.json"
                if trace.exists():
                    result["backend_diagnostics"]["ordering_decision_trace"] = strict_json(trace.read_text())
            verifier = {"needle_mask_points": verify_needle, "needle_arc": verify_needle_arc,
                        "thread_qp": verify_native_qp}.get(name, verify_thread)
            verification = verifier(
                {**self.inputs, "baseline_m": self.case["baseline_m"]}, folder / "backend",
                self.pipeline_tools, backend_report, self.limits)
            if fingerprints({**self.inputs, "case_manifest": str(self.case_path)}) != self.input_fingerprints:
                verification["failures"].append("input_changed_during_execution")
                verification["accepted"] = False
            if fingerprints(self.source_paths) != self.source_fingerprints:
                verification["failures"].append("source_changed_during_execution")
                verification["accepted"] = False
            result["verification"] = verification
            result["accepted"] = completed.returncode == 0 and backend_report.get("status") == "passed" and verification["accepted"]
            result["status"] = "accepted" if result["accepted"] else "failed"
        except subprocess.TimeoutExpired:
            result.update(status="failed", error="backend_timeout")
        except Exception as error:
            result["error"] = str(error)
            if result["status"] == "running" or "backend_exit_code" in result:
                result["status"] = "failed"
        result["runtime_seconds"] = time.perf_counter() - start
        result["artifacts"] = [str(p.relative_to(self.out)) for p in sorted(folder.rglob("*")) if p.is_file()]
        result["artifact_sha256"] = {str(p.relative_to(self.out)): sha256(p)
                                     for p in sorted(folder.rglob("*")) if p.is_file()}
        (folder / "tool_result.json").write_text(json.dumps(scrub(result), indent=2, allow_nan=False) + "\n")
        self._event("tool_finished", **result)
        return result


def heuristic_request(observation):
    """Untuned transparent initial rule; not an evaluated heuristic baseline."""
    if observation.get("thread_component_count", 0) > 1 and observation.get("left_occluder_pixels", 0) > 0:
        return {"tool": "thread_occlusion", "parameters": {},
                "reason": "Multiple retained thread fragments and a supplied occluder; test the constrained join."}
    if "thread_qp" in observation.get("available_tools", []):
        return {"tool": "thread_qp", "parameters": {},
                "reason": "No multi-fragment occlusion evidence; use production ordering and QP defaults."}
    return {"tool": "thread_distance", "parameters": {},
            "reason": "No multi-fragment occlusion evidence; use the unchanged distance connector defaults."}


def agent_request(json_text):
    """Model-neutral adapter only; this function does not call an LLM."""
    if len(json_text) > 8192:
        raise ValueError("Agent request is too large")
    return strict_json(json_text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", action="store_true")
    parser.add_argument("--case", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--controller", choices=("fixed", "heuristic", "agent"), default="heuristic")
    parser.add_argument("--request", type=Path, help="JSON tool request; agent mode uses recorded requests, not an LLM.")
    parser.add_argument("--pipeline-tools", type=Path, default=DEFAULT_TOOLS)
    parser.add_argument("--original-src", type=Path, help="Original thread_reconstruction package parent for thread_qp.")
    parser.add_argument("--max-calls", type=int, default=4)
    parser.add_argument("--purpose", choices=("development", "evaluation"), default="development")
    args = parser.parse_args()
    if args.schema:
        print(json.dumps({"tools": tool_schemas(), "shared_limits": LIMITS}, indent=2))
        return
    if not args.case or not args.out:
        parser.error("--case and --out are required")
    session = ReconstructionSession(args.case, args.out, args.controller,
                                    "recorded_request_not_llm" if args.controller == "agent" else "initial_rule",
                                    args.pipeline_tools, args.max_calls, purpose=args.purpose,
                                    original_src=args.original_src)
    if args.request:
        request = args.request.read_text()
    elif args.controller == "agent":
        parser.error("Agent adapter requires --request; no LLM has been invoked")
    elif args.controller == "fixed":
        request = {"tool": "thread_qp", "parameters": {}, "reason": "Production QP defaults baseline."}
    else:
        request = heuristic_request(session.observe())
    result = session.call(request)
    print(json.dumps({"out": str(session.out), "call_id": result["call_id"],
                      "status": result["status"], "verification": result["verification"]}, indent=2))
    if not result["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
