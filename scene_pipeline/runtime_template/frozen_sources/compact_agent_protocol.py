"""Versioned surgical refinement interface; raw evidence stays outside prompts.

This is a controller adapter to existing tools, not a new estimator or verifier.
Image coordinates remain those of the original rectified stereo pair.
"""
from collections import Counter
import hashlib
import json

from reconstruction_tools import TOOLS, parameters_for, strict_json

PROTOCOL = "surgical_refinement_v1"
FIELDS = {"scene_version", "target", "operation", "parameters", "evidence_ids", "reason"}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def accepted_target(result, target):
    allowed = {"needle_arc"} if target == "needle" else {"thread_qp", "thread_occlusion", "thread_distance"}
    return bool(result and result.get("tool") in allowed and result.get("accepted") is True
                and (result.get("verification") or {}).get("accepted") is True)


def relevant_tools(observation, target):
    if target not in {"needle", "thread"}:
        raise ValueError("Unknown target")
    names = []
    for name in observation.get("available_tools", []):
        if name not in TOOLS or TOOLS[name]["target"] != target:
            continue
        if name == "thread_occlusion" and not observation.get("left_occluder_pixels", 0):
            continue
        names.append(name)
    return names


def tool_contracts(names):
    return {name: {"purpose": TOOLS[name]["description"], "defaults": TOOLS[name]["defaults"],
                   "bounds": TOOLS[name]["bounds"],
                   "kind": "diagnostic_reference" if name == "thread_distance" else "reconstruction"}
            for name in names}


def make_state(observation, provenance, target, remaining, result=None, revision=0, attempted=()):
    """No confidence/accuracy promotion. All candidate records remain addressable.

    ``provenance`` includes immutable input/source fingerprints. The version binds
    these, full diagnostics, current result and budgets, not just displayed fields.
    Candidate order is preserved, with no quality ranking or silent filtering.
    """
    names = relevant_tools(observation, target)
    diagnostics = {k: v for k, v in observation.items() if k not in {"images", "available_tools"}}
    catalog = {"observation": diagnostics}
    current = None
    candidate_summary = None
    if result is not None:
        check = result.get("verification") or {}
        catalog["verification"] = check
        catalog["backend_diagnostics"] = result.get("backend_diagnostics") or {}
        current = {k: result.get(k) for k in ("call_id", "tool", "status", "accepted", "effective_parameters", "error")}
        current.update(failures=check.get("failures", []), metrics=check.get("metrics", {}))
        report = result.get("backend_report") or {}
        catalog["backend_status"] = {k: report.get(k) for k in ("status", "error", "failures", "joins")}
        candidates = report.get("candidates") or []
        if not isinstance(candidates, list):
            raise ValueError("Expected a candidate list; preserve unsupported reports for inspection")
        rejection_counts = Counter()
        for i, candidate in enumerate(candidates):
            catalog[f"candidate.{i:03d}"] = candidate
            reasons = candidate.get("rejections", candidate.get("failures", [])) if isinstance(candidate, dict) else []
            if isinstance(reasons, str):
                reasons = [reasons]
            rejection_counts.update(str(reason) for reason in reasons)
        candidate_summary = {"count": len(candidates), "id_format": "candidate.NNN (zero based)",
                             "rejection_counts": dict(rejection_counts),
                             "records_inline": False, "ambiguity_resolved_by_summary": False}
    state = {"protocol": PROTOCOL, "target": target, "revision": revision,
             "observation": diagnostics, "remaining": remaining,
             "operations": tool_contracts(names), "current_result": current,
             "backend_diagnostics": compact_backend_diagnostics(catalog.get("backend_diagnostics") or {}),
             "candidate_index": candidate_summary,
             "evidence_ids": [key for key in catalog if not key.startswith("candidate.")],
             "attempted_operations": list(attempted),
             "validity": {"target_input_consistency_passed": accepted_target(result, target),
                          "complete_scene_assembled": False, "physics_validated": False,
                          "trajectory_validated": False, "independent_accuracy_measured": False,
                          "training_eligible": False},
             "calibration_status": provenance.get("calibration_status", "declared_scale_not_independently_verified")}
    state["scene_version"] = digest({"state": state, "catalog": catalog, "provenance": provenance,
                                     "full_result": result})
    return state, catalog


def compact_backend_diagnostics(diagnostic):
    trace = diagnostic.get("ordering_decision_trace") or {}
    return {"ordering_stops": diagnostic.get("native_ordering_stops", []),
            "selected_join_count": len(trace.get("joins", [])),
            "joins_validated": trace.get("joins_validated", False),
            "returns": trace.get("returns", []), "full_record_id": "backend_diagnostics"}


def validate_decision(raw, state, catalog):
    decision = strict_json(raw) if isinstance(raw, str) else decision_from_mapping(raw)
    if not isinstance(decision, dict) or set(decision) != FIELDS:
        raise ValueError("Expected exactly the six protocol fields")
    if decision["scene_version"] != state["scene_version"] or decision["target"] != state["target"]:
        raise ValueError("Stale scene version or wrong target")
    reason = decision["reason"]
    if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 500:
        raise ValueError("Concise reason required")
    ids = decision["evidence_ids"]
    if (not isinstance(ids, list) or not 1 <= len(ids) <= 3 or any(not isinstance(i, str) for i in ids)
            or len(set(ids)) != len(ids) or any(i not in catalog for i in ids)):
        raise ValueError("Cite one to three available evidence IDs")
    op, parameters = decision["operation"], decision["parameters"]
    if not isinstance(parameters, dict) or not isinstance(op, str):
        raise ValueError("Invalid operation/parameters")
    if op in {"finish", "defer", "inspect_evidence"}:
        if parameters:
            raise ValueError("Control operations require empty parameters")
        if op == "finish" and not state["validity"]["target_input_consistency_passed"]:
            raise ValueError("finish requires shared-verified target geometry")
        if op == "inspect_evidence" and state["remaining"].get("evidence_reads", 0) < 1:
            raise ValueError("Evidence-read budget exhausted")
    else:
        if op not in state["operations"]:
            raise ValueError("Operation unavailable for this target/evidence")
        if state["remaining"].get("tool_calls", 0) < 1:
            raise ValueError("Reconstruction budget exhausted")
        parameters_for(op, parameters)  # Same shared bounds; no separate relaxation.
    return decision


def decision_from_mapping(value):
    return strict_json(encoded(value))


def shared_request(decision):
    return {"tool": decision["operation"], "parameters": decision["parameters"], "reason": decision["reason"]}


def signature(decision):
    params = parameters_for(decision["operation"], decision["parameters"])
    return encoded([decision["operation"], {key: float(value) for key, value in params.items()}])


def messages_for(state, image_evidence, requested_evidence=None):
    """A fresh state each turn; original stereo retained without accumulated logs.

    Image evidence uses absolute source paths, hashes logged by the runner. Full
    stereo resolution is retained; overlays are added only after execution.
    No image-token savings or automatic inference-result caching is assumed.
    """
    rules = (
        "Refine one surgical reconstruction target. Return only JSON with exactly: "
        "scene_version (copy current), target, operation, parameters, evidence_ids (1-3 IDs), reason (<=500 chars). "
        "Select one listed operation, finish, defer, or inspect_evidence. Control parameters must be {}. "
        "inspect_evidence retrieves cited records and consumes a model decision plus one evidence read. "
        "Use numeric checks and paired image evidence. Do not repeat effective parameters. "
        "finish requires shared-verified target geometry and does not mean a complete valid scene. "
        "Preserve uncertainty: mask agreement/residuals are not independent accuracy; hidden geometry is inferred. "
        "Do not set needle size, switch estimators, extend endpoints, lower checks, invent occluders or paths. "
        "Thread distance is a diagnostic reference; production baseline is original QP. "
        "Candidate summaries do not resolve ambiguity; inspect relevant competing records or defer. "
        "Images use original rectified coordinates; viewing overlays does not change calibration."
    )
    content = [{"type": "text", "text": rules + "\nSTATE:" + encoded(state)}]
    if requested_evidence:
        content.append({"type": "text", "text": "REQUESTED_EVIDENCE:" + encoded(requested_evidence)})
    for label, path in image_evidence:
        content.extend([{"type": "text", "text": label}, {"type": "image", "url": str(path)}])
    return [{"role": "user", "content": content}]


def token_usage(input_ids, generated_ids, image_token_id=None):
    """Count actual batch-one model tensors, not characters or estimates.

    This adapter assumes decoder-only generation includes its input prefix; it
    rejects other layouts instead of inventing token counts. Image placeholders
    are counted only when an explicit model image-token ID is supplied.
    """
    import numpy as np
    def array(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)
    prompt, sequence = array(input_ids), array(generated_ids)
    if (prompt.ndim != 2 or prompt.shape[0] != 1 or sequence.ndim != 2 or sequence.shape[0] != 1
            or sequence.shape[1] < prompt.shape[1] or not np.array_equal(sequence[:, :prompt.shape[1]], prompt)):
        raise ValueError("Unsupported generation layout for token accounting")
    images = None if image_token_id is None else int((prompt == image_token_id).sum())
    return {"input_tokens_total": int(prompt.size), "image_placeholder_tokens": images,
            "non_image_input_tokens": None if images is None else int(prompt.size) - images,
            "output_tokens": int(sequence.size - prompt.size), "cached_tokens": None,
            "accounting": "actual_generation_tensors", "image_encoder_compute_measured": False}
