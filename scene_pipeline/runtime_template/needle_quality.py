"""Initial needle selection obeys the same rigid-template contract as repairs."""
from pathlib import Path
import json
from fixed_needle_fitter import build_catalog,load_npz,save,CATALOG
from rigid_needle_contract import audit

def recover_needle(case,original_path,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    selected=None;report=None;error=None;verified=False
    before=audit(load_npz(original_path),CATALOG,require_pose=True)
    if before['usable']:selected=str(original_path)
    try:
        from needle_size_stage import geometry_choice
        catalog=build_catalog(case,original_path,[],out/'rigid_catalog')
        row,verified=geometry_choice(catalog)
        if row is not None:selected=row['needle_path'];report=row['report']
    except ValueError as exc:error=str(exc)
    result=dict(selected_path=selected,original_path=str(original_path) if original_path else None,
        legacy_reference_path=str(original_path) if original_path else None,original_preserved=True,
        source_rigid_contract=before,complete_visible_needle=bool(selected and verified),
        requires_perception_feedback=not verified,candidate_report=report,error=error,
        nominal_needle_radius_used=True,representation='rigid_catalog_arc_v2',
        independent_accuracy_measured=False,physics_validated=False)
    save(out/'report.json',result);return result
