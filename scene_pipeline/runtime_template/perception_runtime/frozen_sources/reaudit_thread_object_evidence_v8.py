"""Reverify stored outputs without rerunning a model or rewriting past results."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import shutil
from complex_case_diagnostics import write_json
from reconstruction_tools import fingerprints,sha256
from shared_scene_tools_v7 import SharedSceneToolsV7
from thread_object_evidence_v8 import evaluate_thread_object_evidence


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--pipeline',type=Path,required=True)
    args=parser.parse_args();original=args.original.resolve();out=args.out.resolve();rows=[]
    for old in sorted(original.glob('snapshot_*')):
        folder=out/old.name;folder.mkdir(exist_ok=False)
        shutil.copy2(old/'case.json',folder/'case.json');shutil.copy2(old/'common_needle.npz',folder/'common_needle.npz')
        case=json.loads((folder/'case.json').read_text())
        for method in ['heuristic','agent']:
            source=old/method/'summary.json';summary=json.loads(source.read_text());prior=summary.get('candidate')
            if not prior:rows.append(dict(case_id=old.name,controller=method,no_geometry=True));continue
            assert fingerprints(prior['protected_paths'])==prior['fingerprints']
            target=folder/method;target.mkdir()
            shutil.copy2(prior['fragment_path'],target/'fragments.npz')
            for side in ['left','right']:shutil.copy2(prior['images'][side],target/(side+'_proposal.png'))
            tools=SharedSceneToolsV7(folder/'case.json',args.pipeline,needle_path=folder/'common_needle.npz')
            candidate=tools.audit_geometry(tools.package(target/'fragments.npz',deepcopy(prior['report']),target))
            report=candidate['report']
            audit=evaluate_thread_object_evidence(case,args.pipeline,candidate['fragment_path'],folder/'common_needle.npz',target/'evidence')
            assert sha256(candidate['fragment_path'])==sha256(prior['fragment_path'])
            row=dict(case_id=old.name,controller=method,before_passed=prior['report']['numerical_checks_passed'],
                after_passed=report['numerical_checks_passed'],before_failures=prior['report']['failures'],after_failures=report['failures'],
                raw_coverage=report['original_visible_mask_coverage'],object_aware_coverage=report['object_aware_original_mask_coverage'],
                ambiguous_needle_adjacent_pixels={s:v['ambiguous_needle_adjacent_skeleton_pixels'] for s,v in audit['views'].items()},
                geometry_sha256=sha256(candidate['fragment_path']),original_summary_sha256=sha256(source))
            if row['before_passed']:assert row['after_passed'],'Unexpected newly rejected saved geometry'
            updated=dict(schema='stored_geometry_object_evidence_reaudit_v8',source_summary=str(source),source_summary_sha256=sha256(source),
                candidate=candidate,attachment=summary.get('attachment'),results=row,model_rerun=False,
                geometry_changed=False,independent_accuracy_measured=False,physics_validated=False,training_eligible=False)
            write_json(target/'summary.json',updated);rows.append(row)
    write_json(out/'reverification_results.json',rows);print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
