"""Portable advisor handoffs using the existing scene.json/curve/needle schema.

Runs without credentials or network. Scientific outputs are read-only; every
attempt, including partial geometry and missing outputs, remains indexed.
"""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import html
import json
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
RAW = Path('/home/hitarth/ismr_2027_data/incoming/2026-10-15')
OUT = ROOT / 'advisor_handoff'
sys.path.insert(0, str(ROOT / 'executed_sources'))
from advisor_scene_handoff import build
from newton_scene_adapter import CAMERA_TO_NEWTON, transform
from reconstruction_tools import fingerprints


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def copy(source, target):
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def one(source, method, revision):
    cid = source.name
    short = cid.rsplit('_', 1)[1]
    out = OUT / cid / method
    if (out / 'package_verified.json').is_file():
        saved = json.loads((out / 'package_verified.json').read_text())
        assert all(sha(out / name) == value for name, value in saved['files'].items()), 'Existing handoff changed'
        return
    assert not out.exists(), 'Partial package exists; inspect before repair: ' + str(out)
    summary_path = source / method / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    candidate = summary.get('candidate') or {}
    report = candidate.get('report') or {}
    connected = candidate.get('stage') == 'connected' and report.get('numerical_checks_passed') and report.get('fragment_count') == 1
    if candidate:
        assert fingerprints(candidate['protected_paths']) == candidate['fingerprints']
    export_summary = summary_path
    if connected:
        # The older advisor writer expects a provenance block absent from v7+
        # summaries. Bind its compatibility sidecar to the already-frozen inputs;
        # never rewrite or reinterpret the original controller decision.
        case_path = source / 'case.json'
        case = json.loads(case_path.read_text())
        cache = json.loads((source / 'preprocessing_cache.json').read_text())
        assert fingerprints(case['inputs']) == cache['input_fingerprints']
        reg = json.loads((source.parent / 'preregistration.json').read_text())
        assert all(sha(Path(case['inputs'][view])) == case['immutable_input_sha256'][view] for view in ['left','right']), 'Current input image hash mismatch'
        sidecar = dict(summary)
        sidecar['provenance'] = {view: dict(path=case['inputs'][view], sha256=sha(case['inputs'][view])) for view in ['left', 'right']}
        sidecar['provenance']['case'] = dict(path=str(case_path), sha256=sha(case_path))
        export_summary = ROOT / 'handoff_context' / cid / method / 'summary.json'
        export_summary.parent.mkdir(parents=True, exist_ok=True)
        write(export_summary, sidecar)
    common = source / 'common_needle.npz'
    context = None
    if common.is_file():
        with np.load(common, allow_pickle=False) as data:
            needle = {key: data[key].copy() for key in data.files}
        context_folder = ROOT / 'handoff_context'
        context_folder.mkdir(exist_ok=True)
        context = context_folder / (cid + '.npz')
        if not context.exists():
            np.savez_compressed(context, needle_points_newton_m=transform(needle['points_m'], CAMERA_TO_NEWTON),
                                needle_inferred=needle['inferred'], needle_tube_radius_m=needle['tube_radius_m'])
    manifest = build(RAW / cid, out, export_summary if connected else None, context)
    status = 'connected_proposal' if connected else ('checked_partial' if report.get('numerical_checks_passed') else 'rejected_or_conversion_failed')
    manifest.update(controller=method, scientific_revision=revision, reconstruction_status=status,
                    all_current_endpoints_assessed=(summary.get('completion') or {}).get('all_current_endpoints_assessed', False),
                    original_controller_source_summary=str(summary_path),
                    original_controller_source_summary_sha256=sha(summary_path) if summary_path.is_file() else None,
                    geometry_failures=report.get('failures', []),
                    hypothesis_lineage=report.get('hypothesis_lineage', []),
                    independent_accuracy_measured=False, simulation_ready=False, training_eligible=False,
                    fitted_robot_tools_available=False)
    if not summary:
        manifest['geometry_failures'] = ['No controller summary; inspect conversion_failure.json']
    if candidate:
        copy(Path(candidate['fragment_path']), out / 'diagnostics/fragments.npz')
        copy(Path(candidate['protected_paths']['report']), out / 'diagnostics/reconstruction_report.json')
        manifest['diagnostic_fragments'] = 'diagnostics/fragments.npz'
    for name in ['history.json', 'endpoint_ledger.json']:
        copy(source / method / name, out / 'diagnostics' / name)
    for name in ['preparation_failure.json', 'preprocessing.json']:
        copy(source / name, out / 'diagnostics' / name)
    inspection = source / method / 'inspection'
    for name in ['scene_newton_m.obj', 'scene.mtl']:
        copy(inspection / name, out / 'geometry' / name)
    copy(source / 'comparison.jpg', out / 'review/comparison.jpg')
    attachment = summary.get('attachment') or {}
    metadata = attachment.get('metadata') or {}
    attached = connected and attachment.get('status') == 'geometric_attachment_proposal'
    if attached:
        assert attachment['source_geometry_sha256'] == sha(candidate['fragment_path'])
        with np.load(attachment['artifact'], allow_pickle=False) as data:
            geometry = {key: data[key].copy() for key in data.files}
        geometry['needle_tube_radius_m'] = needle['tube_radius_m']
        geometry['tool_vertices_newton_m'] = np.empty((0, 3))
        geometry['tool_faces'] = np.empty((0, 3), np.int32)
        (out / 'newton').mkdir()
        np.savez_compressed(out / 'newton/scene_bundle.npz', **geometry)
        ti, ni = metadata['thread_endpoint_index'], metadata['needle_endpoint_index']
        assert np.array_equal(geometry['points_newton_m'][ti], geometry['needle_points_newton_m'][ni])
        connection = dict(scene_bundle_sha256=sha(out / 'newton/scene_bundle.npz'), attachment=metadata,
                          source_attachment_sha256=sha(attachment['artifact']), hypothesis_lineage=manifest['hypothesis_lineage'],
                          independent_accuracy_measured=False, motion_tested=False, training_eligible=False)
        write(out / 'newton/connection_report.json', connection)
        manifest['attachments'] = [dict(type='needle_thread_ball_joint_proposal', geometry='newton/scene_bundle.npz',
                                       contract='newton/connection_report.json', status='inferred_geometric_proposal',
                                       mechanical_validation='not_tested_in_this_package', **metadata)]
        manifest['needle']['needle_thread_attachment_established'] = False
        manifest['needle']['needle_thread_attachment_proposed'] = True
    else:
        manifest['attachment_status'] = attachment.get('status', 'unavailable')
    # Historical checks are optional and cannot validate a newly reconstructed scene.
    old_newton = BASE / 'ismr_new9_newton_20261016_v1/scenes' / short
    if method == 'agent':
        from newton_artifact_provenance import include_exact_prior_check
        include_exact_prior_check(manifest, old_newton, summary_path, out, sha, copy)
    write(out / 'scene.json', manifest)
    with (out / 'README.md').open('a') as stream:
        stream.write('\nController: ' + method + '; reconstruction: ' + status + '; source revision: ' + revision + '.\n\n'
                     'geometry/scene_newton_m.obj, when present, is the prior inspection mesh and omits the new attachment connector. '
                     'newton/scene_bundle.npz, when present, includes the inferred attachment bridge; use its paired connection_report.json to construct the ball joint. '
                     'Hidden samples remain inferred. Attachment is a proposal, not independently measured truth. '
                     'diagnostics/fragments.npz may contain partial/rejected geometry and must not be promoted to a valid scene. '
                     'Robot states retain their recorded base frames; camera/base calibration and articulated tool models are still missing.\n')
    files = {str(path.relative_to(out)): sha(path) for path in out.rglob('*') if path.is_file()}
    write(out / 'package_verified.json', dict(case_id=cid, controller=method, source_summary_sha256=manifest['original_controller_source_summary_sha256'],
                                            status=status, files=files, portable_relative_artifact_paths=True))


def index(final=False):
    rows = []
    for path in sorted(OUT.glob('snapshot_*/*/scene.json')):
        scene = json.loads(path.read_text())
        rows.append(dict(scene_id=scene['scene_id'], controller=scene['controller'],
                         status=scene['reconstruction_status'], revision=scene['scientific_revision'],
                         attachment_proposed=bool(scene['attachments']),
                         thread_curve=scene['thread'] is not None, needle_geometry=scene['needle'] is not None,
                         robot_arm_ids=[arm['arm_id'] for arm in scene['robots']],
                         path=str(path.relative_to(OUT)), independent_accuracy_measured=False, training_eligible=False))
    manifest = json.loads((ROOT / 'job_manifest.json').read_text())
    if final:
        expected = {(cid, method) for cid in sorted(p.name for p in RAW.glob('snapshot_*') if p.is_dir()) for method in ['heuristic', 'agent']}
        assert {(row['scene_id'], row['controller']) for row in rows} == expected, 'Missing handoff attempts'
    write(OUT / 'index.json', dict(schema='ismr_advisor_collection_v1', finalized=final, acquisition_scenes=63,
                                  packaged_controller_attempts=len(rows), remaining_run_cases=manifest['cases'], scenes=rows,
                                  first_ten_reused_with_revision_labels=True, independent_accuracy_measured=False, training_eligible=False))
    with (OUT / 'index.csv').open('w', newline='') as stream:
        fields = ['scene_id', 'controller', 'status', 'revision', 'attachment_proposed', 'thread_curve', 'needle_geometry', 'path']
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    table = ''.join('<tr>' + ''.join('<td>' + html.escape(str(row[k])) + '</td>' for k in ['scene_id', 'controller', 'status', 'revision']) +
                    '<td><a href="' + row['path'] + '">scene.json</a> · <a href="' + str(Path(row['path']).parent / 'README.md') + '">README</a></td></tr>' for row in rows)
    (OUT / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Advisor scene collection</title><style>body{font:16px system-ui;margin:28px}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style><h1>October15 reconstruction collection</h1><p>Scene files for inspection and Newton integration. Partial and failed attempts retained; robot calibration/materials/action mapping remain incomplete. Revision labels distinguish reused development cases from the frozen remaining53 run.</p><table><tr><th>Scene</th><th>Controller</th><th>Result</th><th>Revision</th><th>Files</th></tr>' + table + '</table>')
    (OUT / 'README.md').write_text('''# Advisor handoff — October15 stereo scenes

Open index.html or index.csv first. Every scene has heuristic/ and agent/ folders, each starting with scene.json in the same schema as the previous advisor handoff. Completed prior cases are reused with source revision labels; they are not pooled as one frozen comparison.

- observations/: original left/right rectified images, camera calibration and per-arm recorded robot states.
- thread_curve.npz: checked connected centerline, support labels and optional BSpline coefficients, when available.
- needle_geometry.npz: estimated rigid needle geometry, body-frame convention and camera pose, when available.
- geometry/: preserved OBJ/MTL inspection export. This export omits the later attachment connector.
- newton/: exact anchored centerlines and ball-joint contract when the agent proposed an attachment. No new dynamics validation is implied.
- prior_newton_check/: exact preserved 0.1-second diagnostics for the nine already tested agent scenes only.
- diagnostics/: partial/rejected fragments, numerical report, controller decision history and endpoint ledger. These remain failures/partials in the index.
- package_verified.json: hashes for every per-controller package file.

All geometry is in meters under provisional scale; canonical camera frame and T_newton_from_camera are explicit. Do not apply the transform twice. OBJ meshes contain no physical joints. For Newton use the sampled centerline to build a flexible rod, the rigid needle geometry, and the proposed ball-joint anchor if present. Calibrate physical radii, mass, stiffness, damping, support/contact settings before real manipulation claims.

Robot PSM1/PSM2 states are recorded in their original base frames and lag the images. No camera/base identity transform, robot identity from image position, fitted tool, articulation or action mapping has been invented. Static snapshots are not imitation-learning demonstrations. The prior Newton prototype used zero gravity and 200× rod mass scaling and emitted inertia-correction warnings. No independent geometry accuracy or physical fidelity has been measured.
''')
    print(json.dumps(dict(packaged_controller_attempts=len(rows), finalized=final, statuses=dict(Counter(row['status'] for row in rows)))))


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--existing', action='store_true')
    group.add_argument('--batch')
    group.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    if args.existing:
        refs = {}
        for batch in (BASE / 'ismr_new10_20261015_v1').glob('batch*'):
            if batch.is_dir():
                for folder in batch.glob('snapshot_*'):
                    refs[folder.name] = (folder, 'v9_original_first10')
        refs['snapshot_20261015_145924'] = (BASE / 'ismr_partial_repair_20261016_v2/snapshot_20261015_145924', 'v10.1_boundary_hypothesis')
        refs['snapshot_20261015_150704'] = (BASE / 'ismr_partial_repair_20261016_v3/snapshot_20261015_150704', 'v10.2_lossless_text_dictionary_encoding')
        assert len(refs) == 10
        for folder, revision in refs.values():
            for method in ['heuristic', 'agent']:
                one(folder, method, revision)
    elif args.batch:
        folder = ROOT / args.batch
        reg = json.loads((folder / 'preregistration.json').read_text())
        for short in reg['cases']:
            for method in ['heuristic', 'agent']:
                one(folder / ('snapshot_20261015_' + short), method, 'v10.2_lossless_text_dictionary_encoding')
    index(args.finalize)


if __name__ == '__main__':
    main()
