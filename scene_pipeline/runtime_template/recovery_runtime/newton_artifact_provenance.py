"""Include historical simulator diagnostics only for the exact source and bundle."""
from pathlib import Path
import json

def include_exact_prior_check(manifest, old_folder, summary_path, output, sha, copy):
    old_folder=Path(old_folder)
    if not old_folder.is_dir():return 'absent'
    connection=json.loads((old_folder/'connection_report.json').read_text())
    expected=connection['source_summary_sha256'];actual=sha(summary_path)
    if expected!=actual:
        manifest['excluded_prior_newton_check']=dict(
            reason='historical_diagnostic_belongs_to_different_source_summary',
            historical_source_summary_sha256=expected,current_source_summary_sha256=actual,
            historical_report_path=str(old_folder/'connection_report.json'),
            inherited_validation=False)
        return 'excluded_different_revision'
    assert connection['scene_bundle_sha256']==sha(old_folder/'scene_bundle.npz'),'Historical bundle hash mismatch'
    for name in ['scene_bundle.npz','connection_report.json','rest/report.json','viewer/before.png','viewer/newton_playback.mp4']:
        copy(old_folder/name,Path(output)/'prior_newton_check'/name)
    manifest['prior_newton_check']=dict(report='prior_newton_check/rest/report.json',
        exact_bundle='prior_newton_check/scene_bundle.npz',duration_s=.1,motion_tested=False,
        rod_mass_scale=200,materials_calibrated=False)
    return 'included_exact_revision'
