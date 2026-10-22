"""Standalone and pipeline wrapper for the optional background stage."""
from pathlib import Path
import json
import os
import sys
import time

from .background import build, configuration, decision, sha
from .runner import save


def model_selector(scene, output, key):
    """Use the existing audited transport and the remaining shared scene budget."""
    runtime = scene.parent
    sys.path[:0] = [str(Path(__file__).parent/'runtime_template'), str(runtime), str(runtime/'executed_sources')]
    from openai_scene_backend import OpenAISceneModel
    # The shared 23-attempt scene cap still applies; this stage gets at most one.
    os.environ['ISMR_API_SCENE_DIR'] = str(scene)
    if output.relative_to(scene).parts[0] != 'background':
        raise ValueError('VLM attempts must be under SCENE/background for shared budget accounting')
    previous = os.environ.get('OPENAI_API_KEY')
    os.environ['OPENAI_API_KEY'] = key
    try:
        backend = OpenAISceneModel(output/'model_requests', max_calls=1)
    finally:
        if previous is None: os.environ.pop('OPENAI_API_KEY', None)
        else: os.environ['OPENAI_API_KEY'] = previous

    def select(case, diagnostics):
        prompt = ('Inspect the background surface using both complete stereo images. '
                  'Ignore tools, thread and needle. Select an approximate surface/material preset; '
                  'do not estimate measured friction, stiffness, camera pose or metric scale. '
                  'Return ONLY JSON with exactly surface_model (auto|plane|depth_mesh|sam3d), '
                  'material_preset (cloth|silicone|rigid_plastic|metal|tissue_like|unknown), '
                  'roughness (0..1), metallic (0..1), reason (string <=500 characters). '
                  'Use auto when geometry is unclear. sam3d requires an externally generated '
                  'candidate and stereo anchors; do not select it merely to fill a flat cloth plane. '
                  'Appearance is approximate and physical coefficients come from declared presets. '
                  'Stereo diagnostics: '+json.dumps(diagnostics))
        content = [dict(type='text', text=prompt)]
        for side in ['left', 'right']:
            content += [dict(type='text', text=side.upper()+' FULL VIEW'), dict(type='image', url=case['inputs'][side])]
        try:
            return decision(json.loads(backend([dict(role='user', content=content)])))
        finally:
            backend.key = ''
    return select


def run_case(case_path, output, config=None, key=None):
    scene = Path(case_path).resolve().parent; output = Path(output).resolve()
    cfg = configuration(config)
    marker = output.parent/(output.name+'_status.json')
    if marker.exists() or output.exists():
        raise FileExistsError('Background attempt exists; preserve it and use a new output directory')
    start = time.monotonic(); save(marker, dict(state='running', output=str(output)))
    selector = None
    try:
        # Create the output only in build; transport starts lazily after its diagnostics.
        if cfg['vlm']:
            if not key: raise ValueError('VLM background selection requires a credential via stdin')
            def selector(case, diagnostics):
                return model_selector(scene, output, key)(case, diagnostics)
        result = build(case_path, output, cfg, selector)
        save(marker, dict(state='complete', geometry_status=result['status'],
                          output=str(output), manifest_sha256=sha(output/'background.json'),
                          elapsed_seconds=time.monotonic()-start))
        return result
    except Exception as error:
        save(marker, dict(state='failed_preserved', error_type=type(error).__name__,
                          output=str(output), elapsed_seconds=time.monotonic()-start))
        raise


def main(root):
    root = Path(root).resolve(); cfg = json.loads((root/'config.json').read_text()).get('background', {})
    cfg = configuration(cfg)
    if not cfg['enabled']: raise ValueError('Enable background in the run configuration first')
    # Existing completed foreground stages are read-only. No foreground inference is called.
    key = sys.stdin.readline().strip() if cfg['vlm'] else None
    for row in json.loads((root/'scope.json').read_text())['scenes']:
        scene = root/'gpt'/('day'+row['case_id'].split('_')[1])/row['case_id']
        marker = scene/'background_status.json'
        if marker.exists() or (scene/'background').exists(): continue
        os.environ['ISMR_API_STATE_DIR'] = str(root/'API_state')
        try: run_case(scene/'case.json', scene/'background', cfg, key)
        except Exception: continue  # Each failure is explicit in its receipt; remaining scenes proceed.
    key = None
