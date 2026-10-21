"""Export calibration comes from the case, independent of acquisition date."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest


class ObservationExportTests(unittest.TestCase):
    def test_explicit_calibration_outside_raw_directory_and_input_integrity(self):
        module = Path(__file__).resolve().parents[1] / 'scene_pipeline/runtime_template/recovery_runtime/package_advisor_base_v2.py'
        tree = ast.parse(module.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'observation_snapshot')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'source/snapshot_20261021_140201'; source.mkdir(parents=True)
            raw = root/'raw'; raw.mkdir(); calibration = root/'actual-calibration.json'; calibration.write_text('{"image_size":[640,480]}')
            files = {}
            sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for side in ['left', 'right']:
                path = raw/(side+'_rect.png'); path.write_bytes(side.encode()); files[side] = str(path)
            case = dict(inputs=files, immutable_input_sha256={s:sha(files[s]) for s in files}, observation_metadata={'calibration.json':dict(path=str(calibration),sha256=sha(calibration))})
            (source/'case.json').write_text(json.dumps(case))
            namespace = dict(Path=Path, json=json, shutil=shutil, ROOT=root, sha=sha)
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(module), 'exec'), namespace)
            snapshot = namespace['observation_snapshot'](source)
            self.assertEqual(snapshot.name, source.name)
            self.assertEqual((snapshot/'calibration.json').read_bytes(), calibration.read_bytes())
            self.assertEqual((snapshot/'left_rect.png').read_bytes(), b'left')
            self.assertEqual(namespace['observation_snapshot'](source), snapshot)
            calibration.write_text('changed')
            with self.assertRaisesRegex(AssertionError, 'metadata changed'):
                namespace['observation_snapshot'](source)


if __name__ == '__main__':
    unittest.main()
