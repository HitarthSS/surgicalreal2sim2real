"""Portable preparation metadata must survive every reconstruction loader."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / 'scene_pipeline/runtime_template'


class CaseMetadataTests(unittest.TestCase):
    def load(self, path):
        verification = types.ModuleType('reconstruction_verification')
        for name in ['inspect_case', 'verify_needle', 'verify_thread', 'verify_native_qp']:
            setattr(verification, name, lambda *a, **kw: {})
        arc = types.ModuleType('needle_arc_verification')
        arc.verify_needle_arc = lambda *a, **kw: {}
        spec = importlib.util.spec_from_file_location('metadata_test_loader', path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, reconstruction_verification=verification, needle_arc_verification=arc):
            spec.loader.exec_module(module)
        module.fingerprints = lambda paths: {}
        return module

    def test_all_loaders_accept_prepared_metadata_without_changing_observation(self):
        for path in ROOT.rglob('reconstruction_tools.py'):
            with self.subTest(loader=str(path)), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                image = root / 'input'; image.write_bytes(b'preserved')
                case = dict(case_id='snapshot_20260101_120000', session_id='session',
                            calibration_id='calibration', split='validation', baseline_m=.005,
                            inputs={k: str(image) for k in ['left', 'right', 'K', 'depth_npy',
                                                           'thread_left_mask', 'thread_right_mask']})
                case_path = root / 'case.json'
                module = self.load(path)
                case_path.write_text(json.dumps(case))
                baseline = module.ReconstructionSession(case_path, root/'baseline', 'heuristic')
                case.update(immutable_input_sha256={'left': 'a'*64},
                            observation_metadata={'calibration.json': {'path': str(image), 'sha256': 'b'*64}})
                case_path.write_text(json.dumps(case))
                enriched = module.ReconstructionSession(case_path, root/'enriched', 'agent')
                self.assertEqual(baseline.observe(), enriched.observe())
                self.assertEqual(dict(baseline.limits), dict(enriched.limits))
                self.assertEqual(enriched.case['observation_metadata'], case['observation_metadata'])
                self.assertIn('observation_metadata', (root/'enriched/events.jsonl').read_text())
                self.assertEqual(image.read_bytes(), b'preserved')
                for name, extra in [('unknown', {'unexpected_option': True}),
                                    ('bad_hash', {'immutable_input_sha256': {'left': 'invalid'}}),
                                    ('bad_metadata', {'observation_metadata': {'calibration.json': {'path': str(image)}}})]:
                    bad = dict(case); bad.update(extra); case_path.write_text(json.dumps(bad))
                    with self.assertRaises(ValueError):
                        module.ReconstructionSession(case_path, root/name, 'agent')


if __name__ == '__main__':
    unittest.main()
