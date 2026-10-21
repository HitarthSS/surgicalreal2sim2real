"""Preparation must preserve input identity and explicit observation metadata."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scene_pipeline import prepare


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PrepareTests(unittest.TestCase):
    def fixture(self, root):
        package = root / "package"
        template = package / "runtime_template"
        template.mkdir(parents=True)
        (template / "contract.py").write_text(
            "def test_rows(root):\n return []\n\ndef api_attempts(folder):\n return []\n"
        )
        (template / "preregistration.json").write_text("{}")
        for name in ("left.png", "right.png", "K.csv", "calibration.json", "state.json"):
            (root / name).write_text(name)
        case = {
            "case_id": "snapshot_20260101_120000", "baseline_m": 0.006,
            "inputs": {"left": str(root / "left.png"), "right": str(root / "right.png"),
                       "K": str(root / "K.csv")},
        }
        row = {
            "case_id": case["case_id"], "difficulty": "hard", "case": case,
            "image_sha256": {"left": digest(root / "left.png"), "right": digest(root / "right.png")},
            "calibration_path": str(root / "calibration.json"),
            "robot_state_path": str(root / "state.json"),
        }
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"scenes": [row]}))
        return package, manifest, row

    def test_metadata_and_input_hashes_are_bound_to_prepared_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, manifest, row = self.fixture(root)
            with patch.object(prepare, "PACKAGE", package), patch.object(prepare, "doctor", return_value={"sam3_checkpoint": "/models/model.pt", "sam3_sha256": "hash", "native_tracker": "/assets/tracker", "legacy_assets": "/assets"}):
                prepare.main(manifest, root / "config.json", root / "output")
            path = root / "output/gpt/day20260101" / row["case_id"] / "case.json"
            case = json.loads(path.read_text())
            self.assertEqual(case["immutable_input_sha256"]["left"], row["image_sha256"]["left"])
            self.assertEqual(case["observation_metadata"]["calibration.json"]["sha256"], digest(root / "calibration.json"))
            self.assertEqual(case["observation_metadata"]["psm_state.json"]["path"], str((root / "state.json").resolve()))
            self.assertEqual((root / "left.png").read_text(), "left.png")

    def test_explicit_missing_metadata_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, manifest, row = self.fixture(root)
            (root / "calibration.json").unlink()
            with patch.object(prepare, "PACKAGE", package), patch.object(prepare, "doctor", return_value={"sam3_checkpoint": "/models/model.pt", "sam3_sha256": "hash", "native_tracker": "/assets/tracker", "legacy_assets": "/assets"}):
                with self.assertRaises(FileNotFoundError):
                    prepare.main(manifest, root / "config.json", root / "output")

    def test_existing_output_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, manifest, _ = self.fixture(root)
            output = root / "output"
            output.mkdir()
            (output / "important.txt").write_text("keep me")
            with patch.object(prepare, "doctor", return_value={}):
                with self.assertRaises(FileExistsError):
                    prepare.main(manifest, root / "config.json", output)
            self.assertEqual((output / "important.txt").read_text(), "keep me")


if __name__ == "__main__":
    unittest.main()
