import io,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from scene_pipeline import runner,submit
class SubmissionTests(unittest.TestCase):
 def test_submission_is_not_repeated(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'config.json').write_text(json.dumps(dict(container_image='/data/runtime.sif')))
   with patch('scene_pipeline.submit.subprocess.run') as run:
    run.return_value.stdout='12345\n';submit.main(root)
    with self.assertRaises(ValueError):submit.main(root)
    self.assertEqual(run.call_count,1)
 def test_interrupted_scientific_scene_is_preserved_without_replay(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);cid='snapshot_20261020_120000';f=root/'gpt/day20261020'/cid;f.mkdir(parents=True)
   (root/'scope.json').write_text(json.dumps(dict(scenes=[dict(case_id=cid,difficulty='hard')])));(root/'config.json').write_text(json.dumps(dict(ffs_root='/data/ffs',ffs_checkpoint='/data/model.pth')));(f/'run_started.json').write_text('{}')
   with patch.dict(os.environ,dict(SLURM_JOB_ID='test',APPTAINER_CONTAINER='test.sif')),patch('sys.stdin',io.StringIO('dummy-secret-for-unit-test\n')),patch('scene_pipeline.runner.run') as run:
    runner.main(root);run.assert_not_called()
   self.assertEqual(json.loads((f/'run_result.json').read_text())['state'],'interrupted_preserved')
if __name__=='__main__':unittest.main()
