import json,os,sys,tempfile,time,unittest
from pathlib import Path
from scene_pipeline.runner import run,save
class RunnerTests(unittest.TestCase):
 def test_credential_uses_stdin_not_arguments(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);script=p/'child.py';script.write_text('import sys;assert sys.stdin.readline().strip()=="dummy-test-secret";assert "dummy-test-secret" not in repr(sys.argv)\n')
   run([sys.executable,str(script)],p/'log',os.environ.copy(),5,'dummy-test-secret')
 def test_failure_is_visible(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(RuntimeError):run([sys.executable,'-c','raise SystemExit(7)'],Path(d)/'log',os.environ.copy(),5)
 def test_watchdog_terminates_process_group(self):
  with tempfile.TemporaryDirectory() as d:
   start=time.monotonic()
   with self.assertRaises(TimeoutError):run([sys.executable,'-c','import time;time.sleep(60)'],Path(d)/'log',os.environ.copy(),.1)
   self.assertLess(time.monotonic()-start,3)
 def test_atomic_status_preserves_terminal_state(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'status.json';save(p,dict(state='failed_preserved'));self.assertEqual(json.loads(p.read_text())['state'],'failed_preserved');self.assertFalse(p.with_suffix('.tmp').exists())
if __name__=='__main__':unittest.main()
