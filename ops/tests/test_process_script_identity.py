import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refcloth_resource as guard


@unittest.skipUnless(Path('/proc/self/cmdline').exists(), 'Linux /proc required')
class ProcessScriptIdentity(unittest.TestCase):
    def test_logical_script_alias_remains_the_same_live_claim_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / 'real'
            real.mkdir()
            script = real / 'worker.py'
            script.write_text('import time\ntime.sleep(60)\n')
            alias = root / 'alias'
            alias.symlink_to(real, target_is_directory=True)
            process = subprocess.Popen([sys.executable, str(alias / 'worker.py')])
            try:
                identity = guard.identity(process.pid)
                self.assertNotIn(str(script.resolve()), identity['cmd'])
                self.assertTrue(guard.process_runs_script(process.pid, script, identity['ticks']))
                self.assertTrue(guard.process_runs_script(process.pid, alias / 'worker.py', identity['ticks']))
                self.assertFalse(guard.process_runs_script(process.pid, script, identity['ticks'] + 1))
                self.assertFalse(guard.process_runs_script(process.pid, root / 'elsewhere/worker.py', identity['ticks']))
            finally:
                process.terminate()
                process.wait(timeout=5)
            self.assertFalse(guard.process_runs_script(process.pid, script, identity['ticks']))


if __name__ == '__main__':
    unittest.main()
