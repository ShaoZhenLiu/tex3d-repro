"""Regression checks for memory admission, process ownership and sample recovery."""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import refcloth_resource as g
from refcloth_resume import select_checkpoint, consolidate


def healthy():
    return dict(time=time.time(), available=120*g.GIB, headroom=30*g.GIB,
                memory_psi={}, io_psi={}, iowait_percent=0., io_mib_s=0., swap_mib_s=0.)


class Admission(unittest.TestCase):
    def test_shared_ancestor_is_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            r=Path(temp);leaf=r/'user.slice/user-1008.slice/session.scope';leaf.mkdir(parents=True)
            (r/'cgroup').write_text('0::/user.slice/user-1008.slice/session.scope\n')
            for p,used,high in [(leaf,1,2**60),(leaf.parent,2,2**60),(leaf.parent.parent,100*g.GIB,100*g.GIB)]:
                (p/'memory.current').write_text(str(used))
                (p/'memory.high').write_text('max' if high==2**60 else str(high))
                (p/'memory.max').write_text('max')
            groups=g.memory_groups(r/'cgroup',r)
            self.assertEqual(min(v['headroom'] for v in groups if v['headroom'] is not None),0)
            s=healthy();s['headroom']=0
            self.assertIn('ancestor_headroom_below_24GiB',g.admission_reasons(s))

    def test_v1_unlimited_does_not_invent_headroom(self):
        with tempfile.TemporaryDirectory() as temp:
            r=Path(temp);p=r/'user.slice';p.mkdir()
            (r/'cgroup').write_text('8:memory:/user.slice\n')
            for q in [p,r]:
                (q/'memory.usage_in_bytes').write_text(str(200*g.GIB))
                (q/'memory.limit_in_bytes').write_text(str(2**63-4096))
                (q/'memory.soft_limit_in_bytes').write_text(str(2**63-4096))
            self.assertTrue(all(v['headroom'] is None for v in g.memory_groups(r/'cgroup',v1root=r)))

    def test_healthy_then_pressure(self):
        s=healthy();self.assertFalse(g.admission_reasons(s));self.assertFalse(g.severe(s))
        s['io_psi']={'full':{'avg10':50.}}
        self.assertTrue(g.admission_reasons(s));self.assertTrue(g.severe(s))

    def test_one_cold_load_and_stable_window(self):
        s=dict(healthy(),admission_reasons=[],healthy_since=time.time()-70)
        with patch.object(g,'read',return_value=s),patch.object(g,'active',return_value=[]):
            self.assertTrue(g.can_launch()[0])
        with patch.object(g,'read',return_value=s),patch.object(g,'active',return_value=[{'phase':'loading'}]):
            self.assertIn('another_model_loading',g.can_launch()[1])
        s['healthy_since']=time.time()-1
        with patch.object(g,'read',return_value=s),patch.object(g,'active',return_value=[]):
            self.assertFalse(g.can_launch()[0])


class Ownership(unittest.TestCase):
    def test_recycled_pid_and_foreign_uid_are_never_signaled(self):
        entry=dict(pid=123456,ticks=10,uid=os.getuid(),script='/data/x/code/tex3d/ops/worker.py')
        for current in [dict(pid=123456,ticks=11,uid=os.getuid(),cmd=entry['script']),
                        dict(pid=123456,ticks=10,uid=os.getuid()+1,cmd=entry['script']),
                        dict(pid=123456,ticks=10,uid=os.getuid(),cmd='unrelated.py')]:
            with patch.object(g,'identity',return_value=current), \
                 patch.object(g.os,'kill') as kill, \
                 patch.object(g.os,'pidfd_open',create=True,return_value=999), \
                 patch.object(g.os,'close'), \
                 patch.object(g.signal,'pidfd_send_signal',create=True) as send:
                self.assertFalse(g.matches(entry))
                self.assertFalse(g.stop_entry(entry,15));kill.assert_not_called();send.assert_not_called()

    def test_identity_match(self):
        entry=dict(pid=1,ticks=10,uid=os.getuid(),script='/data/x/code/tex3d/ops/worker.py')
        with patch.object(g,'identity',return_value=dict(entry,cmd=entry['script'])):
            self.assertTrue(g.matches(entry))

    def test_signal_reaches_only_the_registered_child(self):
        marker='/data/x/code/tex3d/ops/harmless_test_child.py'
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)',marker])
        try:
            with tempfile.TemporaryDirectory() as temp,patch.object(g,'BASE',Path(temp)):
                token=g.register(child.pid,marker,'test',0,0,'harmless_test',temp)
                entry=g.read(Path(temp)/'jobs.json')[token]
                self.assertTrue(g.stop_entry(entry,15))
                self.assertEqual(child.wait(timeout=3),-15)
                self.assertFalse(g.matches(entry))
        finally:
            if child.poll() is None:
                child.terminate();child.wait(timeout=3)


class Recovery(unittest.TestCase):
    def test_prefix_is_truncated_at_checkpoint_and_tail_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            r=Path(temp);p=r/'train_20261102_120000_test';p.mkdir()
            rows=['%d %d %d 0 1.000000 0.000000 0.000000 0.000000 1.000000 0.000000'%(i,s,i*2+s) for i in range(6) for s in range(2)]
            expected=r/'stream.txt';expected.write_text('# expected\n'+'\n'.join(rows)+'\n')
            (p/'run_config.txt').write_text(str({'resume_dir':''})+'\n')
            # An interrupted log contains rows beyond the last committed iterate.
            (p/'sampling_log.txt').write_text('# header\n'+'\n'.join(rows[:8])+'\n')
            checkpoint=p/'Ep0_Cloth_Delta_OV_it00002.pt';checkpoint.write_bytes(b'committed')
            (p/'gradient_log.txt').write_text('[ckpt] it00002 frames_accessed=6\n')
            (p/'Ep0_Cloth_Delta_OV_it00005.pt').write_bytes(b'partial_uncommitted')
            self.assertEqual(select_checkpoint(r,'test'),checkpoint)
            q=r/'train_20261102_130000_test';q.mkdir()
            (q/'run_config.txt').write_text(str({'resume_dir':str(p),'resume_checkpoint':str(checkpoint)})+'\n')
            (q/'sampling_log.txt').write_text('# header\n'+'\n'.join(rows[6:])+'\n')
            report=consolidate(q,expected,6,2)
            self.assertTrue(report['exact_prescribed_stream'])
            self.assertEqual((q/'sampling_log.txt').read_text(),'# header\n'+'\n'.join(rows)+'\n')
            self.assertEqual((q/'sampling_log_segment.txt').read_text(),'# header\n'+'\n'.join(rows[6:])+'\n')
            self.assertEqual(consolidate(q,expected,6,2),report)


if __name__=='__main__':
    unittest.main(verbosity=2)
