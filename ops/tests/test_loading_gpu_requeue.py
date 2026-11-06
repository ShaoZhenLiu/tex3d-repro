import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refcloth_resource as guard


class LoadingGpuRequeue(unittest.TestCase):
    def test_only_cold_evaluation_is_withdrawn_when_other_allocations_consume_reservation(self):
        cold = dict(pid=101, gpu=2, kind='eval', phase='loading')
        warm = dict(pid=102, gpu=2, kind='eval', phase='warm')
        source = dict(pid=103, gpu=2, kind='source', phase='loading')
        other_gpu = dict(pid=104, gpu=1, kind='eval', phase='loading')
        rows = [cold, warm, source, other_gpu]
        self.assertEqual(guard.cold_eval_cancellations(rows, {2:-5624,1:19614}), [cold])
        self.assertEqual(guard.cold_eval_cancellations(rows, {2:500,1:19614}), [])
        self.assertEqual(guard.cold_eval_cancellations(rows, {}), [])


if __name__ == '__main__':
    unittest.main()
