import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reference_coverage_20261106.common import HERE, SEEDS, jobs
from full_reference_20261105.common import compatible_reuse


class ReferenceCoveragePlan(unittest.TestCase):
    def test_all_eighteen_starts_and_every_task_are_in_both_suites(self):
        self.assertEqual(len(SEEDS),18)
        for suite,node in [('goal','4090'),('long','09')]:
            rows=jobs(suite)
            self.assertEqual(len(rows),190)
            self.assertEqual(len({r['id'] for r in rows}),190)
            self.assertEqual(sum(10*len(r['conditions']) for r in rows),3700)
            self.assertTrue(all(r['suite']==suite and r['node']==node for r in rows))
            self.assertEqual({r['task'] for r in rows},set(range(10)))

    def test_suite_order_keeps_the_original_query_noise_indices(self):
        original=json.loads((HERE.parent/'full_reference_20261105/pi05.json').read_text())['evaluation']
        for suite in ['goal','long']:
            current=json.loads((HERE/(suite+'.json')).read_text())['evaluation']
            self.assertTrue(compatible_reuse(original,current))
            self.assertEqual(current['suites'].index(suite),2 if suite=='goal' else 3)


if __name__=='__main__':unittest.main()
