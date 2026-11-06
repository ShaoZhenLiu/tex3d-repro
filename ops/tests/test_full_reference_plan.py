import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from full_reference_20261105.common import HERE, LONG_ROOT, compatible_reuse, jobs, long_jobs, evaluation_sources
from refcloth_benchmark import create_suites


class FullReferencePlan(unittest.TestCase):
    def test_paper_long_label_resolves_to_libero_ten(self):
        registry = {name: (lambda value=name: value) for name in
                    ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']}
        result = create_suites(registry, ['spatial', 'object', 'goal', 'long'])
        self.assertEqual(result['long'], 'libero_10')
        with self.assertRaisesRegex(KeyError, 'libero_10'):
            create_suites({}, ['long'])

    def test_full_counts_unique_tasks_and_target_specific_assignment(self):
        for target, node in [('pi0', '09'), ('pi05', '4090')]:
            rows = jobs(target)
            self.assertEqual(len(rows), 160)
            self.assertEqual(len({r['id'] for r in rows}), 160)
            self.assertEqual(sum(10*len(r['conditions']) for r in rows), 2800)
            self.assertTrue(all(r['target'] == target and r['node'] == node for r in rows))

    def test_reuse_preserves_suite_noise_indices_and_rejects_changed_checkpoint(self):
        new = json.loads((HERE/'pi05.json').read_text())['evaluation']
        old = dict(new, suites=['spatial', 'object'])
        self.assertTrue(compatible_reuse(old, new))
        self.assertFalse(compatible_reuse(old, dict(new, suites=['goal', 'long', 'spatial', 'object'])))
        self.assertFalse(compatible_reuse(old, dict(new, checkpoint='different-checkpoint')))
        self.assertFalse(compatible_reuse(old, dict(new, num_denoise=5)))
        self.assertFalse(compatible_reuse(old, dict(new, max_steps=300)))

    def test_entire_long_block_is_isolated_on_09_with_original_noise_index(self):
        rows = long_jobs()
        self.assertEqual(len(rows), 40)
        self.assertEqual(sum(10*len(j['conditions']) for j in rows), 700)
        self.assertTrue(all(j['node'] == '09' and j['suite'] == 'long' for j in rows))
        original = json.loads((HERE/'pi05.json').read_text())['evaluation']
        rerun = json.loads((HERE/'pi05_long09.json').read_text())['evaluation']
        self.assertTrue(compatible_reuse(original, rerun))
        self.assertEqual(rerun['suites'].index('long'), 3)
        sources = evaluation_sources('pi05', migrated=True)
        self.assertEqual(sources[0][1], ['spatial', 'object', 'goal'])
        self.assertEqual(sources[1], (LONG_ROOT, ['long'], '09'))
        self.assertEqual([suite for _, suites, _ in sources for suite in suites], original['suites'])


if __name__ == '__main__':
    unittest.main()
