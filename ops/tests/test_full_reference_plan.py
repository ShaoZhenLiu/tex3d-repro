import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from full_reference_20261105.common import HERE, compatible_reuse, jobs
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


if __name__ == '__main__':
    unittest.main()
