"""Analytic tests of execution geometry, LP bounds and sign-induced failure."""
import unittest
from box_maximin import (solve_box_maximin, update_box, mean_sign,
                         scaled_pcgrad, normalize_rows)


def gain(g, step):
    return sum(x*y for x, y in zip(g, step))


class GeometryTests(unittest.TestCase):
    def test_mean_and_pcgrad_sign_can_destroy_common_ascent(self):
        gradients = [[1.0, 0.0], [-2.0, 1.0]]
        lower, upper = [-1.0, -1.0], [1.0, 1.0]
        average = mean_sign(gradients, lower, upper)
        projected_sign = scaled_pcgrad(gradients, lower, upper, take_sign=True)
        projected_fractional = scaled_pcgrad(gradients, lower, upper, take_sign=False)
        self.assertLess(gain(gradients[0], average), 0)
        self.assertLess(gain(gradients[1], projected_sign), 0)
        self.assertGreater(min(gain(g, projected_fractional) for g in gradients), 0)
        result = solve_box_maximin(gradients, lower, upper, method='mw', iterations=16000)
        self.assertAlmostEqual(result['objective'], 1.0/3, delta=.008)
        self.assertGreaterEqual(result['dual_upper']+1e-10, 1.0/3)
        self.assertLessEqual(result['objective'], 1.0/3+1e-10)
        self.assertLess(result['duality_gap'], .01)
        self.assertTrue(all(l-1e-12 <= x <= u+1e-12 for x, l, u in zip(result['delta'], lower, upper)))

    def test_l1_normalization_does_not_fix_mean_sign(self):
        rows, _ = normalize_rows([[1.0, 0.0], [-2.0, 1.0]], mode='l1')
        step = mean_sign(rows, [-1, -1], [1, 1])
        self.assertLess(min(gain(g, step) for g in rows), 0)
        result = solve_box_maximin(rows, [-1, -1], [1, 1], method='mw', iterations=16000)
        self.assertAlmostEqual(result['objective'], .2, delta=.008)

    def test_opposed_gradients_cannot_have_strict_common_ascent(self):
        result = solve_box_maximin([[1.0], [-1.0]], [-1.0], [1.0], method='mw')
        self.assertAlmostEqual(result['objective'], 0)
        self.assertAlmostEqual(result['dual_upper'], 0)

    def test_zero_group_is_not_dropped(self):
        result = solve_box_maximin([[1.0, 1.0], [0.0, 0.0]], [-1, -1], [1, 1], method='mw')
        self.assertLessEqual(result['objective'], 0)

    def test_executable_box_accounts_for_parameter_clip(self):
        lower, upper = update_box([5.0, -5.0, 0.0], .2)
        self.assertEqual(lower, [-.2, 0.0, -.2])
        self.assertEqual(upper, [0.0, .2, .2])
        result = solve_box_maximin([[1.0, -1.0, 1.0]], lower, upper, method='mw')
        self.assertAlmostEqual(result['objective'], .2)

    def test_fixed_uncertainty_offsets_shift_guaranteed_gain(self):
        result = solve_box_maximin([[1.0], [2.0]], [-1], [1], offsets=[-.2, -.7], method='mw')
        self.assertAlmostEqual(result['objective'], .8)

    def test_nonfinite_input_rejected(self):
        with self.assertRaises(ValueError):
            solve_box_maximin([[float('nan')]], [-1], [1], method='mw')


if __name__ == '__main__':
    unittest.main(verbosity=2)
