import math
import unittest
from fusion_math import four_corner_stats, protected_cone_direction


class AlgebraTests(unittest.TestCase):
    def test_separable_renderer_error_cancels_exactly(self):
        native = {'00':[1,2], '10':[2,3], '01':[4,5], '11':[9,4]}
        # e00=b, e10=b+p, e01=b+q, e11=b+p+q.
        errors = {'00':[2,1], '10':[5,7], '01':[-3,3], '11':[0,9]}
        rendered = {k:[x+y for x,y in zip(native[k],errors[k])] for k in native}
        m = four_corner_stats(rendered, native)
        self.assertAlmostEqual(m['interaction_prediction_error_rms'], 0)
        self.assertGreater(m['joint_prediction_error_rms'], 0)
        self.assertTrue(m['verified_bound'])

    def test_nonseparable_error_does_not_cancel(self):
        native = {'00':[0,0], '10':[0,0], '01':[0,0], '11':[3,4]}
        rendered = dict(native, **{'11':[6,8]})
        m = four_corner_stats(rendered, native)
        self.assertAlmostEqual(m['interaction_relative_error'], 1)
        self.assertTrue(m['verified_bound'])

    def test_additive_action_has_zero_interaction_and_no_ratio(self):
        native = {'00':[1,2], '10':[2,3], '01':[4,5], '11':[5,6]}
        m = four_corner_stats(native, native)
        self.assertEqual(m['source_native_interaction_rms'], 0)
        self.assertIsNone(m['interaction_relative_error'])


class ProjectionTests(unittest.TestCase):
    def assertVector(self, actual, expected):
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=8)

    def test_unconstrained_direction(self):
        self.assertVector(protected_cone_direction([3,4], [], 2), [1.2,1.6])

    def test_single_active_constraint(self):
        self.assertVector(protected_cone_direction([1,1], [[1,0]], 2), [0,2])

    def test_two_active_constraints_can_force_zero(self):
        self.assertVector(protected_cone_direction([1,1], [[1,0],[0,1]], 2), [0,0])

    def test_opposite_constraints_form_a_subspace(self):
        self.assertVector(protected_cone_direction([2,3], [[1,0],[-1,0]], 2), [0,2])

    def test_scaling_constraint_does_not_change_direction(self):
        a = protected_cone_direction([1,-1], [[1,0],[1,1]], 1)
        b = protected_cone_direction([1,-1], [[800,0],[.02,.02]], 1)
        self.assertVector(a, b)

    def test_kkt_solution_beats_feasible_angular_search(self):
        g, hs = [1.3, .7], [[1,0],[.2,1]]
        d = protected_cone_direction(g, hs, 1)
        optimum = sum(x*y for x,y in zip(d,g))
        for i in range(10000):
            v = [math.cos(i*2*math.pi/10000), math.sin(i*2*math.pi/10000)]
            if all(sum(x*y for x,y in zip(v,h)) <= 1e-12 for h in hs):
                self.assertGreaterEqual(optimum+1e-8, sum(x*y for x,y in zip(v,g)))


if __name__ == '__main__':
    unittest.main()
