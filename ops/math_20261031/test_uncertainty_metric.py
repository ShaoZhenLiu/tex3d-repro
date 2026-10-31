"""Numerical checks of the inner optimizer, conditional bound and gradient."""
import unittest
import torch
from uncertainty_metric import solve_worst_action, robust_action_score


class EllipsoidTests(unittest.TestCase):
    def setUp(self):
        self.shape = dict(values=torch.tensor([0.1, 2.0], dtype=torch.double),
                          vectors=torch.eye(2, dtype=torch.double), radius=0.4)

    def test_isotropic_closed_form(self):
        shape = dict(values=torch.ones(2, dtype=torch.double), vectors=torch.eye(2, dtype=torch.double))
        x = torch.tensor([[3.0, 4.0]], dtype=torch.double)
        value, gradient, _ = solve_worst_action(x, shape, 2.0)
        self.assertAlmostEqual(float(value), 9.0, places=10)
        torch.testing.assert_close(gradient, torch.tensor([[3.6, 4.8]], dtype=torch.double))

    def test_inside_and_zero_radius(self):
        x = torch.tensor([[0.01, -0.01]], dtype=torch.double)
        value, gradient, _ = solve_worst_action(x, self.shape, 1.0)
        self.assertEqual(float(value), 0)
        self.assertEqual(float(gradient.abs().max()), 0)
        value, gradient, _ = solve_worst_action(x, self.shape, 0.0)
        torch.testing.assert_close(value, x.square().sum(-1))
        torch.testing.assert_close(gradient, 2*x)

    def test_constraint_active_and_conditional_bound(self):
        x = torch.tensor([[0.8, 0.4]], dtype=torch.double)
        value, _, info = solve_worst_action(x, self.shape, 0.4)
        torch.testing.assert_close(info['error_mahalanobis_squared'], torch.tensor([0.16], dtype=torch.double))
        generator = torch.Generator().manual_seed(9)
        z = torch.randn(1000, 2, generator=generator, dtype=torch.double)
        z = z/z.norm(dim=-1, keepdim=True).clamp_min(1e-12)*0.4
        errors = z*self.shape['values'].sqrt()
        self.assertLessEqual(float(value), float((x+errors).square().sum(-1).min())+1e-10)

    def test_envelope_gradients_by_finite_difference(self):
        x = torch.tensor([[0.8, 0.4]], dtype=torch.double)
        for relative in [False, True]:
            value, gradient, _ = solve_worst_action(x, self.shape, 0.4, relative=relative)
            for i in range(2):
                delta = torch.zeros_like(x)
                delta[0, i] = 1e-5
                a = solve_worst_action(x+delta, self.shape, 0.4, relative=relative)[0]
                b = solve_worst_action(x-delta, self.shape, 0.4, relative=relative)[0]
                numeric = float((a-b)/(2e-5))
                self.assertAlmostEqual(float(gradient[0, i]), numeric, places=7)

    def test_relative_homogeneity_and_autograd(self):
        x = torch.tensor([[0.8, 0.4]], requires_grad=True)
        model = dict(relative=self.shape)
        score = robust_action_score(x, model, mode='relative')
        score.backward()
        _, gradient, _ = solve_worst_action(x, self.shape, 0.4, relative=True)
        torch.testing.assert_close(x.grad, gradient.float()/2)
        doubled = robust_action_score(2*x, model, mode='relative')
        torch.testing.assert_close(doubled, 4*score)


if __name__ == '__main__':
    unittest.main()
