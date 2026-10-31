import unittest
import torch

from gate_audit import candidate_passes
from diagnose import crossed_masks, fit_transport, pca_basis, projection_metrics
from local_transport import transport


class GateTests(unittest.TestCase):
    def setUp(self):
        self.thresholds=dict(max_q_err=.5,min_q_signal=.8,min_group_q_signal=.6)

    def test_absent_group_and_error_are_not_signal_constraints(self):
        c=dict(q_err=.4,q_signal=.9,**{'by_amp:lo.q_signal':None,
            'by_task:0.q_signal':.7,'by_task:0.q_err':.2})
        self.assertTrue(candidate_passes(c,self.thresholds))

    def test_present_bad_group_rejected(self):
        self.assertFalse(candidate_passes(dict(q_err=.4,q_signal=.9,
            **{'by_task:0.q_signal':.59}),self.thresholds))

    def test_invalid_global_stat_rejected(self):
        for value in [None,float('nan'),float('inf'),.51]:
            self.assertFalse(candidate_passes(dict(q_err=value,q_signal=.9),self.thresholds))


class DiagnosticTests(unittest.TestCase):
    def test_crossed_split_disjoint_and_complete(self):
        rows=[dict(step=step,texture_seed=seed) for step in [0,60] for seed in [101,102]]
        masks=crossed_masks(rows)
        self.assertTrue(torch.equal(sum(mask.int() for mask in masks.values()),torch.ones(4,dtype=torch.int)))
        self.assertEqual([int(mask.sum()) for mask in masks.values()],[1]*4)

    def test_transport_recovers_heldout_channel_scaling(self):
        g=torch.Generator().manual_seed(4)
        train=torch.randn(10,56,4096,generator=g)
        test=torch.randn(5,56,4096,generator=g)
        weights=torch.linspace(.2,1.8,4096).reshape(1,1,-1)
        fitted=fit_transport(train,train*weights,'channel')
        torch.testing.assert_close(test*fitted,test*weights,rtol=1e-5,atol=1e-5)

    def test_projection_preserves_orthogonal_signal(self):
        residual=torch.tensor([[1.,0.,0.],[2.,0.,0.]])
        native=torch.tensor([[0.,1.,0.],[0.,2.,0.]])
        basis=pca_basis(residual,1)
        metrics=projection_metrics(native+residual,native,basis)
        self.assertLess(metrics['q_err'],1e-10)
        self.assertAlmostEqual(metrics['q_signal'],1.)

    def test_secant_transport_corrects_probe_span_and_preserves_complement(self):
        probes=torch.tensor([[1.,0.,0.],[0.,1.,0.]])
        native=torch.tensor([[2.,1.,0.],[1.,3.,0.]])
        query=torch.tensor([[2.,3.,7.]])
        expected=torch.tensor([[7.,11.,7.]])
        torch.testing.assert_close(transport(query,probes,native,ridge=0),expected)
        torch.testing.assert_close(transport(torch.zeros_like(query),probes,native),torch.zeros_like(query))

    def test_secant_transport_device_and_gradient(self):
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
        probes=torch.tensor([[1.,0.,0.],[0.,1.,0.]],device=device)
        native=torch.tensor([[2.,1.,0.],[1.,3.,0.]],device=device)
        query=torch.tensor([[2.,3.,7.]],device=device,requires_grad=True)
        result=transport(query,probes,native,ridge=0)
        self.assertEqual(result.device,query.device)
        result.sum().backward()
        torch.testing.assert_close(query.grad,torch.tensor([[3.,4.,1.]],device=device))


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
