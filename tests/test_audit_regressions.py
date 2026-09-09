"""Regressions missed by the original mocks: zero traces and skipped groups."""
import torch

from cfm_mol.bgfm_loss import divergence_exact_atomwise, divergence_hutchinson
from cfm_mol.bgfm_density import within_group_variance_loss


def test_last_coordinate_exact_trace_retains_parameter_gradient():
    x = torch.randn(1, 3, requires_grad=True)
    a = torch.tensor(0.4, requires_grad=True)
    v = lambda z: a*z
    trace = divergence_exact_atomwise(v, x, torch.tensor([1]))
    assert torch.autograd.grad(trace.sum(), a)[0].item() == 3.0


def test_disconnected_constant_field_has_zero_divergence():
    x = torch.randn(3, 3, requires_grad=True)
    vf = lambda z: torch.ones_like(z)
    counts = torch.tensor([1, 2])
    assert torch.equal(divergence_exact_atomwise(vf, x, counts), torch.zeros(2))
    assert torch.equal(divergence_hutchinson(vf, x, counts), torch.zeros(2))


def test_all_invalid_residuals_have_finite_zero_loss_and_zero_gradient():
    x = torch.tensor([float('nan'), float('inf')], requires_grad=True)
    loss, diag = within_group_variance_loss(x, torch.tensor([0, 0]))
    assert loss.item() == 0 and diag['n_groups_used'] == 0
    loss.backward()
    assert torch.equal(x.grad, torch.zeros(2))


def test_only_singletons_plus_nan_have_finite_zero_loss():
    x = torch.tensor([1.0, float('nan'), 2.0], requires_grad=True)
    loss, diag = within_group_variance_loss(x, torch.tensor([0, 0, 1]))
    assert loss.item() == 0 and diag['n_groups_used'] == 0
    loss.backward()
    assert torch.equal(x.grad, torch.zeros(3))
