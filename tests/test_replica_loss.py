import itertools
import pytest
import torch

from cfm_mol.replica_loss import grouped_replica_residual


def test_product_removes_noise_bias_and_matches_exact_expected_gradient():
    # Enumerate both independent Rademacher variables, not a noisy Monte Carlo
    # assertion. Noise amplitude depends on a parameter, so the squared loss
    # has a biased GRADIENT as well as a biased value.
    a = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    base = a*torch.tensor([-1., 0., 1.], dtype=torch.float64)
    noise = a*torch.tensor([1., -2., 1.], dtype=torch.float64)
    energy = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    pid = torch.zeros(3, dtype=torch.long)
    exact, _ = grouped_replica_residual(base, energy, pid)
    products, squares = [], []
    for s1, s2 in itertools.product([-1, 1], repeat=2):
        q = torch.stack([base+s1*noise, base+s2*noise])
        value, diag = grouped_replica_residual(q, energy, pid, estimator='replica_product')
        square, _ = grouped_replica_residual(q, energy, pid)
        assert float(square-value) == pytest.approx(diag['trace_noise_penalty'], abs=1e-12)
        products.append(value); squares.append(square)
    product = torch.stack(products).mean()
    square = torch.stack(squares).mean()
    assert product.item() == pytest.approx(exact.item(), abs=1e-12)
    assert square.item() > exact.item()
    g_exact = torch.autograd.grad(exact, a, retain_graph=True)[0]
    g_product = torch.autograd.grad(product, a, retain_graph=True)[0]
    g_square = torch.autograd.grad(square, a)[0]
    assert g_product.item() == pytest.approx(g_exact.item(), abs=1e-12)
    assert g_square.item() > g_exact.item()


def test_offsets_equal_parent_weighting_and_negative_values():
    q = torch.tensor([[1., -1., 2., -2., 0.], [-1., 1., -2., 2., 0.]])
    groups = torch.tensor([0, 0, 7, 7, 7])
    e = torch.zeros(5, dtype=torch.float64)
    loss, diag = grouped_replica_residual(q, e, groups, estimator='replica_product')
    assert loss.item() == pytest.approx((-1.-8./3.)/2)
    offset = torch.tensor([1e10, 1e10, -1e10, -1e10, -1e10], dtype=torch.float64)
    shifted, _ = grouped_replica_residual(q, e+offset, groups, estimator='replica_product')
    assert shifted.item() == loss.item()
    assert diag['n_groups_used'] == 2


def test_invalid_geometries_excluded_jointly_and_bad_temperature_rejected():
    q = torch.tensor([[1., float('nan'), 3.], [2., 4., 6.]], requires_grad=True)
    e = torch.zeros(3); groups = torch.zeros(3, dtype=torch.long)
    loss, diag = grouped_replica_residual(q, e, groups, estimator='replica_product')
    assert loss.item() == 2.
    assert diag['n_valid'] == 2
    loss.backward()
    assert torch.isfinite(q.grad).all()
    with pytest.raises(ValueError, match='kT'):
        grouped_replica_residual(q, e, groups, kT=0)
