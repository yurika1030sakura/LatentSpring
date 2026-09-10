import math

import torch

from cfm_mol.linear_entropy_adapter import LinearEntropyAdapter, endpoint_kl_change
from cfm_mol.nonequilibrium import centered_orthonormal_basis


def test_adapter_symmetries_spectral_bounds_inverse_and_intrinsic_determinant():
    numbers = torch.tensor([1, 6, 8, 1])
    model = LinearEntropyAdapter(numbers)
    with torch.no_grad():model.raw_weights.copy_(torch.linspace(-2., 2., len(model.raw_weights)))
    transform, logdet, generator = model.matrices()
    basis = centered_orthonormal_basis(4)
    torch.testing.assert_close(transform@torch.ones(4, dtype=torch.float64), torch.ones(4, dtype=torch.float64), atol=1e-12, rtol=0)
    singular = torch.linalg.eigvalsh(transform)
    assert float(singular.min()) >= math.exp(-.25)-1e-12
    assert float(singular.max()) <= math.exp(.25)+1e-12
    sign, direct = torch.linalg.slogdet(basis.T@transform@basis)
    assert sign == 1
    torch.testing.assert_close(logdet, 3*direct, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(torch.matrix_exp(-generator)@transform, torch.eye(4, dtype=torch.float64), atol=1e-12, rtol=1e-12)
    x = torch.randn(5, 4, 3, dtype=torch.float64)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64)); order = torch.tensor([2, 0, 3, 1])
    other = LinearEntropyAdapter(numbers[order]); other.load_state_dict({**other.state_dict(), 'raw_weights': model.raw_weights.detach()})
    result, _ = model(x); changed, _ = other(x[:, order]@rotation+3.)
    torch.testing.assert_close(changed, result[:, order]@rotation+3., atol=1e-12, rtol=1e-12)


def test_gaussian_kl_change_and_parameter_gradient_match_analytic_result():
    model = LinearEntropyAdapter([1, 8], kind='scalar')
    with torch.no_grad():model.raw_weights.fill_(.7)
    basis = centered_orthonormal_basis(2)
    z = torch.cat([torch.eye(3), -torch.eye(3)]).double()*3**.5
    x = torch.einsum('nk,bkd->bnd', basis, z.reshape(6, 1, 3))
    y, logdet = model(x)
    zero = torch.zeros(6, dtype=torch.float64)
    actual = endpoint_kl_change(zero, zero, x, y, kT=1., restraint=1., log_volume=logdet).mean()
    w = .25*torch.tanh(model.raw_weights[0])
    expected = 1.5*(torch.exp(-2*w)-1+2*w)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    gradient, = torch.autograd.grad(actual, model.raw_weights)
    reference, = torch.autograd.grad(expected, model.raw_weights)
    torch.testing.assert_close(gradient, reference, atol=1e-12, rtol=1e-12)
