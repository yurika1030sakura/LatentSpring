import pytest
import torch

from cfm_mol.centered_convex_flow import convex_point_map, centered_convex_forward, centered_convex_inverse
from cfm_mol.nonequilibrium import centered_orthonormal_basis


def context(batch=1, features=5):
    return (torch.randn(batch, features, dtype=torch.float64),
        torch.randn(batch, features, 3, dtype=torch.float64), torch.randn(batch, features, dtype=torch.float64),
        torch.randn(batch, dtype=torch.float64), torch.randn(batch, dtype=torch.float64))


@pytest.mark.parametrize('atoms', [2, 4, 7])
def test_full_intrinsic_jacobian_matches_centered_block_determinant(atoms):
    torch.manual_seed(9149+atoms)
    basis = centered_orthonormal_basis(atoms); z = torch.randn(3*(atoms-1), dtype=torch.float64)
    parameters = context()
    x = (basis@z.reshape(atoms-1, 3))[None]
    y, logdet = centered_convex_forward(x, *parameters)
    def transform(z):
        out, _ = centered_convex_forward((basis@z.reshape(atoms-1, 3))[None], *parameters)
        return (basis.T@out[0]).flatten()
    jacobian = torch.autograd.functional.jacobian(transform, z)
    sign, direct = torch.linalg.slogdet(jacobian)
    assert sign == 1
    torch.testing.assert_close(logdet[0], direct, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(jacobian, jacobian.T, atol=1e-11, rtol=1e-11)
    _, blocks, scale = convex_point_map(x, *parameters)
    assert float(torch.linalg.eigvalsh(blocks).min()) >= .75*float(scale)-1e-12
    assert float(torch.linalg.eigvalsh(blocks).max()) <= 1.25*float(scale)+1e-12
    inverse, diagnostic = centered_convex_inverse(y, *parameters, tolerance=1e-11)
    torch.testing.assert_close(inverse, x, atol=2e-11, rtol=0)
    assert diagnostic['maximum_residual'] < 1e-11


def test_symmetries_identity_and_max_atoms_inverse():
    torch.manual_seed(9153)
    x = torch.randn(1, 200, 3, dtype=torch.float64)*10.; x -= x.mean(1, keepdim=True)
    parameters = context()
    y, logdet = centered_convex_forward(x, *parameters)
    restored, _ = centered_convex_inverse(y, *parameters, tolerance=1e-10)
    torch.testing.assert_close(restored, x, atol=2e-10, rtol=0)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64)); order = torch.randperm(200)
    changed_parameters = (parameters[0], parameters[1]@rotation, *parameters[2:])
    changed, new_logdet = centered_convex_forward(x[:, order]@rotation, *changed_parameters)
    torch.testing.assert_close(changed, y[:, order]@rotation, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(logdet, new_logdet, atol=1e-10, rtol=1e-10)
    identity_parameters = (torch.zeros_like(parameters[0]), parameters[1], parameters[2],
        torch.zeros_like(parameters[3]), torch.zeros_like(parameters[4]))
    same, zero_logdet = centered_convex_forward(x, *identity_parameters)
    torch.testing.assert_close(same, x, atol=1e-13, rtol=0)
    torch.testing.assert_close(zero_logdet, torch.zeros_like(zero_logdet), atol=1e-13, rtol=0)


def test_complete_parameter_gradient_through_map_and_exact_volume():
    torch.manual_seed(9154)
    x = torch.randn(2, 4, 3, dtype=torch.float64); x -= x.mean(1, keepdim=True)
    parameters = list(context(batch=2)); parameters[0].requires_grad_(True)
    def objective(weights):
        y, logdet = centered_convex_forward(x, weights, *parameters[1:])
        return y.square().mean()-.17*logdet.mean()
    gradient, = torch.autograd.grad(objective(parameters[0]), parameters[0])
    direction = torch.randn_like(parameters[0]); h = 1e-5
    fd = (objective(parameters[0].detach()+h*direction)-objective(parameters[0].detach()-h*direction))/(2*h)
    torch.testing.assert_close((gradient*direction).sum(), fd, atol=1e-8, rtol=1e-7)
