import pytest
import torch

from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.species_coupling_adapter import SpeciesCouplingAdapter
from cfm_mol.affine_species_adapter import AffineSpeciesCouplingAdapter


@pytest.fixture(params=[SpeciesCouplingAdapter, AffineSpeciesCouplingAdapter])
def model_class(request):
    return request.param


def model_for(numbers, seed=9161, model_class=SpeciesCouplingAdapter):
    torch.manual_seed(seed)
    model = model_class(numbers, charge=1, spin_multiplicity=1, kT=.025852, hidden=8, radial=8).double()
    with torch.no_grad():
        for head in [model.conditioner.node_head, model.conditioner.group_head]:
            head.weight.normal_(std=.12); head.bias.normal_(std=.05)
    return model


@pytest.mark.parametrize('numbers', [[1, 1], [1, 8], [1, 6, 1, 8], [1, 15, 8, 6, 1, 6, 8, 12]])
def test_full_context_jacobian_and_inverse(numbers, model_class):
    model = model_for(numbers, model_class=model_class)
    n = len(numbers); basis = centered_orthonormal_basis(n)
    z = torch.randn(3*(n-1), dtype=torch.float64)
    x = (basis@z.reshape(n-1, 3))[None]
    y, volume = model(x)
    def transform(v):
        out, _ = model((basis@v.reshape(n-1, 3))[None])
        return (basis.T@out[0]).flatten()
    jacobian = torch.autograd.functional.jacobian(transform, z)
    sign, reference = torch.linalg.slogdet(jacobian)
    assert sign == 1
    torch.testing.assert_close(volume[0], reference, atol=1e-9, rtol=1e-9)
    restored, inverse_volume, diagnostic = model.inverse(y)
    torch.testing.assert_close(restored, x, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(inverse_volume, -volume, atol=1e-9, rtol=1e-9)
    assert diagnostic['maximum_residual'] < 1e-9
    for layer in model.layers:
        changed, _, _ = model.apply_layer(x, layer)
        torch.testing.assert_close(model.split_context(x, layer)[0], model.split_context(changed, layer)[0], atol=1e-12, rtol=1e-12)


def test_joint_permutation_rotation_batch_independence_and_identity(model_class):
    numbers = torch.tensor([1, 6, 1, 8, 6])
    model = model_for(numbers, model_class=model_class); x = torch.randn(3, 5, 3, dtype=torch.float64); x -= x.mean(1, keepdim=True)
    y, volume = model(x)
    separate = [model(row[None]) for row in x]
    torch.testing.assert_close(y, torch.cat([row[0] for row in separate]), atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(volume, torch.cat([row[1] for row in separate]), atol=1e-11, rtol=1e-11)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64)); order = torch.tensor([4, 0, 3, 1, 2])
    other = model_for(numbers[order], model_class=model_class); state = model.state_dict(); state['numbers'] = numbers[order]
    other.load_state_dict(state)
    transformed, transformed_volume = other(x[:, order]@rotation)
    torch.testing.assert_close(transformed, y[:, order]@rotation, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(transformed_volume, volume, atol=1e-10, rtol=1e-10)
    identity = model_class(numbers, charge=1, spin_multiplicity=1, kT=.025852, hidden=8, radial=8).double()
    same, zero = identity(x)
    torch.testing.assert_close(same, x, atol=1e-13, rtol=0)
    torch.testing.assert_close(zero, torch.zeros_like(zero), atol=1e-13, rtol=0)


def test_parameter_gradient_through_conditioner_and_volume(model_class):
    model = model_for([1, 6, 1, 8], model_class=model_class)
    x = torch.randn(2, 4, 3, dtype=torch.float64); x -= x.mean(1, keepdim=True)
    parameter = model.conditioner.messages[0][0].weight
    direction = torch.randn_like(parameter)
    def objective():
        y, volume = model(x)
        return y.square().mean()-.2*volume.mean()
    gradient, = torch.autograd.grad(objective(), parameter)
    original = parameter.detach().clone(); h = 1e-5
    with torch.no_grad():parameter.copy_(original+h*direction)
    plus = objective().detach()
    with torch.no_grad():parameter.copy_(original-h*direction)
    minus = objective().detach()
    with torch.no_grad():parameter.copy_(original)
    torch.testing.assert_close((gradient*direction).sum(), (plus-minus)/(2*h), atol=1e-7, rtol=1e-6)


def test_two_hundred_atoms_complete_model_inverse(model_class):
    numbers = [1]*100+[6]*100
    model = model_for(numbers, model_class=model_class)
    x = torch.randn(1, 200, 3, dtype=torch.float64); x -= x.mean(1, keepdim=True)
    with torch.no_grad():
        y, volume = model(x)
        restored, inverse_volume, _ = model.inverse(y)
    torch.testing.assert_close(restored, x, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(volume+inverse_volume, torch.zeros_like(volume), atol=1e-8, rtol=0)
