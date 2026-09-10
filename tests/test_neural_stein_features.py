import torch

from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.proposal_score import ProposalEnergyCritic
from cfm_mol.neural_stein_features import neural_head_features, antithetic_divergence, typed_radial_features


def test_neural_features_reconstruct_score_preserve_state_and_transform_correctly():
    torch.manual_seed(9121)
    model = ProposalEnergyCritic(hidden=8, radial=8).double()
    torch.nn.init.normal_(model.readout[-1].weight, std=.2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    basis = centered_orthonormal_basis(4); z = torch.randn(2, 9, dtype=torch.float64)
    numbers = torch.tensor([1, 6, 8, 1]); condition = torch.tensor([.2, 0., -3.6], dtype=torch.float64)
    features, score, _, error = neural_head_features(model, z, basis, numbers, condition)
    assert error < 1e-12
    direct = model.score(z, basis, numbers, condition)
    torch.testing.assert_close(score, torch.einsum('nk,bkd->bnd', basis, direct.reshape(2, 3, 3)))
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    order = torch.tensor([3, 2, 0, 1]); x = torch.einsum('nk,bkd->bnd', basis, z.reshape(2, 3, 3))
    changed = torch.einsum('nk,bnd->bkd', basis, x[:, order]@rotation).flatten(1)
    rotated, _, _, _ = neural_head_features(model, changed, basis, numbers[order], condition)
    torch.testing.assert_close(rotated, features[:, :, order]@rotation, atol=1e-11, rtol=1e-11)
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert len(model.readout[-1]._forward_pre_hooks) == 0


def test_antithetic_identity_is_exact_in_expectation_for_nonlinear_field_at_finite_noise():
    # Standard Gaussian three-point quadrature matches moments through degree5.
    eps = torch.tensor([-3**.5, 0., 3**.5], dtype=torch.float64)
    weights = torch.tensor([1/6, 2/3, 1/6], dtype=torch.float64)
    mean, sigma = .7, .4
    y_plus = mean+sigma*eps; y_minus = mean-sigma*eps
    plus = y_plus.pow(3)[:, None, None, None]; minus = y_minus.pow(3)[:, None, None, None]
    result = antithetic_divergence(plus, minus, eps[:, None, None], sigma, exact_scale=False)
    expected = 3*(mean**2+sigma**2)
    torch.testing.assert_close(weights@result[:, 0], torch.tensor(expected, dtype=torch.float64), atol=1e-13, rtol=0)


def test_typed_features_keep_pair_identity_and_correct_divergence():
    basis = centered_orthonormal_basis(3)
    z = torch.tensor([.3, .5, -.2, .8, -.4, .6], dtype=torch.float64)
    x = (basis@z.reshape(2, 3))[None]; numbers = torch.tensor([1, 8, 1])
    fields, div, names = typed_radial_features(x, numbers, centers=(1.,), width=.5)
    order = torch.tensor([2, 0, 1])
    changed, changed_div, changed_names = typed_radial_features(x[:, order], numbers[order], centers=(1.,), width=.5)
    assert names == changed_names
    torch.testing.assert_close(changed, fields[:, :, order]); torch.testing.assert_close(changed_div, div)
    def potential(v):
        coordinates = basis@v.reshape(2, 3)
        radius = ((coordinates[0]-coordinates[2]).square().sum()+1e-8).sqrt()
        return torch.exp(-.5*((radius-1.)/.5)**2)
    torch.testing.assert_close(div[0, names.index('1_1_1')], torch.autograd.functional.hessian(potential, z).trace(), atol=1e-12, rtol=1e-12)
