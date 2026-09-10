import math

import pytest
import torch

from cfm_mol.innovation_posterior import (GaussianInnovationMap, GaussianAuxiliary,
    fit_gauss_newton_auxiliary, innovation_log_joint)
from cfm_mol.path_balance import fixed_path_log_factors
from cfm_mol.path_work import gaussian_training_path


@pytest.mark.parametrize('mean', ['native', 'reference'])
@pytest.mark.parametrize('power', [0., .5])
def test_nonlinear_path_reparameterization_preserves_endpoint_density_and_gradients(mean, power):
    drift = lambda x, t: .2*torch.sin(x)+torch.as_tensor(t, dtype=x.dtype, device=x.device)[..., None]*.1
    grid = [0., .2, .6, 1.]
    settings = dict(prior_std=.8, terminal_std=.5, max_drift_norm=2.,
                    mean_parameterization=mean, noise_annealing_power=power)
    gen = torch.Generator().manual_seed(1234)
    x0 = torch.randn(7, 2, dtype=torch.float64, generator=gen)*settings['prior_std']
    path = gaussian_training_path(x0, drift, drift, grid, .3, gen, retain_states=True, **settings)
    mapping = GaussianInnovationMap(drift, 2, grid, .3, **settings)
    z = mapping.from_states(path.states).detach().requires_grad_(True)
    final_mean, states = mapping(z, retain_states=True)
    torch.testing.assert_close(states, path.states[:, :-1], atol=1e-13, rtol=1e-13)
    q0, logf, _ = fixed_path_log_factors(path.states, drift, drift, grid, .3, **settings)
    logjoint = innovation_log_joint(z, path.terminal, final_mean, mapping.terminal_noise_std)
    torch.testing.assert_close(logjoint, q0+logf+mapping.log_state_jacobian, atol=1e-12, rtol=1e-12)
    gradient = torch.autograd.grad(final_mean.sum(), z)[0]
    h = 1e-5
    for j in range(z.shape[1]):
        shift = torch.zeros_like(z); shift[:, j] = h
        fd = (mapping(z.detach()+shift)-mapping(z.detach()-shift)).sum(-1)/(2*h)
        torch.testing.assert_close(gradient[:, j], fd, atol=1e-8, rtol=1e-7)


def test_exact_linear_gaussian_posterior_removes_all_auxiliary_weight_noise():
    # More latent than endpoint dimensions; direct multivariate-normal reference.
    matrix = torch.tensor([[1., .3, -.2, 0., .1], [.1, -.2, .7, 1., .2]], dtype=torch.float64)
    offset = torch.tensor([.2, -.4], dtype=torch.float64)
    sigma = .25
    generator = torch.Generator().manual_seed(321)
    z = torch.randn(24, 5, dtype=torch.float64, generator=generator)
    function = lambda u: u@matrix.T+offset
    y = function(z)+sigma*torch.randn(24, 2, dtype=torch.float64, generator=generator)
    posterior, diagnostic = fit_gauss_newton_auxiliary(function, y, 5, sigma, iterations=1)
    marginal = torch.distributions.MultivariateNormal(offset, covariance_matrix=matrix@matrix.T+sigma**2*torch.eye(2))
    weights = marginal.log_prob(y)+posterior.log_prob(z)-innovation_log_joint(z, y, function(z), sigma)
    torch.testing.assert_close(weights, torch.zeros_like(weights), atol=1e-11, rtol=0)
    assert max(diagnostic['posterior_gradient_norm']) < 1e-11
    precision = torch.eye(5)+matrix.T@matrix/sigma**2
    reference = torch.distributions.MultivariateNormal(posterior.mean, precision_matrix=precision.expand(24, 5, 5))
    torch.testing.assert_close(posterior.log_prob(z), reference.log_prob(z), atol=1e-11, rtol=1e-11)
    # Same endpoints, wrong auxiliary precision: weight fluctuations return.
    wrong = marginal.log_prob(y)+posterior.log_prob(z, metric_scale=0.)-innovation_log_joint(z, y, function(z), sigma)
    assert float(wrong.std()) > .5


def test_nonlinear_auxiliary_is_endpoint_only_and_line_search_does_not_increase_objective():
    function = lambda z: torch.stack([z[:, 0]+.1*z[:, 1]**2, torch.sin(z[:, 1])+.3*z[:, 2]], -1)
    y = torch.tensor([[.2, .4], [-.1, -.2]], dtype=torch.float64)
    first, diagnostic = fit_gauss_newton_auxiliary(function, y, 3, .2)
    second, _ = fit_gauss_newton_auxiliary(function, y.flip(0), 3, .2)
    torch.testing.assert_close(first.mean, second.mean.flip(0), atol=0, rtol=0)
    torch.testing.assert_close(first.jacobian, second.jacobian.flip(0), atol=0, rtol=0)
    for row in diagnostic['history']:
        assert all(a <= b for a, b in zip(row['objective_after'], row['objective_before']))
    z = torch.zeros_like(first.mean)
    expected = -.5*3*math.log(2*math.pi)-.5*first.mean.square().sum(-1)
    torch.testing.assert_close(first.log_prob(z, metric_scale=0.), expected)
