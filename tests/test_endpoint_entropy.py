import torch

from cfm_mol.endpoint_entropy import denoising_score_loss, endpoint_kl_surrogate


def quadrature():
    # Exact independent first/second moments for this Gaussian quadratic check.
    z = torch.tensor([[-1.], [-1.], [1.], [1.]], dtype=torch.float64)
    eps = torch.tensor([[-1.], [1.], [-1.], [1.]], dtype=torch.float64)
    return z, eps


def test_endpoint_gradient_matches_true_kl_finite_difference_without_score_parameter_gradient():
    z, eps = quadrature()
    a = torch.tensor(.8, dtype=torch.float64, requires_grad=True)
    sigma, target_variance = .2, .25
    critic = torch.tensor(1/(float(a.detach())**2+sigma**2), dtype=torch.float64, requires_grad=True)
    y = a*z+sigma*eps
    proxy = endpoint_kl_surrogate(y, -critic*y, -y/target_variance)
    gradient, critic_gradient = torch.autograd.grad(proxy, [a, critic], allow_unused=True)
    assert critic_gradient is None
    def true_kl(v):
        ratio = (v*v+sigma**2)/target_variance
        return .5*(ratio-1-torch.log(torch.as_tensor(ratio, dtype=torch.float64)))
    h = 1e-5
    finite_difference = (true_kl(float(a)+h)-true_kl(float(a)-h))/(2*h)
    torch.testing.assert_close(gradient, finite_difference, rtol=1e-8, atol=1e-8)


def test_actual_noise_dsm_has_known_gaussian_conditional_optimum():
    z, eps = quadrature()
    sigma, a = .2, .8
    y = a*z+sigma*eps
    coefficient = torch.tensor(1/(a*a+sigma*sigma), dtype=torch.float64, requires_grad=True)
    gradient = torch.autograd.grad(denoising_score_loss(-coefficient*y, eps, sigma), coefficient)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-14, rtol=0)
