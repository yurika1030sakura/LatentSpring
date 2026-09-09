import math

import torch


def test_gaussian_smoothing_hessian_matches_posterior_covariance_identity():
    means=torch.tensor([[0.,1.],[2.,-.5],[-1.,-1.]],dtype=torch.float64)
    x=torch.tensor([.3,-.4],dtype=torch.float64,requires_grad=True);sigma=.4
    def log_density(value):return torch.logsumexp(-.5*((value-means)/sigma).square().sum(-1),0)-math.log(3*2*math.pi*sigma**2)
    hessian=torch.autograd.functional.hessian(log_density,x)
    weights=torch.softmax(-.5*((x.detach()-means)/sigma).square().sum(-1),0)
    centered=means-(weights[:,None]*means).sum(0)
    covariance=centered.T@(centered*weights[:,None])
    expected=-torch.eye(2)/sigma**2+covariance/sigma**4
    torch.testing.assert_close(hessian,expected,rtol=1e-12,atol=1e-12)
    assert float(torch.linalg.eigvalsh(hessian).min())>=-1/sigma**2-1e-12


def test_fixed_and_annealed_reference_noise_have_temperature_independent_curvature_ceiling():
    kappa=.1;steps=16;noise=.2
    fixed=kappa/(-math.expm1(-noise**2/steps))
    annealed=kappa/(-math.expm1(-noise**2/steps**2))
    assert 40<fixed<41 and 640<annealed<641
    for kT in [.025851999786435,1.]:
        variance=kT/kappa*(-math.expm1(-noise**2/steps))
        assert abs(kT/variance-fixed)<1e-10
