import math

import torch

from cfm_mol.tempered_smc import DensityValue,IsotropicGaussianMixture,tempered_smc


def test_hmc_smc_recovers_shifted_gaussian_normalizer_moments_and_exact_query_budget():
    generator=torch.Generator().manual_seed(9069)
    initial=IsotropicGaussianMixture(torch.zeros(1,2,dtype=torch.float64),1.)
    x,_=initial.sample(4096,generator);calls=0
    mean=torch.tensor([1.,-.5],dtype=torch.float64);variance=.3
    def target(z):
        nonlocal calls
        calls+=len(z)
        return DensityValue(-.5*(z-mean).square().sum(-1)/variance-11.,-(z-mean)/variance)
    result=tempered_smc(x,initial,target,torch.linspace(0,1,17,dtype=torch.float64),
        proposal_std=.2,generator=generator,kernel='hmc',hmc_leapfrog_steps=5,
        hmc_step_size_schedule=lambda beta:.2/math.sqrt(1+beta),resample_threshold=.5)
    assert calls==result.target_evaluations==4096*(1+16*5)
    assert abs(result.log_normalizer_estimate-(math.log(2*math.pi*variance)-11))<.04
    weights=torch.softmax(result.log_weights,0)
    torch.testing.assert_close(weights@result.positions,mean,atol=.035,rtol=0)
    actual=weights@(result.positions-mean).square()
    torch.testing.assert_close(actual,torch.full((2,),variance,dtype=torch.float64),atol=.035,rtol=0)
    # Cached bridge extraction must retain the actual terminal potential.
    torch.testing.assert_close(result.target_log_values,target(result.positions).log_value,atol=1e-10,rtol=1e-12)
