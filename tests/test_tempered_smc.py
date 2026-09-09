import math

import pytest
import torch

from cfm_mol.nonequilibrium import normalized_weights
from cfm_mol.tempered_smc import (DensityValue,IsotropicGaussianMixture,
    metropolis_log_acceptance,tempered_smc)


def gaussian(x,mean=0.,std=1.,offset=0.):
    return DensityValue(-.5*((x-mean)/std).square().sum(-1)-x.shape[-1]*math.log(std*math.sqrt(2*math.pi))+offset,
                        -(x-mean)/std**2)


def test_mixture_score_against_autodiff_and_distribution_reference():
    centers=torch.tensor([[-2.,.3],[1.,-.2],[3.,2.]],dtype=torch.float64)
    mixture=IsotropicGaussianMixture(centers,.7)
    x=torch.randn(19,2,dtype=torch.float64,generator=torch.Generator().manual_seed(1)).requires_grad_()
    actual=mixture(x)
    normal=torch.distributions.Independent(torch.distributions.Normal(centers,.7),1)
    ref=torch.logsumexp(normal.log_prob(x[:,None,:]),-1)-math.log(3)
    torch.testing.assert_close(actual.log_value,ref)
    grad,=torch.autograd.grad(actual.log_value.sum(),x)
    torch.testing.assert_close(actual.score,grad)


def test_asymmetric_proposal_detailed_balance_pointwise():
    # Independent of any sampler trajectory; deliberately nonconservative drift.
    g=torch.Generator().manual_seed(2)
    x=torch.randn(300,2,dtype=torch.float64,generator=g)
    y=torch.randn(300,2,dtype=torch.float64,generator=g)
    def mean(z):return .9*z+.25*z.flip(-1)*z.new_tensor([1.,-1.])
    mx,my=mean(x),mean(y);std=.6
    normal=lambda value,mu:torch.distributions.Independent(torch.distributions.Normal(mu,std),1).log_prob(value)
    logx,logy=gaussian(x).log_value,gaussian(y).log_value
    ratio=metropolis_log_acceptance(x,y,logx,logy,mx,my,std)
    forward_flux=logx+normal(y,mx)+ratio.clamp_max(0)
    reverse_flux=logy+normal(x,my)+(-ratio).clamp_max(0)
    torch.testing.assert_close(forward_flux,reverse_flux,rtol=1e-13,atol=1e-13)
    wrong=logy-logx
    assert (logx+normal(y,mx)+wrong.clamp_max(0)-logy-normal(x,my)-(-wrong).clamp_max(0)).abs().max()>1


@pytest.mark.parametrize('kernel',['rwm','mala'])
@pytest.mark.parametrize('threshold',[0.,.8])
def test_tempered_gaussian_target_and_normalizer(kernel,threshold):
    g=torch.Generator().manual_seed(440)
    x=torch.randn(12000,1,dtype=torch.float64,generator=g)
    out=tempered_smc(x,gaussian,lambda z:gaussian(z,2.,.7,1.2),torch.linspace(0,1,33),
        proposal_std=.8,generator=g,kernel=kernel,resample_threshold=threshold)
    w=normalized_weights(out.log_weights)
    assert abs(float(w@out.positions[:,0])-2)<.04
    assert abs(float(w@(out.positions[:,0]-2).square())-.49)<.04
    assert abs(out.log_normalizer_estimate-1.2)<.05
    assert out.target_evaluations==12000*33
    assert len(out.history)==32
    if threshold==0:assert out.summary()['distinct_initial_ancestors']==12000
    else:assert out.summary()['resampling_events']>0


def test_constant_energy_offset_changes_only_normalizer():
    x=torch.randn(300,2,dtype=torch.float64,generator=torch.Generator().manual_seed(3))
    kwargs=dict(betas=[0,.25,.5,1.],proposal_std=.3,kernel='mala',resample_threshold=.5)
    left=tempered_smc(x,gaussian,lambda z:gaussian(z,.6),generator=torch.Generator().manual_seed(4),**kwargs)
    right=tempered_smc(x,gaussian,lambda z:gaussian(z,.6,offset=20000),generator=torch.Generator().manual_seed(4),**kwargs)
    torch.testing.assert_close(left.positions,right.positions,rtol=0,atol=0)
    torch.testing.assert_close(left.log_weights,right.log_weights,rtol=0,atol=1e-10)
    assert right.log_normalizer_estimate-left.log_normalizer_estimate==pytest.approx(20000)


def test_missing_force_is_not_silently_used_as_mala():
    with pytest.raises(ValueError,match='finite score'):
        tempered_smc(torch.zeros(2,1),lambda z:DensityValue(torch.zeros(2)),gaussian,[0,1],
            proposal_std=.2,generator=torch.Generator())
