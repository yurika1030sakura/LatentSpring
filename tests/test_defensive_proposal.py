import math

import torch

from cfm_mol.defensive_proposal import DefensiveProposal
from cfm_mol.tempered_smc import IsotropicGaussianMixture,tempered_smc


def test_tail_weights_are_bounded_where_narrow_proposal_has_infinite_variance():
    local=IsotropicGaussianMixture(torch.zeros(1,2,dtype=torch.float64),.3)
    proposal=DefensiveProposal(local,2,math.sqrt(10),.2)
    radii=torch.linspace(0,1000,300,dtype=torch.float64)
    x=torch.stack([radii,torch.zeros_like(radii)],-1)
    target=proposal.wide(x).log_value
    logw=target-proposal(x).log_value
    assert float(logw.max())<=math.log(5)+1e-9
    assert float((2*target-local(x).log_value)[-1])>1e6


def test_mixture_score_matches_autodiff():
    local=IsotropicGaussianMixture(torch.tensor([[1.,2.],[-1.,.3]],dtype=torch.float64),.4)
    proposal=DefensiveProposal(local,2,3.,.2)
    x=torch.randn(20,2,dtype=torch.float64,generator=torch.Generator().manual_seed(3)).requires_grad_()
    value=proposal(x);gradient,=torch.autograd.grad(value.log_value.sum(),x)
    torch.testing.assert_close(value.score,gradient,rtol=1e-12,atol=1e-12)


def test_sampler_realizes_declared_mixture_and_labels():
    local=IsotropicGaussianMixture(torch.zeros(1,2,dtype=torch.float64),.3)
    proposal=DefensiveProposal(local,2,2.,.25)
    x,labels=proposal.sample(100000,torch.Generator().manual_seed(224))
    assert abs(float((labels==-1).double().mean())-.25)<.005
    expected=.75*.3**2+.25*2**2
    torch.testing.assert_close((x*x).mean(0),torch.full((2,),expected,dtype=torch.float64),rtol=0,atol=.025)


def test_tempered_normalizer_product_obeys_defensive_bound():
    local=IsotropicGaussianMixture(torch.zeros(1,2,dtype=torch.float64),.3)
    proposal=DefensiveProposal(local,2,2.,.2)
    generator=torch.Generator().manual_seed(443)
    x,_=proposal.sample(4096,generator)
    result=tempered_smc(x,proposal,proposal.wide,[0,.01,.1,.5,1],proposal_std=.5,generator=generator)
    assert result.log_normalizer_estimate<=math.log(5)+1e-12
    assert abs(result.log_normalizer_estimate)<.1
