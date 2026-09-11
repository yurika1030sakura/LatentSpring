import math

import pytest
import torch

from cfm_mol.fixed_target_mcmc import mala_population,hmc_population,budgeted_population
from cfm_mol.tempered_smc import DensityValue


@pytest.mark.parametrize('kT',[1.,.025851999786435])
def test_clipped_mala_preserves_gaussian_moments_and_counts_rejected_queries(kT):
    generator=torch.Generator().manual_seed(9056)
    initial=torch.randn(8192,2,dtype=torch.float64,generator=generator)*math.sqrt(kT)
    queried=0
    def target(x):
        nonlocal queried
        queried+=len(x)
        return DensityValue(-.5*x.square().sum(-1)/kT,-x/kT)
    final,value,stats=mala_population(initial,target,steps=30,proposal_std=1.2*math.sqrt(kT),
        max_score_norm=.4/kT,generator=generator)
    assert queried==stats['target_evaluations']==len(initial)*31
    assert .1<stats['acceptance_fraction']<.9
    assert float(final.mean(0).abs().max())/math.sqrt(kT)<.04
    assert float((final.var(0)/kT-1).abs().max())<.06
    torch.testing.assert_close(value.log_value,-.5*final.square().sum(-1)/kT)
    torch.testing.assert_close(value.score,-final/kT)


def test_zero_moves_does_not_relabel_arbitrary_starts_as_equilibrium():
    x=torch.full((4,2),10.)
    final,_,stats=mala_population(x,lambda z:DensityValue(-.5*z.square().sum(-1),-z),
        steps=0,proposal_std=.1,generator=torch.Generator().manual_seed(9))
    torch.testing.assert_close(final,x.double())
    assert stats['target_evaluations']==4 and stats['acceptance_fraction'] is None
    assert stats['endpoint_density'].startswith('unknown')


def test_hmc_clipped_kicks_keep_target_moments_and_count_every_leapfrog_query():
    generator=torch.Generator().manual_seed(315)
    initial=torch.randn(8192,2,dtype=torch.float64,generator=generator)
    target=lambda x:DensityValue(-.5*x.square().sum(-1),-x)
    final,value,stats=hmc_population(initial,target,leapfrog_counts=[7]*20,step_size=.3,
        max_score_norm=.5,generator=generator)
    assert stats['target_evaluations']==len(initial)*(1+7*20)
    assert .1<stats['acceptance_fraction']<.95
    assert float(final.mean(0).abs().max())<.04
    assert float((final.var(0)-1).abs().max())<.06
    torch.testing.assert_close(value.log_value,-.5*final.square().sum(-1))
    torch.testing.assert_close(value.score,-final)


def test_hmc_reuses_an_explicit_initial_value_without_charging_an_oracle_call():
    initial=torch.ones(8,2,dtype=torch.float64);queried=[]
    def target(z):
        queried.append(len(z));return DensityValue(-.5*z.square().sum(-1),-z)
    cached=target(initial);queried.clear()
    _,_,stats=hmc_population(initial,target,leapfrog_counts=[3],step_size=.1,
        generator=torch.Generator().manual_seed(10),initial_value=cached)
    assert sum(queried)==stats['target_evaluations']==24


@pytest.mark.parametrize('kernel',['mala','hmc'])
def test_budgeted_population_replays_proposals_and_cached_subset_without_extra_queries(kernel):
    initial=torch.randn(16,5,dtype=torch.float64,generator=torch.Generator().manual_seed(8))
    trace=[]
    def target(z):
        value=DensityValue(-.5*z.square().sum(-1),-z)
        trace.append((z.clone(),value.log_value.clone(),value.score.clone()))
        return value
    kwargs=dict(kernel=kernel,force_updates=4,tail_indices=torch.arange(0,16,2),
                step_size=.4,max_score_norm=.3,seed=27,hmc_length=4)
    first,value,stats=budgeted_population(initial,target,**kwargs)
    assert stats['target_evaluations']==sum(len(row[0]) for row in trace)==16*5+8
    assert len(trace)==6 and len(trace[-1][0])==8
    iterator=iter(trace)
    def replay(z):
        x,log_value,score=next(iterator)
        torch.testing.assert_close(z,x,atol=0,rtol=0)
        return DensityValue(log_value,score)
    second,replayed,other=budgeted_population(initial,replay,**kwargs)
    assert list(iterator)==[] and other==stats
    torch.testing.assert_close(first,second,atol=0,rtol=0)
    torch.testing.assert_close(value.log_value,replayed.log_value,atol=0,rtol=0)
    torch.testing.assert_close(value.score,-first,atol=0,rtol=0)
    with pytest.raises(ValueError,match='unique subset'):
        budgeted_population(initial,target,**{**kwargs,'tail_indices':torch.tensor([0,0])})
