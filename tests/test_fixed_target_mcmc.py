import torch

from cfm_mol.fixed_target_mcmc import mala_population
from cfm_mol.tempered_smc import DensityValue


def test_clipped_mala_preserves_gaussian_moments_and_counts_rejected_queries():
    generator=torch.Generator().manual_seed(9056)
    initial=torch.randn(8192,2,dtype=torch.float64,generator=generator)
    queried=0
    def target(x):
        nonlocal queried
        queried+=len(x)
        return DensityValue(-.5*x.square().sum(-1),-x)
    final,value,stats=mala_population(initial,target,steps=30,proposal_std=1.2,
        max_score_norm=.4,generator=generator)
    assert queried==stats['target_evaluations']==len(initial)*31
    assert .1<stats['acceptance_fraction']<.9
    assert float(final.mean(0).abs().max())<.04
    assert float((final.var(0)-1).abs().max())<.06
    torch.testing.assert_close(value.log_value,-.5*final.square().sum(-1))
    torch.testing.assert_close(value.score,-final)


def test_zero_moves_does_not_relabel_arbitrary_starts_as_equilibrium():
    x=torch.full((4,2),10.)
    final,_,stats=mala_population(x,lambda z:DensityValue(-.5*z.square().sum(-1),-z),
        steps=0,proposal_std=.1,generator=torch.Generator().manual_seed(9))
    torch.testing.assert_close(final,x.double())
    assert stats['target_evaluations']==4 and stats['acceptance_fraction'] is None
    assert stats['endpoint_density'].startswith('unknown')
