import math

import pytest
import torch

from cfm_mol.fixed_target_mcmc import mala_population
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
