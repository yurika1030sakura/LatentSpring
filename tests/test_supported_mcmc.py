import torch
from scipy.special import gammainc
from cfm_mol.fixed_target_mcmc import mala_population
from cfm_mol.supported_mcmc import supported_mala_population
from cfm_mol.tempered_smc import DensityValue


def test_supported_mala_matches_existing_kernel_when_support_is_full():
    x=torch.randn(32,3,dtype=torch.float64,generator=torch.Generator().manual_seed(4))
    target=lambda z:DensityValue(-.5*z.square().sum(-1),-z)
    options=dict(steps=8,proposal_std=.7,max_score_norm=.3)
    old,old_value,_=mala_population(x,target,**options,generator=torch.Generator().manual_seed(5))
    new,new_value,stats=supported_mala_population(x,target,**options,generator=torch.Generator().manual_seed(5))
    torch.testing.assert_close(old,new,atol=0,rtol=0)
    torch.testing.assert_close(old_value.log_value,new_value.log_value,atol=0,rtol=0)
    assert stats['target_evaluations']==32*9


def test_hard_support_rejects_without_physical_evaluation_and_preserves_moment():
    generator=torch.Generator().manual_seed(9857)
    pool=torch.randn(16384,2,dtype=torch.float64,generator=generator)
    x=pool[pool.square().sum(-1)<2.25][:8192]
    valid_evaluations=0
    def target(z):
        nonlocal valid_evaluations
        valid=z.square().sum(-1)<2.25;valid_evaluations+=int(valid.sum())
        logp=torch.full((len(z),),-torch.inf,dtype=z.dtype);score=torch.zeros_like(z)
        logp[valid]=-.5*z[valid].square().sum(-1);score[valid]=-z[valid]
        return DensityValue(logp,score)
    result,value,stats=supported_mala_population(x,target,steps=10,proposal_std=1.,max_score_norm=10.,generator=generator)
    assert torch.isfinite(value.log_value).all() and (result.square().sum(-1)<2.25).all()
    assert valid_evaluations<stats['target_evaluations']==len(x)*11
    expected=2*gammainc(2,2.25/2)/gammainc(1,2.25/2)
    assert abs(float(result.square().sum(-1).mean())-expected)<.04
