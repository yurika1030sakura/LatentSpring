import pytest
import torch

from cfm_mol.path_work import gaussian_training_path
from cfm_mol.path_balance import fixed_path_log_factors,log_variance_balance


@pytest.mark.parametrize('mean',['reference','native'])
@pytest.mark.parametrize('power',[0.,.5])
def test_fixed_path_values_match_simulation_but_scores_hold_observations_fixed(mean,power):
    a=torch.tensor(.2,dtype=torch.float64,requires_grad=True);b=torch.tensor(-.3,dtype=torch.float64,requires_grad=True)
    generator=torch.Generator().manual_seed(1082);x=torch.randn(32,2,dtype=torch.float64,generator=generator)
    times=[0.,.2,.6,1.];settings=dict(terminal_std=2.,max_drift_norm=3.,mean_parameterization=mean,noise_annealing_power=power)
    path=gaussian_training_path(x,lambda z,t:a*z+t*.1,lambda z,t:b*z+t*.05,times,.4,generator,retain_states=True,**settings)
    assert not path.states.requires_grad
    calls=[]
    def evaluate(av,bv):
        def forward(z,t):calls.append('f');return av*z+t[:,None]*.1
        def backward(z,t):calls.append('b');return bv*z+t[:,None]*.05
        initial,factor_f,factor_b=fixed_path_log_factors(path.states,forward,backward,times,.4,**settings)
        return initial+factor_f-factor_b
    actual=evaluate(a,b)
    torch.testing.assert_close(actual,path.log_initial+path.log_forward_minus_backward,rtol=1e-12,atol=1e-10)
    assert calls==['f','b']
    grad=torch.autograd.grad(actual.mean(),(a,b));h=1e-5
    finite=[(evaluate(a+h,b).mean()-evaluate(a-h,b).mean())/(2*h),
            (evaluate(a,b+h).mean()-evaluate(a,b-h).mean())/(2*h)]
    for g,r in zip(grad,finite):torch.testing.assert_close(g,r,rtol=1e-7,atol=1e-7)


def test_balance_is_invariant_to_same_condition_energy_offset():
    values=torch.tensor([1.,2.,4.,8.],dtype=torch.float64,requires_grad=True)
    original=log_variance_balance(values);shifted=log_variance_balance(values+1e5)
    torch.testing.assert_close(original,shifted,rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(original,values)[0],torch.autograd.grad(shifted,values)[0],rtol=0,atol=0)
