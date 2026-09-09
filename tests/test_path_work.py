import math

import pytest
import torch

from cfm_mol.path_work import gaussian_training_path,external_energy


def objective(a,b,checkpoint_steps=False):
    g=torch.Generator().manual_seed(117)
    x=torch.randn(40,2,dtype=torch.float64,generator=g)
    path=gaussian_training_path(x,lambda z,t:a*z,lambda z,t:b*z,
        [0,.2,.6,1.],.4,g,checkpoint_steps=checkpoint_steps)
    return path.work(.5*((path.terminal-.3)/.8).square().sum(-1)).mean()


def test_complete_path_gradients_match_fixed_noise_finite_differences():
    a=torch.tensor(.2,dtype=torch.float64,requires_grad=True);b=torch.tensor(-.3,dtype=torch.float64,requires_grad=True)
    gradient=torch.autograd.grad(objective(a,b),(a,b));step=1e-5
    expected=[(objective(a+step,b)-objective(a-step,b))/(2*step),
              (objective(a,b+step)-objective(a,b-step))/(2*step)]
    for actual,reference in zip(gradient,expected):torch.testing.assert_close(actual,reference,rtol=1e-7,atol=1e-8)


def test_checkpoint_reuses_noise_times_and_complete_parameter_paths():
    a=torch.tensor(.2,dtype=torch.float64,requires_grad=True);b=torch.tensor(-.3,dtype=torch.float64,requires_grad=True)
    plain=objective(a,b);pg=torch.autograd.grad(plain,(a,b))
    checked=objective(a,b,True);cg=torch.autograd.grad(checked,(a,b))
    torch.testing.assert_close(plain,checked,rtol=0,atol=0)
    for x,y in zip(pg,cg):torch.testing.assert_close(x,y,rtol=1e-12,atol=1e-12)


def test_exact_gaussian_reverse_kernel_has_constant_work():
    g=torch.Generator().manual_seed(314);x=torch.randn(500,2,dtype=torch.float64,generator=g)
    path=gaussian_training_path(x,lambda z,t:-.2*z,lambda z,t:-.2*z,[0,1],.6,g)
    work=path.work(.5*path.terminal.square().sum(-1)+math.log(2*math.pi))
    torch.testing.assert_close(work,torch.zeros_like(work),rtol=0,atol=2e-14)


def test_external_energy_force_handoff_preserves_parameter_gradient():
    scale=torch.tensor(.7,dtype=torch.float64,requires_grad=True)
    initial=torch.randn(5,3,3,dtype=torch.float64,generator=torch.Generator().manual_seed(21))
    x=initial*scale
    energy=.5*x.detach().square().sum((1,2));force=-x.detach()
    actual,=torch.autograd.grad(external_energy(x,energy,force).sum(),scale)
    expected=scale.detach()*initial.square().sum()
    torch.testing.assert_close(actual,expected)


def test_invalid_oracle_does_not_enter_training():
    x=torch.zeros(2,3,3,requires_grad=True)
    with pytest.raises(ValueError,match='Non-finite'):
        external_energy(x,torch.tensor([float('nan'),0]),torch.zeros_like(x))
