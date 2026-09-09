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


@pytest.mark.parametrize('steps',[1,3,16])
def test_gaussian_reference_bridge_has_constant_work_for_different_endpoint_widths(steps):
    g=torch.Generator().manual_seed(173);initial=.8;terminal=math.sqrt(10)
    x=torch.randn(500,3,dtype=torch.float64,generator=g)*initial
    path=gaussian_training_path(x,torch_zero,torch_zero,torch.linspace(0,1,steps+1),.4,g,
        prior_std=initial,terminal_std=terminal)
    energy=.5*(path.terminal/terminal).square().sum(-1)+3*math.log(terminal*math.sqrt(2*math.pi))
    torch.testing.assert_close(path.work(energy),torch.zeros(500,dtype=torch.float64),rtol=0,atol=2e-12)


def torch_zero(x,t):return torch.zeros_like(x)


@pytest.mark.parametrize('checked',[False,True])
@pytest.mark.parametrize('mean_parameterization',['reference','native'])
def test_energy_gradient_control_preserves_values_and_backward_gradient(checked,mean_parameterization):
    a=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
    b=torch.tensor(-.3,dtype=torch.float64,requires_grad=True)
    def draw(control):
        g=torch.Generator().manual_seed(447)
        x=torch.randn(40,2,dtype=torch.float64,generator=g)
        return gaussian_training_path(x,lambda z,t:a*torch.tanh(z),lambda z,t:b*torch.tanh(z),
            [0,.2,.6,1.],.4,g,terminal_std=2.,max_drift_norm=3.,
            checkpoint_steps=checked,forward_energy_only=control,mean_parameterization=mean_parameterization)
    joint=draw(False);control=draw(True)
    energy=lambda path:.5*((path.terminal-.3)/.8).square().sum(-1)
    joint_loss=joint.work(energy(joint)).mean();control_loss=control.work(energy(control)).mean()
    torch.testing.assert_close(joint.terminal,control.terminal,rtol=0,atol=0)
    torch.testing.assert_close(joint_loss,control_loss,rtol=0,atol=0)
    joint_grad=torch.autograd.grad(joint_loss,(a,b))
    control_grad=torch.autograd.grad(control_loss,(a,b),retain_graph=True)
    energy_grad,=torch.autograd.grad(energy(control).mean(),a)
    torch.testing.assert_close(control_grad[0],energy_grad,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(control_grad[1],joint_grad[1],rtol=1e-12,atol=1e-12)
    assert abs(float(joint_grad[0]-control_grad[0]))>.01


@pytest.mark.parametrize('mean_parameterization',['reference','native'])
def test_reference_residual_gradient_and_checkpoint_agreement(mean_parameterization):
    a=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
    def loss(value,checked=False):
        g=torch.Generator().manual_seed(247);x=torch.randn(80,2,dtype=torch.float64,generator=g)
        path=gaussian_training_path(x,lambda z,t:value*torch.tanh(z),lambda z,t:-value*torch.tanh(z),
            [0,.3,.7,1.],.3,g,terminal_std=2.,max_drift_norm=3.,checkpoint_steps=checked,
            mean_parameterization=mean_parameterization)
        return path.work(.5*path.terminal.square().sum(-1)/4).mean()
    gradient,=torch.autograd.grad(loss(a),a)
    finite=(loss(a+1e-5)-loss(a-1e-5))/2e-5
    torch.testing.assert_close(gradient,finite,rtol=1e-7,atol=1e-9)
    checked,=torch.autograd.grad(loss(a,True),a)
    torch.testing.assert_close(gradient,checked,rtol=0,atol=1e-12)


def test_native_means_cancel_reference_dilation_and_keep_actual_density_factors():
    from cfm_mol.nonequilibrium import gaussian_log_density
    g=torch.Generator().manual_seed(448);x=torch.randn(40,2,dtype=torch.float64,generator=g)
    state=g.get_state();noise=torch.randn(x.shape,dtype=x.dtype,generator=g);g.set_state(state)
    path=gaussian_training_path(x,lambda z,t:.2*z,lambda z,t:-.3*z,[0,1],.4,g,
        terminal_std=2.,mean_parameterization='native')
    relative=math.sqrt(-math.expm1(-.4**2));mean=1.2*x;y=mean+2*relative*noise
    expected=gaussian_log_density(y,mean,2*relative)-gaussian_log_density(x,.7*y,relative)
    torch.testing.assert_close(path.terminal,y,rtol=0,atol=1e-14)
    torch.testing.assert_close(path.log_forward_minus_backward,expected,rtol=0,atol=1e-12)
