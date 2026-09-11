import math
import torch
from cfm_mol.spherical_proposal import vmf_log_prob,vmf_sample,force_vmf_parameter


def test_normalization_uniform_limit_density_gradient_and_sampler_moments():
    torch.manual_seed(125)
    # Axisymmetric integration is a one-dimensional independent quadrature.
    t=torch.linspace(-1,1,20001,dtype=torch.float64)
    u=torch.stack([(1-t*t).clamp_min(0).sqrt(),torch.zeros_like(t),t],1)
    for k in [0.,1e-6,1.,10.]:
        eta=torch.tensor([0.,0.,k],dtype=t.dtype).expand(len(t),-1)
        integral=2*math.pi*torch.trapezoid(vmf_log_prob(u,eta).exp(),t)
        assert abs(float(integral)-1)<1e-6
    eta=torch.zeros(2,3,dtype=torch.float64,requires_grad=True);directions=torch.tensor([[1.,0.,0.],[0.,0.,1.]],dtype=eta.dtype)
    gradient=torch.autograd.grad(vmf_log_prob(directions,eta).sum(),eta)[0]
    torch.testing.assert_close(gradient,directions)
    g=torch.Generator().manual_seed(125)
    samples,_=vmf_sample(torch.tensor([0.,0.,4.],dtype=t.dtype).expand(16384,-1),generator=g)
    expected=1/math.tanh(4)-.25
    assert abs(float(samples[:,2].mean())-expected)<.015
    assert float(samples[:,:2].mean(0).abs().max())<.015


def test_force_parameters_and_density_are_rotation_reflection_equivariant():
    torch.manual_seed(126);u=torch.randn(8,3,dtype=torch.float64);u=u/u.norm(dim=1,keepdim=True)
    force=torch.randn_like(u);q=torch.linalg.qr(torch.randn(3,3,dtype=u.dtype))[0];q[:,0]*=-1
    eta=force_vmf_parameter(u,force,radius=1.4,kT=.02585,concentration=10.)
    transformed=force_vmf_parameter(u@q,force@q,radius=1.4,kT=.02585,concentration=10.)
    torch.testing.assert_close(transformed,eta@q)
    torch.testing.assert_close(vmf_log_prob(u,eta),vmf_log_prob(u@q,transformed))


def test_force_guided_angular_mh_preserves_a_known_vmf_target():
    rng=torch.Generator().manual_seed(127);n=8192
    target=torch.tensor([0.,0.,4.],dtype=torch.float64).expand(n,-1)
    x,_=vmf_sample(target,generator=rng)
    for _ in range(16):
        labels=torch.tensor([1.,10.,100.],dtype=x.dtype)[torch.randint(3,(n,),generator=rng)]
        eta=force_vmf_parameter(x,target,radius=1.,kT=1.,concentration=labels)
        y,_=vmf_sample(eta,generator=rng)
        reverse=force_vmf_parameter(y,target,radius=1.,kT=1.,concentration=labels)
        ratio=((y-x)*target).sum(1)+vmf_log_prob(x,reverse)-vmf_log_prob(y,eta)
        take=torch.rand(n,dtype=x.dtype,generator=rng).log()<ratio.clamp_max(0)
        x=torch.where(take[:,None],y,x)
    expected=1/math.tanh(4)-.25
    assert abs(float(x[:,2].mean())-expected)<.02
    assert abs(float(x[:,2].square().mean())-(1-expected/2))<.025
    assert float(x[:,:2].mean(0).abs().max())<.02
