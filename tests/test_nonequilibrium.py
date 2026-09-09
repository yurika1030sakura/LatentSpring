"""Independent finite-state / Gaussian checks of work and teacher weighting."""
import itertools
import math

import pytest
import torch

from cfm_mol.nonequilibrium import (centered_orthonormal_basis, gaussian_path_sample, normalized_weights,
    path_log_weight, random_walk_ais, weighted_flow_matching_loss)


def test_path_identity_by_exhaustive_enumeration():
    # Arbitrary non-reversible kernels: no equilibrium / small-step assumption.
    q0 = torch.tensor([.3, .7], dtype=torch.float64)
    gamma = torch.tensor([1.4, .6], dtype=torch.float64)
    forward = [torch.tensor([[.8,.2],[.4,.6]], dtype=torch.float64),
               torch.tensor([[.1,.9],[.7,.3]], dtype=torch.float64)]
    backward = [torch.tensor([[.6,.4],[.2,.8]], dtype=torch.float64),
                torch.tensor([[.3,.7],[.9,.1]], dtype=torch.float64)]
    initial=[]; terminal=[]; lk=[]; ll=[]; probabilities=[]; endpoints=[]
    for x0,x1,x2 in itertools.product(range(2), repeat=3):
        initial.append(q0[x0].log()); terminal.append(gamma[x2].log())
        lk.append(torch.stack([forward[0][x0,x1].log(),forward[1][x1,x2].log()]))
        ll.append(torch.stack([backward[0][x1,x0].log(),backward[1][x2,x1].log()]))
        probabilities.append(q0[x0]*forward[0][x0,x1]*forward[1][x1,x2]); endpoints.append(x2)
    weights = path_log_weight(torch.stack(initial),torch.stack(terminal),torch.stack(lk).T,torch.stack(ll).T).exp()
    masses = torch.zeros(2,dtype=torch.float64)
    masses.scatter_add_(0,torch.tensor(endpoints),torch.stack(probabilities)*weights)
    torch.testing.assert_close(masses,gamma,atol=1e-14,rtol=1e-14)
    wrong = (torch.stack(terminal)-torch.stack(initial)).exp()
    wrong_mass = torch.zeros_like(masses).scatter_add_(0,torch.tensor(endpoints),torch.stack(probabilities)*wrong)
    assert not torch.allclose(wrong_mass,gamma,atol=.01)


def test_exact_gaussian_reverse_has_constant_work_even_at_one_step():
    # Forward x1=a*x0+noise, x0~N(0,1); use the exact conditional x0|x1.
    a=.8; noise=.6; variance=a*a+noise*noise  # stationary marginal variance 1
    g=torch.Generator().manual_seed(77)
    x=torch.randn(4000,2,dtype=torch.float64,generator=g)
    logq=lambda z: -.5*z.square().sum(-1)-math.log(2*math.pi)
    result=gaussian_path_sample(x,logq,lambda z:logq(z)+1.7,
        lambda z,t:(a-1)*z,lambda z,t:(a/variance-1)*z,[0,1],noise,g)
    torch.testing.assert_close(result.log_weights,torch.full((len(x),),1.7,dtype=torch.float64),atol=2e-14,rtol=0)
    assert result.summary()['ess_fraction']==pytest.approx(1)


def test_ais_recovers_gaussian_normalizer_and_moment():
    g=torch.Generator().manual_seed(19)
    x=torch.randn(12000,1,dtype=torch.float64,generator=g)
    logq=lambda z:-.5*z[:,0].square()-.5*math.log(2*math.pi)
    logtarget=lambda z:-.5*((z[:,0]-1.5)/.8).square()-math.log(.8*math.sqrt(2*math.pi))+.7
    result=random_walk_ais(x,logq,logtarget,torch.linspace(0,1,33),.8,g)
    weight=normalized_weights(result.log_weights)
    assert abs(float(weight@result.positions[:,0])-1.5)<.035
    assert abs(float(weight@(result.positions[:,0]-1.5).square())-.64)<.035
    assert abs(result.summary()['log_mean_weight']-.7)<.035
    assert 0 < result.summary()['ess_fraction'] <= 1


def test_ais_constant_target_offset_changes_only_work():
    x=torch.tensor([[-1.],[0.],[1.]],dtype=torch.float64)
    logq=lambda z:-.5*z[:,0].square()-.5*math.log(2*math.pi)
    a=random_walk_ais(x,logq,logq,[0,.2,1],.5,torch.Generator().manual_seed(13))
    b=random_walk_ais(x,logq,lambda z:logq(z)+1000,[0,.2,1],.5,torch.Generator().manual_seed(13))
    torch.testing.assert_close(a.positions,b.positions,atol=0,rtol=0)
    torch.testing.assert_close(b.log_weights-a.log_weights,torch.full((3,),1000.,dtype=torch.float64))


def test_weighted_cfm_optimum_tracks_target_mass_not_particle_count():
    theta=torch.tensor(0.,requires_grad=True)
    pred=theta.expand(2,1)
    target=torch.tensor([[-1.],[1.]])
    logw=torch.tensor([.2,.8]).log().requires_grad_()
    loss=weighted_flow_matching_loss(pred,target,logw)
    grad,=torch.autograd.grad(loss,theta)
    torch.testing.assert_close(grad,torch.tensor(-1.2))
    assert logw.grad is None


def test_molecular_basis_preserves_measure_and_gaussian_dimension():
    basis=centered_orthonormal_basis(7)
    torch.testing.assert_close(basis.T@basis,torch.eye(6,dtype=torch.float64))
    torch.testing.assert_close(basis@basis.T,torch.eye(7,dtype=torch.float64)-torch.ones(7,7,dtype=torch.float64)/7)
    latent=torch.randn(11,6,3,dtype=torch.float64,generator=torch.Generator().manual_seed(32))
    positions=torch.einsum('nk,bkd->bnd',basis,latent)
    torch.testing.assert_close(positions.sum(1),torch.zeros(11,3,dtype=torch.float64),atol=1e-14,rtol=0)
    torch.testing.assert_close(positions.square().sum((1,2)),latent.square().sum((1,2)))


@pytest.mark.parametrize('bad',[torch.tensor([float('nan'),0.]),torch.tensor([float('inf'),0.]),torch.tensor([-float('inf'),-float('inf')])])
def test_nonfinite_weight_guards(bad):
    with pytest.raises(ValueError): normalized_weights(bad)


def test_reject_invalid_schedule_and_nonfinite_oracle():
    x=torch.zeros(2,1)
    logq=lambda z:-z[:,0].square()
    with pytest.raises(ValueError,match='increase strictly'):
        random_walk_ais(x,logq,logq,[0,1,.5,1],.5,torch.Generator())
    with pytest.raises(ValueError,match='finite log densities'):
        random_walk_ais(x,logq,lambda z:torch.full((len(z),),float('nan')),[0,1],.5,torch.Generator())
