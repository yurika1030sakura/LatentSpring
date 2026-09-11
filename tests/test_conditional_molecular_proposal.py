import math
import pytest
import torch
from cfm_mol.conditional_molecular_proposal import (ConditionalMolecularProposal,center,
    invariant_jump_squared,metropolis_transition)
from cfm_mol.nonequilibrium import centered_orthonormal_basis


def example(n=4,nonlinear=True):
    torch.manual_seed(9951)
    model=ConditionalMolecularProposal(hidden=8,radial=8,features=2,rank=2,
        step_size=.3,nonlinear=nonlinear).double()
    with torch.no_grad():
        for head in [model.node_head,model.group_head,model.vector_head]:
            head.weight.add_(.1*torch.randn_like(head.weight));head.bias.add_(.1*torch.randn_like(head.bias))
    x=center(torch.randn(2,n,3,dtype=torch.float64))
    noise=center(torch.randn_like(x));numbers=torch.full((n,),6,dtype=torch.long)
    electronic=torch.tensor([0.,0.,0.],dtype=torch.float64)
    return model,x,noise,numbers,electronic


@pytest.mark.parametrize('nonlinear',[False,True])
def test_forward_density_matches_inverse_and_complete_intrinsic_jacobian(nonlinear):
    model,x,noise,numbers,electronic=example(nonlinear=nonlinear)
    y,logq=model.transform(x,noise,numbers,electronic)
    torch.testing.assert_close(model.log_prob(y,x,numbers,electronic),logq,atol=1e-9,rtol=1e-9)
    basis=centered_orthonormal_basis(len(numbers));z=(basis.T@noise[0]).flatten().requires_grad_()
    def mapped(z):
        n=basis@z.reshape(-1,3)
        v,_=model.transform(x[:1],n[None],numbers,electronic)
        return (basis.T@v[0]).flatten()
    jac=torch.autograd.functional.jacobian(mapped,z)
    direct=-.5*z.square().sum()-.5*len(z)*math.log(2*math.pi)-torch.linalg.slogdet(jac)[1]
    torch.testing.assert_close(direct,logq[0],atol=1e-9,rtol=1e-9)
    assert float(y.mean(1).abs().max())<1e-12


def test_identity_is_normalized_com_random_walk():
    model=ConditionalMolecularProposal(step_size=.2).double()
    x=center(torch.randn(2,5,3,dtype=torch.float64));noise=center(torch.randn_like(x))
    y,logq=model.transform(x,noise,torch.tensor([1,6,6,8,1]),torch.zeros(3,dtype=torch.float64))
    torch.testing.assert_close(y,x+.2*noise,atol=1e-12,rtol=0)
    expected=-.5*noise.square().sum((1,2))-6*math.log(2*math.pi)-12*math.log(.2)
    torch.testing.assert_close(logq,expected,atol=1e-12,rtol=0)


def test_reflections_permutations_and_batch_independence_with_homogeneous_atoms():
    model,x,noise,numbers,electronic=example()
    y,q=model.transform(x,noise,numbers,electronic)
    rotation=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))[0];rotation[:,0]*=-torch.linalg.det(rotation)
    order=torch.tensor([2,0,3,1])
    transformed,logq=model.transform(x[:,order]@rotation,noise[:,order]@rotation,numbers[order],electronic)
    torch.testing.assert_close(transformed,y[:,order]@rotation,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(logq,q,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(model.log_prob(transformed,x[:,order]@rotation,numbers[order],electronic),q,atol=1e-9,rtol=1e-9)
    alone,aq=model.transform(x[:1],noise[:1],numbers,electronic)
    torch.testing.assert_close(alone,y[:1],atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(aq,q[:1],atol=1e-10,rtol=1e-10)


def test_reverse_density_retains_parameter_and_proposed_context_gradients():
    model,x,noise,numbers,electronic=example()
    def objective():
        y,_=model.transform(x,noise,numbers,electronic)
        return model.log_prob(x,y,numbers,electronic).sum()
    value=objective();parameter=model.node_head.weight
    gradient=torch.autograd.grad(value,parameter)[0]
    index=(0,0);original=float(parameter[index]);h=1e-5
    with torch.no_grad():parameter[index]=original+h
    plus=float(objective())
    with torch.no_grad():parameter[index]=original-h
    minus=float(objective())
    with torch.no_grad():parameter[index]=original
    torch.testing.assert_close(gradient[index],torch.tensor((plus-minus)/(2*h)),atol=2e-5,rtol=2e-4,check_dtype=False)
    y,_=model.transform(x,noise,numbers,electronic)
    context_gradient=torch.autograd.grad(model.log_prob(x,y,numbers,electronic).sum(),y)[0]
    assert float(context_gradient.norm())>1e-5


def test_two_hundred_atoms_and_nonzero_context_heads_reconstruct():
    model,x,noise,numbers,electronic=example(n=200)
    with torch.no_grad():
        y,q=model.transform(x[:1],noise[:1],numbers,electronic)
        torch.testing.assert_close(model.log_prob(y,x[:1],numbers,electronic),q,atol=1e-7,rtol=1e-8)


def test_jump_observable_cannot_reward_rotations_or_same_element_relabelling():
    _,x,_,numbers,_=example()
    rotation=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))[0]
    y=x[:,torch.tensor([2,3,1,0])]@rotation
    assert float(invariant_jump_squared(x,y,numbers).max())<1e-20


def test_metropolized_nonzero_proposal_preserves_known_gaussian_moments():
    model,_,_,numbers,electronic=example()
    generator=torch.Generator().manual_seed(9957)
    x=center(torch.randn(2048,4,3,dtype=torch.float64,generator=generator))
    target=lambda z:-.5*z.square().sum((1,2))
    value=target(x)
    for _ in range(4):
        x,value,info=metropolis_transition(model,x,value,target,numbers,electronic,generator=generator)
        torch.testing.assert_close(value,target(x),atol=1e-12,rtol=0)
        assert 0<float(info['accepted'].double().mean())<1
    assert abs(float(x.square().sum((1,2)).mean())-9)<.35
    assert float(x.mean(0).abs().max())<.08
