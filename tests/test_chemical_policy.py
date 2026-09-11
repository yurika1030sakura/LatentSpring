import pytest
import torch
from cfm_mol.chemical_policy import ChemicalMovePolicy,accepted_action_mass


def example():
    torch.manual_seed(3101)
    x=torch.randn(2,6,3,dtype=torch.float64)
    bonds=torch.zeros(2,6,6,dtype=x.dtype)
    for i,j in [(0,2),(1,3),(2,3),(3,4),(2,5)]:bonds[:,i,j]=bonds[:,j,i]=1
    numbers=torch.tensor([1,9,6,16,9,1])
    actions=torch.tensor([[[0,1,2,3],[0,4,2,3],[1,5,3,2]]]*2)
    return x,bonds,numbers,torch.tensor([0.,1.,.02585],dtype=x.dtype),actions


def test_uniform_initialization_empty_and_defensive_probabilities():
    model=ChemicalMovePolicy().double();args=example()
    p=model(*args).exp()
    torch.testing.assert_close(p,torch.tensor([[.5,1/6,1/6,1/6]]*2,dtype=p.dtype))
    mask=torch.tensor([[True,False,True],[False,False,False]])
    p=model(*args,mask=mask).exp()
    torch.testing.assert_close(p,torch.tensor([[.5,.25,0,.25],[1,0,0,0]],dtype=p.dtype))
    empty=model(*args[:-1],args[-1][:,:0]).exp()
    torch.testing.assert_close(empty,torch.ones(2,1,dtype=p.dtype))
    with torch.no_grad():
        model.family_head.bias.fill_(100)
        model.action_head[-1].weight.normal_(0,100)
    p=model(*args).exp()
    torch.testing.assert_close(p.sum(1),torch.ones(2,dtype=p.dtype))
    assert bool((p[:,1:]>=.1*.1/3-1e-12).all())
    assert bool((p[:,0]<=.9+1e-12).all())


def test_learned_policy_rigid_and_atom_permutation_symmetry_and_batch_independence():
    model=ChemicalMovePolicy().double();x,b,z,e,a=example()
    with torch.no_grad():
        model.family_head.weight.normal_();model.action_head[-1].weight.normal_()
    expected=model(x,b,z,e,a)
    q=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype))[0]
    q[:,0]*=-1
    torch.testing.assert_close(model(x@q+17,b,z,e,a),expected)
    perm=torch.tensor([3,0,5,2,1,4]);inv=torch.argsort(perm)
    torch.testing.assert_close(model(x[:,perm],b[:,perm][:,:,perm],z[perm],e,inv[a]),expected)
    torch.testing.assert_close(model(x,b,z,e,a[:,:,[1,0,3,2]]),expected)
    order=torch.tensor([2,0,1])
    torch.testing.assert_close(model(x,b,z,e,a[:,order])[:,1:],expected[:,1:][:,order])
    torch.testing.assert_close(model(x[:1],b[:1],z,e,a[:1]),expected[:1])
    with pytest.raises(ValueError):model(x,b,z,e,a-10,mask=torch.ones(a.shape[:2],dtype=torch.bool))


def test_family_and_action_reverse_ratio_stationarity_and_gradient():
    # Three-state chain, two nonlocal involutive actions and one local identity.
    # State-dependent selection alone is generally not reversible.
    theta=torch.tensor([[.7,-1.,.2],[-.2,.5,1.2],[1.1,-.4,.6]],dtype=torch.float64,requires_grad=True)
    pi=torch.tensor([.2,.3,.5],dtype=theta.dtype)
    utility=torch.tensor([[0.,1.,2.],[1.,0.,3.],[2.,3.,0.]],dtype=theta.dtype)
    def objective(t):
        logp=t.log_softmax(1)
        base=(pi.log()[None,:]-pi.log()[:,None])
        mass=accepted_action_mass(logp,logp.T,base)
        return (pi[:,None]*mass*utility).sum(),mass
    loss,mass=objective(theta)
    torch.testing.assert_close(pi[:,None]*mass,pi[None,:]*mass.T)
    transition=mass-torch.diag(mass.diagonal())
    transition=transition+torch.diag(1-transition.sum(1))
    torch.testing.assert_close(pi@transition,pi)
    gradient=torch.autograd.grad(loss,theta)[0]
    eps=1e-5
    for i in range(3):
        for j in range(3):
            delta=torch.zeros_like(theta);delta[i,j]=eps
            finite=(objective(theta.detach()+delta)[0]-objective(theta.detach()-delta)[0])/(2*eps)
            torch.testing.assert_close(gradient[i,j],finite,atol=1e-9,rtol=1e-6)
    assert accepted_action_mass(torch.tensor(0.),torch.tensor(0.),torch.tensor(-torch.inf))==0
