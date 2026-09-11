import pytest
import torch

from cfm_mol.equivariant_pair_adapter import EquivariantPairAdapter
from cfm_mol.nonequilibrium import centered_orthonormal_basis


def model_for(numbers,sweeps=2,seed=9581,affine=False):
    torch.manual_seed(seed)
    model=EquivariantPairAdapter(numbers,charge=0,spin_multiplicity=1,kT=1.,sweeps=sweeps,hidden=8,affine=affine).double()
    with torch.no_grad():
        model.pair_head[-1].weight.normal_(std=.3);model.pair_head[-1].bias.normal_(std=.2)
        model.scale_head.weight.normal_(std=.1)
    return model


def coordinates(n,batch=2):
    x=torch.randn(batch,n,3,dtype=torch.float64);return x-x.mean(1,keepdim=True)


@pytest.mark.parametrize('numbers',[[6]*4,[1,8],[1,6,6,8]])
@pytest.mark.parametrize('affine',[False,True])
def test_intrinsic_jacobian_inverse_and_gradient(numbers,affine):
    model=model_for(numbers,affine=affine);n=len(numbers);basis=centered_orthonormal_basis(n)
    z=torch.randn(3*(n-1),dtype=torch.float64);x=(basis@z.reshape(n-1,3))[None]
    y,volume=model(x)
    def function(v):return (basis.T@model((basis@v.reshape(n-1,3))[None])[0][0]).flatten()
    sign,reference=torch.linalg.slogdet(torch.autograd.functional.jacobian(function,z))
    assert sign==1
    torch.testing.assert_close(volume[0],reference,atol=1e-10,rtol=1e-10)
    restored,inv_volume,diagnostic=model.inverse(y)
    torch.testing.assert_close(restored,x,atol=1e-9,rtol=1e-9)
    torch.testing.assert_close(inv_volume,-volume,atol=1e-9,rtol=1e-9)
    parameter=model.pair_head[-1].weight;direction=torch.randn_like(parameter)
    loss=y.square().mean()-.2*volume.mean()
    gradient,=torch.autograd.grad(loss,parameter)
    original=parameter.detach().clone();h=1e-5
    with torch.no_grad():parameter.copy_(original+h*direction)
    yp,vp=model(x)
    with torch.no_grad():parameter.copy_(original-h*direction)
    ym,vm=model(x)
    with torch.no_grad():parameter.copy_(original)
    finite_difference=(yp.square().mean()-.2*vp.mean()-ym.square().mean()+.2*vm.mean())/(2*h)
    torch.testing.assert_close((gradient*direction).sum(),finite_difference,atol=1e-8,rtol=1e-6)


def test_homogeneous_permutation_rotation_and_collective_motion():
    model=model_for([6]*8)
    with torch.no_grad():model.scale_head.weight.zero_();model.scale_head.bias.zero_()
    x=coordinates(8);y,volume=model(x)
    order=torch.tensor([4,1,6,3,0,5,2,7])
    rotation,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))
    transformed,tv=model(x[:,order]@rotation)
    torch.testing.assert_close(transformed,y[:,order]@rotation,atol=1e-11,rtol=1e-11)
    torch.testing.assert_close(tv,volume,atol=1e-11,rtol=1e-11)
    mirrored,mv=model(-x)
    torch.testing.assert_close(mirrored,-y,atol=1e-11,rtol=1e-11)
    torch.testing.assert_close(mv,volume,atol=1e-11,rtol=1e-11)
    assert float((y[:,:4].mean(1)-x[:,:4].mean(1)).abs().max())>1e-5
    assert float(y.mean(1).abs().max())<1e-12
    separate=[model(row[None]) for row in x]
    torch.testing.assert_close(y,torch.cat([v[0] for v in separate]),atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(volume,torch.cat([v[1] for v in separate]),atol=1e-12,rtol=1e-12)


def test_joint_element_permutation_and_identity():
    numbers=torch.tensor([1,6,1,8,6]);model=model_for(numbers);x=coordinates(5)
    order=torch.tensor([2,4,3,1,0]);other=model_for(numbers[order]);state=model.state_dict()
    state['numbers']=numbers[order];other.load_state_dict(state)
    y,v=model(x);yp,vp=other(x[:,order])
    torch.testing.assert_close(yp,y[:,order],atol=1e-11,rtol=1e-11)
    torch.testing.assert_close(vp,v,atol=1e-11,rtol=1e-11)
    identity=EquivariantPairAdapter(numbers,charge=0,spin_multiplicity=1,kT=1.).double()
    same,zero=identity(x)
    torch.testing.assert_close(same,x,atol=0,rtol=0)
    torch.testing.assert_close(zero,torch.zeros_like(zero),atol=0,rtol=0)


def test_two_hundred_atoms_roundtrip():
    model=model_for([6]*200,sweeps=1);x=coordinates(200,batch=1)
    with torch.no_grad():y,volume=model(x);back,inv_volume,_=model.inverse(y)
    torch.testing.assert_close(back,x,atol=1e-9,rtol=1e-9)
    torch.testing.assert_close(volume+inv_volume,torch.zeros_like(volume),atol=1e-9,rtol=0)


@pytest.mark.parametrize('direction',[-1,1])
def test_extreme_coefficients_attain_the_declared_curvature_bound(direction):
    import math
    model=model_for([6]*4,sweeps=1)
    with torch.no_grad():
        model.pair_head[-1].weight.zero_();model.pair_head[-1].bias.fill_(direction*1000.)
        model.scale_head.weight.zero_();model.scale_head.bias.fill_(1000.)
    x=torch.zeros(1,4,3,dtype=torch.float64)
    y,volume=model(x)
    # At coincident points, each radial Hessian is I/ell. The complete graph
    # Laplacian has eigenvalue N on H, giving the exact extremal spectrum.
    expected=9*(model.scale_bound+math.log(1+direction*model.curvature_bound))
    torch.testing.assert_close(volume,volume.new_tensor([expected]),atol=1e-11,rtol=1e-11)
    assert torch.isfinite(y).all()


def test_capacity_change_preserves_initial_parameter_gradients():
    models=[]
    for bound in [.25,.75]:
        torch.manual_seed(9581)
        models.append(EquivariantPairAdapter([6]*4,charge=0,spin_multiplicity=1,kT=1.,
            hidden=8,sweeps=2,curvature_bound=bound).double())
    x=coordinates(4)
    gradients=[]
    for model in models:
        y,volume=model(x)
        loss=(y.square().sum((1,2))-volume).mean()
        gradients.append(torch.autograd.grad(loss,tuple(model.parameters())))
    for left,right in zip(*gradients):
        torch.testing.assert_close(left,right,atol=1e-12,rtol=1e-12)


def test_larger_curvature_bound_inverse_near_its_worst_case():
    model=EquivariantPairAdapter([6]*4,charge=0,spin_multiplicity=1,kT=1.,
        hidden=8,sweeps=1,curvature_bound=.75).double()
    with torch.no_grad():model.pair_head[-1].bias.fill_(-1000.)
    x=1e-3*coordinates(4)
    y,volume=model(x)
    restored,iv,diagnostics=model.inverse(y,tolerance=1e-13)
    torch.testing.assert_close(restored,x,atol=1e-11,rtol=1e-11)
    torch.testing.assert_close(iv,-volume,atol=1e-10,rtol=1e-10)
    assert diagnostics['iterations'][0]>64
