import json
from pathlib import Path
import pytest
import torch
from cfm_mol.geometry_recovery_field import GeometryRecoveryField,denoising_targets,GeometryThenPhysical
from cfm_mol import matched_egnn as base
from cfm_mol.physical_connection import PhysicalConnection
from cfm_mol.matched_physical_connection import PhysicalFieldTransform


@pytest.mark.parametrize('mode',['radial','moments'])
def test_vector_field_symmetry_centering_bound_and_gradients(mode):
    torch.manual_seed(76101)
    model=GeometryRecoveryField([1,6,8],mode=mode).double()
    with torch.no_grad():model.network[-1].weight.normal_(std=.3)
    x=base.center(torch.randn(3,6,3,dtype=torch.float64));z=torch.tensor([[6,6,8,1,1,1]]*3)
    t=torch.tensor([0.,.6,.95],dtype=x.dtype)
    value=model(x,z,t);q,_=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype));p=torch.tensor([3,5,1,0,2,4])
    transformed=model(x[:,p]@q+2,z[:,p],t)
    torch.testing.assert_close(transformed,value[:,p]@q,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(value.mean(1),torch.zeros(3,3,dtype=x.dtype),atol=1e-12,rtol=0)
    assert value[0].eq(0).all() and (value.norm(dim=-1)<=4*t[:,None]**2+1e-12).all()
    value.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_recovery_labels_are_centered_bounded_and_reproducible():
    x=base.center(torch.randn(32,7,3,dtype=torch.float64));t=torch.linspace(.4,.95,32,dtype=x.dtype)
    a=denoising_targets(x,t,torch.Generator().manual_seed(76102))
    b=denoising_targets(x,t,torch.Generator().manual_seed(76102))
    torch.testing.assert_close(a[0],b[0],atol=0,rtol=0);torch.testing.assert_close(a[1],b[1],atol=0,rtol=0)
    torch.testing.assert_close(a[1].mean(1),torch.zeros(32,3,dtype=x.dtype),atol=1e-12,rtol=0)
    assert (a[1].norm(dim=-1)<=4*t[:,None]**2+1e-12).all()
    repaired=a[0]+(1-t[:,None,None])*a[1]
    assert (repaired-x).square().sum()<(a[0]-x).square().sum()
    assert a[2]['identity'].any() and a[1][a[2]['identity']].eq(0).all()


def test_zero_geometry_field_preserves_the_real_physical_adapter():
    torch.set_num_threads(1)
    spec=json.loads(Path('research/evidence/gaga_feedback_distance_s0_v1.json').read_text())
    model=base.initialize(spec,'cpu').double();physical_head=PhysicalConnection(spec['atomic_numbers'],velocity_scale=2.).double()
    with torch.no_grad():physical_head.pair_network[-1].weight.normal_(std=.1)
    physical=PhysicalFieldTransform(model,spec,physical_head,4.,strength_limit=4.)
    geometry=GeometryRecoveryField(spec['atomic_numbers']).double();combined=GeometryThenPhysical(physical,geometry)
    x=base.center(torch.randn(2,5,3,dtype=torch.float64));v=base.center(torch.randn_like(x));t=x.new_tensor([[.2],[.8]])
    z=torch.tensor([[6,6,8,1,1]]*2)
    torch.testing.assert_close(combined(x,t,z,v),physical(x,t,z,v),atol=0,rtol=0)
    assert combined.calls==1 and geometry.forward_calls==1


def test_geometry_training_can_reduce_a_fixed_recovery_error():
    torch.set_num_threads(1);torch.manual_seed(76103)
    model=GeometryRecoveryField([1,6,8]);x=base.center(torch.randn(8,5,3));z=torch.tensor([[6,6,8,1,1]]*8)
    t=torch.full((8,),.75);noisy,target,_=denoising_targets(x,t,torch.Generator().manual_seed(76104))
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    initial=float((model(noisy,z,t)-target).square().mean())
    for _ in range(30):
        loss=(model(noisy,z,t)-target).square().mean();optimizer.zero_grad();loss.backward();optimizer.step()
    assert float((model(noisy,z,t)-target).square().mean())<initial
