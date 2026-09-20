import json
from pathlib import Path
import pytest
import torch
from cfm_mol import matched_egnn as base,connectivity_feedback as feedback
from cfm_mol.physical_connection import PhysicalConnection
from cfm_mol.matched_physical_connection import (PhysicalFieldTransform,endpoint_and_progress,
                                                native_correction,head_inputs)


def setup(kind):
    name='gaga_feedback_distance' if kind=='harmonic_fm' else 'matched_generators_gaga'
    spec=json.loads(Path(f'research/evidence/{name}_s0_v1.json').read_text())
    model=base.initialize(spec,'cpu').eval();source=base.HarmonicSource()
    context=None
    if kind=='harmonic_fm':
        feedback.install(model);context=feedback.GeometryContext(source,'distance')
    return model,spec,source,context


@pytest.mark.parametrize('kind',['harmonic_fm','gaga'])
def test_zero_residual_and_passive_observation_preserve_actual_sampler_and_rng(kind):
    torch.set_num_threads(2)
    model,spec,source,context=setup(kind)
    numbers=[6,6,8,1,1];head=PhysicalConnection(spec['atomic_numbers'],velocity_scale=2.)
    transform=PhysicalFieldTransform(model,spec,head)
    state=base.state_hash(model);calls=[]
    handle=model.dynamics.egnn.register_forward_hook(lambda *_:calls.append(1))
    def sample(hook):
        if context is None:return base.sample(model,numbers,kind,spec,source,60113,2,128,field_transform=hook)
        return feedback.sample(model,numbers,spec,source,context,60113,2,128,field_transform=hook)
    plain=sample(None);assert len(calls)==128
    zero=sample(transform);assert len(calls)==256
    assert transform.calls==head.forward_calls==(64 if kind=='harmonic_fm' else 128)
    observed=[]
    def observer(x,t,z,value):
        observed.append(endpoint_and_progress(model,x,t,value,spec));return value
    watched=sample(observer);assert len(calls)==384
    handle.remove()
    for expected,new,passive in zip(plain,zero,watched):
        torch.testing.assert_close(expected,new,rtol=0,atol=0)
        torch.testing.assert_close(expected,passive,rtol=0,atol=0)
    assert base.state_hash(model)==state and not any(p.requires_grad for p in model.parameters())
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize('kind',['harmonic_fm','gaga'])
def test_endpoint_displacement_conversion_symmetry_and_head_only_gradient(kind):
    model,spec,_,_=setup(kind);model.double();model.requires_grad_(False)
    x=base.center(torch.randn(3,5,3,dtype=torch.float64));value=base.center(torch.randn_like(x))
    t=torch.tensor([[0.],[.17],[.65]],dtype=x.dtype)
    endpoint,progress=endpoint_and_progress(model,x,t,value,spec)
    correction=base.center(torch.randn_like(x))*.02
    new,_=endpoint_and_progress(model,x,t,value+native_correction(model,x,t,correction,spec),spec)
    torch.testing.assert_close(new,endpoint+(1-progress[:,None,None])*correction,atol=1e-9,rtol=1e-8)
    if kind=='gaga':assert torch.equal(native_correction(model,x,t,correction,spec)[0],torch.zeros_like(x[0]))
    head=PhysicalConnection(spec['atomic_numbers'],velocity_scale=2.).double()
    with torch.no_grad():head.pair_network[-1].weight.normal_(std=.1)
    z=torch.tensor([[6,6,8,1,1]]*3);r=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype))[0];p=torch.tensor([4,1,3,2,0])
    a=head(*head_inputs(x,endpoint,z,progress,spec['atomic_numbers'])).reshape_as(x)
    b=head(*head_inputs(x[:,p]@r,endpoint[:,p]@r,z[:,p],progress,spec['atomic_numbers'])).reshape_as(x)
    torch.testing.assert_close(b,a[:,p]@r,atol=1e-10,rtol=1e-9)
    torch.testing.assert_close(a.mean(1),torch.zeros(3,3,dtype=x.dtype),atol=1e-12,rtol=0)
    assert (a.norm(dim=-1)<=2*progress[:,None]**2+1e-12).all()
    (a-correction).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in head.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
