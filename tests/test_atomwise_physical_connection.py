import json
from pathlib import Path
import pytest
import torch
from cfm_mol import matched_egnn as base,connectivity_feedback as feedback
from cfm_mol.physical_connection import make_physical_connection
from cfm_mol.atomwise_physical_connection import balanced_force_shift
from cfm_mol.matched_physical_connection import PhysicalFieldTransform,head_inputs


def test_balanced_shift_matches_movement_budget_and_is_linear_energy_descent():
    torch.manual_seed(62111)
    force=torch.randn(7,9,3,dtype=torch.float64)*torch.linspace(.1,10,9)[None,:,None]
    force-=force.mean(1,keepdim=True);reference=.1*force/force.flatten(1).norm(dim=-1)[:,None,None]
    shift,mobility,scale=balanced_force_shift(force,reference)
    torch.testing.assert_close(shift.flatten(1).norm(dim=-1),reference.flatten(1).norm(dim=-1),rtol=1e-12,atol=1e-12)
    assert ((force*shift).sum((1,2))>=0).all() and (mobility>0).all() and (scale>0).all()
    torch.testing.assert_close(shift.mean(1),torch.zeros(7,3,dtype=shift.dtype),atol=1e-14,rtol=0)
    rotation=torch.linalg.qr(torch.randn(3,3,dtype=force.dtype))[0];perm=torch.randperm(9)
    transformed,_,_=balanced_force_shift(force[:,perm]@rotation,reference[:,perm]@rotation)
    torch.testing.assert_close(transformed,shift[:,perm]@rotation,rtol=1e-11,atol=1e-12)
    zero,_,_=balanced_force_shift(torch.zeros_like(force),torch.zeros_like(force));assert torch.equal(zero,torch.zeros_like(zero))


def test_atomwise_field_symmetry_bound_and_weak_contact_budget():
    torch.manual_seed(62112);config=dict(atomic_numbers=[1,6,8],velocity_scale=2.)
    pair=make_physical_connection(**config).double();atom=make_physical_connection(**config,normalization='atomwise').double()
    atom.load_state_dict(pair.state_dict());assert sum(p.numel() for p in pair.parameters())==sum(p.numel() for p in atom.parameters())
    with torch.no_grad():
        pair.pair_network[-1].weight.zero_();pair.pair_network[-1].bias.fill_(2.)
        atom.load_state_dict(pair.state_dict())
    # A weakly attached H beside a compact heavy cluster. Positive bounded
    # coefficients isolate the normalization's representational effect.
    x=torch.tensor([[[0.,0,0],[.8,.1,0],[.4,.6,.2],[2.4,0,0]]],dtype=torch.float64);x=base.center(x)
    z=torch.tensor([[6,6,8,1]]);t=torch.tensor([.8],dtype=x.dtype);inputs=head_inputs(x,x,z,t,config['atomic_numbers'])
    a=atom(*inputs).reshape_as(x);b=pair(*inputs).reshape_as(x)
    assert a[0,3].norm()>b[0,3].norm()
    assert (a.norm(dim=-1)<=2*t[:,None]**2+1e-12).all()
    torch.testing.assert_close(a.mean(1),torch.zeros(1,3,dtype=x.dtype),atol=1e-12,rtol=0)
    q=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype))[0];p=torch.tensor([3,1,0,2])
    transformed=atom(*head_inputs(x[:,p]@q,x[:,p]@q,z[:,p],t,config['atomic_numbers'])).reshape_as(x)
    torch.testing.assert_close(transformed,a[:,p]@q,rtol=1e-11,atol=1e-12)
    a.square().sum().backward();assert any(p.grad is not None and p.grad.abs().sum()>0 for p in atom.parameters())


@pytest.mark.parametrize('kind',['harmonic_fm','gaga'])
def test_zero_atomwise_head_preserves_real_parent_sampler(kind):
    torch.set_num_threads(2);name='gaga_feedback_distance' if kind=='harmonic_fm' else 'matched_generators_gaga'
    spec=json.loads(Path(f'research/evidence/{name}_s0_v1.json').read_text());model=base.initialize(spec,'cpu').eval();source=base.HarmonicSource();context=None
    if kind=='harmonic_fm':feedback.install(model);context=feedback.GeometryContext(source,'distance')
    head=make_physical_connection(atomic_numbers=spec['atomic_numbers'],velocity_scale=2.,normalization='atomwise')
    transform=PhysicalFieldTransform(model,spec,head,4.,strength_limit=4.);numbers=[6,6,8,1,1]
    def sample(hook):
        if context is None:return base.sample(model,numbers,kind,spec,source,62113,2,128,field_transform=hook)
        return feedback.sample(model,numbers,spec,source,context,62113,2,128,field_transform=hook)
    before=base.state_hash(model);plain=sample(None);corrected=sample(transform)
    for a,b in zip(plain,corrected):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert base.state_hash(model)==before and not any(p.requires_grad for p in model.parameters())
    assert transform.calls==head.forward_calls==(64 if kind=='harmonic_fm' else 128)
