"""Analytic divergence, symmetries, collisions and full density gradients."""
import copy
import pytest
import torch
from torch import nn
from test_clamped_density import graph_batch, Schedule
from cfm_mol.clamped_density import _trace,log_density_clamped_flow
from cfm_mol.radial_reference import RadialPairReference,patch_radial_reference,prepare_research_backbone


def setup_reference(counts=(3,4)):
    torch.manual_seed(716)
    graph,nbi,uem=graph_batch(counts)
    for key,width in [('a',4),('c',7)]:
        values=torch.nn.functional.one_hot(torch.arange(graph.num_nodes())%width,width).double()
        graph.ndata[f'{key}_t']=values
        graph.ndata[f'{key}_1_true']=values
    reference=RadialPairReference(4,7,embedding_dim=4,hidden_dim=8,n_kernels=5).double()
    return graph,nbi,uem,reference


@pytest.mark.parametrize('counts',[(3,4),(1,2)])
def test_analytic_trace_and_its_coordinate_parameter_gradients(counts):
    graph,nbi,uem,reference=setup_reference(counts)
    x=graph.ndata['x_1_true'].clone().requires_grad_()
    t=torch.tensor([.2,.7],dtype=x.dtype)
    velocity,analytic=reference.velocity_and_divergence(graph,x,t,nbi)
    enumerated=_trace(velocity,x,nbi,graph.batch_size,None,True)
    assert torch.allclose(analytic,enumerated,atol=2e-12,rtol=2e-12)
    variables=(x,*tuple(reference.parameters()))
    left=torch.autograd.grad(analytic.sum(),variables,retain_graph=True)
    right=torch.autograd.grad(enumerated.sum(),variables)
    for a,b in zip(left,right):
        assert torch.allclose(a,b,atol=2e-11,rtol=2e-11)


def test_rotation_translation_permutation_equivariance():
    graph,nbi,uem,reference=setup_reference()
    x=graph.ndata['x_1_true'];t=x.new_tensor([.2,.7])
    velocity,divergence=reference.velocity_and_divergence(graph,x,t,nbi)
    rotation=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype)).Q
    transformed=x@rotation+x.new_tensor([[3.,-2.,1.],[-1.,4.,8.]])[nbi]
    vr,dr=reference.velocity_and_divergence(graph,transformed,t,nbi)
    assert torch.allclose(vr,velocity@rotation,atol=1e-12,rtol=1e-12)
    assert torch.allclose(dr,divergence,atol=1e-12,rtol=1e-12)
    order=torch.tensor([2,0,1,5,6,3,4])
    with graph.local_scope():
        for key in ['a_t','c_t']:graph.ndata[key]=graph.ndata[key][order]
        vp,dp=reference.velocity_and_divergence(graph,x[order],t,nbi)
    assert torch.allclose(vp,velocity[order],atol=1e-12,rtol=1e-12)
    assert torch.allclose(dp,divergence,atol=1e-12,rtol=1e-12)


def test_coincident_atoms_have_finite_second_derivatives():
    graph,nbi,uem,reference=setup_reference((2,3))
    x=torch.zeros_like(graph.ndata['x_1_true'],requires_grad=True)
    velocity,divergence=reference.velocity_and_divergence(graph,x,x.new_tensor([.2,.7]),nbi)
    first=torch.autograd.grad(velocity.square().sum()+divergence.sum(),x,create_graph=True)[0]
    second=torch.autograd.grad(first.sum(),x)[0]
    assert torch.equal(velocity,torch.zeros_like(velocity))
    assert all(torch.isfinite(value).all() for value in [divergence,first,second])


class BaseField(nn.Module):
    def __init__(self):
        super().__init__()
        self.old=nn.Parameter(torch.zeros(1,dtype=torch.float64))
        self.n_atom_types=4;self.n_charges=7
        self.interpolant_scheduler=Schedule()


@pytest.mark.parametrize('solver',['midpoint','rk4'])
def test_patch_full_density_gradient_adjoint_and_checkpoint_restore(solver):
    graph,nbi,uem,_=setup_reference((2,2))
    model=nn.Module();model.vector_field=BaseField()
    patch_radial_reference(model)
    assert not model.vector_field.old.requires_grad
    parameters=tuple(p for p in model.parameters() if p.requires_grad)
    def evaluate(adjoint=False):
        q=log_density_clamped_flow(model,graph,nbi,uem,n_ode_steps=3,n_hutchinson=0,
            terminal_time=1.,for_training=True,parameterization='displacement',
            discrete_adjoint=adjoint,solver=solver)
        return q,torch.autograd.grad(q.square().mean(),parameters)
    plain,adjoint=evaluate(),evaluate(True)
    assert torch.allclose(plain[0],adjoint[0],atol=1e-12,rtol=1e-12)
    for a,b in zip(plain[1],adjoint[1]):
        assert torch.allclose(a,b,atol=1e-10,rtol=1e-10)
    # Bypass the analytic hook to independently enumerate the physical field.
    hook=model.vector_field.exact_clamped_divergence
    del model.vector_field.exact_clamped_divergence
    enumerated=evaluate()
    model.vector_field.exact_clamped_divergence=hook
    assert torch.allclose(plain[0],enumerated[0],atol=1e-11,rtol=1e-11)
    for a,b in zip(plain[1],enumerated[1]):
        assert torch.allclose(a,b,atol=1e-9,rtol=1e-9)
    state=copy.deepcopy(model.state_dict())
    restored=nn.Module();restored.vector_field=BaseField()
    prepare_research_backbone(restored,{'position_backbone':'radial_reference'})
    restored.load_state_dict(state,strict=True)
    q=log_density_clamped_flow(restored,graph,nbi,uem,n_ode_steps=3,n_hutchinson=0,
        terminal_time=1.,parameterization='displacement',solver=solver)
    assert torch.equal(q,plain[0])
