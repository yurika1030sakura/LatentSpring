import copy
from types import SimpleNamespace
import torch
from cfm_mol.physical_connection import PhysicalConnection,patch_physical_connection


def inputs():
    torch.manual_seed(12);x=torch.randn(5,3,dtype=torch.float64);x-=x.mean(0)
    h=x+.2*torch.randn_like(x);h-=h.mean(0)
    types=torch.tensor([0,1,2,0,3]);t=torch.tensor([.8],dtype=torch.float64);batch=torch.zeros(5,dtype=torch.long)
    edges=torch.where(~torch.eye(5,dtype=torch.bool))
    return x,h,types,t,batch,*edges


def test_zero_initialization_centering_rotation_permutation_and_bound():
    model=PhysicalConnection([6,7,8,1]).double();args=inputs()
    assert torch.equal(model(*args),torch.zeros_like(args[0]))
    torch.nn.init.normal_(model.pair_network[-1].weight,std=.5)
    x,h,types,t,batch,src,dst=args;y=model(*args)
    assert y.abs().sum()>0 and y.sum(0).abs().max()<1e-12
    assert y.norm(dim=-1).max()<=t.item()**2+1e-12
    rotation=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype))[0]
    torch.testing.assert_close(model(x@rotation,h@rotation,types,t,batch,src,dst),y@rotation,atol=1e-12,rtol=1e-10)
    perm=torch.tensor([3,1,4,0,2]);torch.testing.assert_close(model(x[perm],h[perm],types[perm],t,batch,src,dst),y[perm],atol=1e-12,rtol=1e-10)
    collision=model(torch.zeros_like(x),torch.zeros_like(h),types,t,batch,src,dst)
    assert torch.isfinite(collision).all()


def test_native_backbone_identity_then_nonzero_learning_with_frozen_parent():
    from flowmol.models.ctmc_vector_field import CTMCVectorField
    from flowmol.models.interpolant_scheduler import InterpolantScheduler
    from test_clamped_density import graph_batch
    from cfm_mol.geometry_self_conditioning import patch_geometry_self_conditioning
    from cfm_mol.clamped_fm import clamped_fm_loss
    from cfm_mol.clamped_density import deterministic_field
    torch.manual_seed(4)
    field=CTMCVectorField(n_atom_types=3,canonical_feat_order=['x','a','c','e'],
        interpolant_scheduler=InterpolantScheduler(['x','a','c','e'],schedule_type='linear'),
        n_vec_channels=4,n_hidden_scalars=8,n_hidden_edge_feats=8,n_molecule_updates=1,
        convs_per_update=2,n_message_gvps=1,n_update_gvps=1,n_expansion_gvps=1,rbf_dim=4,self_conditioning=True)
    model=SimpleNamespace(vector_field=field,_research_prior_kind='gaussian');patch_geometry_self_conditioning(model)
    graph,nbi,uem=graph_batch((3,),dtype=torch.float32)
    for k in ['a','c']:graph.ndata[k+'_t']=graph.ndata[k+'_1_true']
    graph.edata['e_t']=graph.edata['e_1_true'];t=torch.tensor([.7])
    with deterministic_field(field):before=field(graph,t,node_batch_idx=nbi,upper_edge_mask=uem)['x'].detach().clone()
    parent={n:p.detach().clone() for n,p in field.named_parameters()}
    patch_physical_connection(model,atomic_numbers=[6,7,8],embedding_dim=16,hidden_dim=64,velocity_scale=1.,gate_power=2)
    with deterministic_field(field):after=field(graph,t,node_batch_idx=nbi,upper_edge_mask=uem)['x']
    torch.testing.assert_close(after,before,atol=0,rtol=0)
    opt=torch.optim.AdamW(field.physical_connection.parameters(),lr=.001)
    loss=clamped_fm_loss(model,graph,nbi,uem,terminal_time=1.,parameterization='displacement',generator=torch.Generator().manual_seed(123))
    loss.backward();assert field.physical_connection.pair_network[-1].weight.grad.abs().sum()>0;opt.step()
    for n,p in field.named_parameters():
        if n in parent:assert p.grad is None and torch.equal(p,parent[n])
    with deterministic_field(field):updated=field(graph,t,node_batch_idx=nbi,upper_edge_mask=uem)['x']
    assert not torch.equal(updated,before)
    from cfm_mol.clamped_density import sample_clamped_flow
    before_calls=field.physical_connection.forward_calls
    sampled=sample_clamped_flow(model,graph,nbi,uem,x0=graph.ndata['x_1_true'],n_ode_steps=2,
        terminal_time=1.,parameterization='displacement')
    assert torch.isfinite(sampled).all() and field.physical_connection.forward_calls-before_calls==4
