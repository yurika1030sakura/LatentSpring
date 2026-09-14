import itertools
import torch
from cfm_mol.dynamic_tree_attention import tree_edge_marginals,local_edge_mass,patch_dynamic_tree_attention
from cfm_mol.radial_reference import prepare_research_backbone
from cfm_mol.clamped_fm import clamped_fm_loss
from test_latent_tree_context import small_model,prepare_graph,predict
from test_source_utility import enumerate_tree_scores


def test_marginals_exact_normalization_cuts_and_derivative():
    raw=torch.tensor([[0.,1.,-6.,-6.],[1.,0.,-6.,-6.],[-6.,-6.,0.,1.],[-6.,-6.,1.,0.]],dtype=torch.double,requires_grad=True)
    a=(raw+raw.T)/2;m=tree_edge_marginals(a)
    torch.testing.assert_close(m.sum()/2,torch.tensor(3.,dtype=torch.double))
    scores=enumerate_tree_scores(a);partition=scores.logsumexp(0)
    gradient=torch.autograd.grad(partition,raw,create_graph=True)[0]
    torch.testing.assert_close(m,gradient*2,atol=1e-11,rtol=1e-10)
    for k in range(1,4):
        for cut in itertools.combinations(range(4),k):
            other=[i for i in range(4) if i not in cut]
            assert float(m[list(cut)][:,other].sum())>=1-1e-11
    assert float(local_edge_mass(a)[:2,2:].sum())<.02
    torch.testing.assert_close(torch.autograd.grad(m.square().sum(),raw,retain_graph=True)[0],torch.autograd.grad((gradient*2).square().sum(),raw)[0],atol=1e-10,rtol=1e-8)


def test_dynamic_backbone_zero_init_learning_symmetry_restore():
    model=small_model();graph,nbi,uem=prepare_graph();before=predict(model,graph,nbi,uem)
    config=dict(atomic_numbers_by_type=[1,6,8],mode='tree_learned',hidden=8)
    patch_dynamic_tree_attention(model,**config)
    assert torch.equal(before,predict(model,graph,nbi,uem))
    block=model.vector_field.dynamic_tree_attention
    with torch.no_grad():block.messages[-1].weight.normal_(std=.03)
    model.vector_field.train()
    loss=clamped_fm_loss(model,graph,nbi,uem,terminal_time=1.,parameterization='displacement',
        prior_positions=graph.ndata['x_1_true'].clone()*.8,generator=torch.Generator().manual_seed(32001))
    loss.backward()
    assert torch.isfinite(block.affinity[-1].weight.grad).all() and block.affinity[-1].weight.grad.norm()>0
    original=predict(model,graph,nbi,uem)
    rotation,_=torch.linalg.qr(torch.randn(3,3,generator=torch.Generator().manual_seed(32002)))
    perm=torch.tensor([2,0,3,1])
    with graph.local_scope():
        for key in ['a_t','a_1_true','c_t','c_1_true']:
            if key in graph.ndata:graph.ndata[key]=graph.ndata[key][perm]
        graph.ndata['x_t']=graph.ndata['x_t'][perm]@rotation
        changed=predict(model,graph,nbi,uem)
    torch.testing.assert_close(changed,original[perm]@rotation,atol=3e-5,rtol=3e-5)
    restored=small_model();prepare_research_backbone(restored,dict(dynamic_tree_attention=config,source_prior_kind='fixed'))
    restored.vector_field.load_state_dict(model.vector_field.state_dict(),strict=True)
    torch.testing.assert_close(original,predict(restored,graph,nbi,uem),rtol=0,atol=0)


def test_batched_tree_mass_and_local_mass():
    torch.manual_seed(32003);raw=torch.randn(3,12,12,dtype=torch.double);a=(raw+raw.transpose(-1,-2))/2
    batched=tree_edge_marginals(a)
    torch.testing.assert_close(batched,torch.stack([tree_edge_marginals(x) for x in a]))
    assert float(batched.min())>=-1e-12 and float(batched.max())<=1+1e-12
    torch.testing.assert_close(local_edge_mass(a).sum((-1,-2))/2,torch.full((3,),11.,dtype=torch.double))


def test_dynamic_adapter_mixed_size_batch_matches_separate_graphs():
    import dgl
    from cfm_mol.dynamic_tree_attention import DynamicTreeAttention
    from test_clamped_density import graph_batch
    graph,nbi,_=graph_batch((3,4),dtype=torch.float32)
    block=DynamicTreeAttention(8,8,[1,6,8],hidden=8)
    with torch.no_grad():block.messages[-1].weight.normal_(std=.02)
    scalars=torch.randn(7,8,generator=torch.Generator().manual_seed(32004))
    edges=torch.randn(graph.num_edges(),8,generator=torch.Generator().manual_seed(32005))
    positions=graph.ndata['x_1_true']
    batched=block(graph,scalars,positions,edges,nbi)
    separate=[];node_start=edge_start=0
    for g in dgl.unbatch(graph):
        n=g.num_nodes();e=g.num_edges()
        separate.append(block(g,scalars[node_start:node_start+n],positions[node_start:node_start+n],edges[edge_start:edge_start+e],torch.zeros(n,dtype=torch.long)))
        node_start+=n;edge_start+=e
    torch.testing.assert_close(batched,torch.cat(separate),atol=1e-6,rtol=1e-5)


def test_relative_weight_floor_preserves_gauge_gradient():
    raw=torch.tensor([[0.,1.,-4.,-2.],[1.,0.,-3.,-.2],[-4.,-3.,0.,.2],[-2.,-.2,.2,0.]],dtype=torch.double)
    for fn in [tree_edge_marginals,local_edge_mass]:
        shift=torch.tensor(0.,dtype=torch.double,requires_grad=True)
        m=fn(raw+shift,relative_floor=.1)
        gradient=torch.autograd.grad(m[0,1]+2*m[1,2],shift)[0]
        torch.testing.assert_close(gradient,torch.zeros_like(gradient),atol=1e-12,rtol=0)
        variable=raw.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(lambda x:fn((x+x.T)/2,relative_floor=.1),variable,eps=1e-5,atol=1e-6,rtol=1e-4)
