import itertools
import numpy as np
import pytest
import torch
from cfm_mol.degree_tree import decode_pruefer,degree_log_partition,tree_degrees,DegreeTreePrior,sample_degree_tree
from cfm_mol.tree_manifold import tree_geometry,to_coordinates,from_coordinates,product_path,sample_product_source,midpoint_product
from cfm_mol.geometric_domain import connected_nonoverlapping
from cfm_mol.chemical_moves import covalent_radii


def test_degree_partition_and_gradient_match_all_labeled_trees():
    torch.manual_seed(32801);psi=torch.randn(4,3,dtype=torch.double,requires_grad=True);caps=[3,2,2,1]
    scores=[]
    for code in itertools.product(range(4),repeat=2):
        degree=tree_degrees(decode_pruefer(code,4),4)
        if all(d<=c for d,c in zip(degree,caps)):
            scores.append(sum(psi[i,d-1] for i,d in enumerate(degree)))
    truth=torch.stack(scores).logsumexp(0);value=degree_log_partition(psi,caps)
    torch.testing.assert_close(value,truth)
    torch.testing.assert_close(torch.autograd.grad(value,psi,retain_graph=True)[0],torch.autograd.grad(truth,psi)[0])
    rng=np.random.default_rng(32802)
    for _ in range(100):
        edges=sample_degree_tree(psi,caps,rng)
        assert all(d<=c for d,c in zip(tree_degrees(edges,4),caps))
    with pytest.raises(ValueError,match='No tree'):degree_log_partition(psi,[1,1,1,1])


def test_learned_tree_probability_normalizes_and_is_equivariant():
    model=DegreeTreePrior().double();torch.nn.init.normal_(model.head[-1].weight,std=.2)
    numbers=[6,6,1,1];edges_list=[decode_pruefer(code,4) for code in itertools.product(range(4),repeat=2)]
    values=torch.stack([model.log_prob(e,numbers,0,1) for e in edges_list])
    torch.testing.assert_close(values.logsumexp(0),torch.zeros((),dtype=torch.double))
    permutation=[2,0,3,1];inverse=np.argsort(permutation)
    for edges,value in zip(edges_list,values):
        transformed=[(int(inverse[i]),int(inverse[j])) for i,j in edges]
        expected=model.log_prob(transformed,[numbers[i] for i in permutation],0,1)
        torch.testing.assert_close(value,expected)


def test_product_path_roundtrip_velocity_and_connected_edges():
    edges=[(0,1),(0,2),(1,3)];numbers=[6,6,1,1]
    y0,u0,b,inv,length=sample_product_source(edges,numbers,generator=torch.Generator().manual_seed(32803))
    y1,u1,*_=sample_product_source(edges,numbers,generator=torch.Generator().manual_seed(32804))
    x=to_coordinates(y0,u0,inv,length);yr,ur=from_coordinates(x,b,length)
    torch.testing.assert_close(yr,y0);torch.testing.assert_close(ur,u0)
    t=.37;y,u,dy,du=product_path(y0,u0,y1,u1,t)
    eps=1e-6;yp,up,*_=product_path(y0,u0,y1,u1,t+eps);ym,um,*_=product_path(y0,u0,y1,u1,t-eps)
    torch.testing.assert_close((up-um)/(2*eps),du,atol=1e-8,rtol=1e-7)
    torch.testing.assert_close((yp-ym)/(2*eps),dy,atol=1e-8,rtol=1e-7)
    for t in [0.,.2,.5,.8,1.]:
        y,u,*_=product_path(y0,u0,y1,u1,t);x=to_coordinates(y,u,inv,length)
        torch.testing.assert_close(u.norm(dim=-1),torch.ones(3,dtype=torch.double))
        ratio=(b@x).norm(dim=-1)/length
        assert (ratio>.65).all() and (ratio<1.20).all()
        torch.testing.assert_close(x.mean(0),torch.zeros(3,dtype=torch.double),atol=1e-12,rtol=0)


def test_manifold_integrator_preserves_support_under_large_neural_fields():
    edges=[(0,1),(0,2),(1,3)];numbers=[6,6,1,1]
    y,u,b,inv,length=sample_product_source(edges,numbers,generator=torch.Generator().manual_seed(32805))
    axis=torch.tensor([1.,2.,3.],dtype=torch.double)
    def field(y,u,t):return torch.ones_like(y)*1000,torch.linalg.cross(axis.expand_as(u),u)*1000
    for step in range(16):y,u=midpoint_product(y,u,step/16,1/16,field)
    x=to_coordinates(y,u,inv,length);ratio=(b@x).norm(dim=-1)/length
    assert torch.isfinite(x).all() and (ratio>=.65-1e-12).all() and (ratio<=1.20+1e-12).all()
    torch.testing.assert_close(u.norm(dim=-1),torch.ones(3,dtype=torch.double),atol=1e-12,rtol=0)


def test_hierarchical_coordination_probability_normalizes_and_sampling_caps():
    from cfm_mol.degree_tree import CoordinationTreePrior,ORGANIC_CAPS
    model=CoordinationTreePrior().double()
    for head in [model.head,model.allocation_head]:torch.nn.init.normal_(head[-1].weight,std=.1)
    for numbers in [[6,6,1,1],[6,7,1,9,1]]:
        n=len(numbers);spin=1 if sum(numbers)%2==0 else 2
        values=[model.log_prob(decode_pruefer(code,n),numbers,0,spin) for code in itertools.product(range(n),repeat=n-2)]
        torch.testing.assert_close(torch.stack(values).logsumexp(0),torch.zeros((),dtype=torch.double))
        rng=np.random.default_rng(32806)
        for _ in range(20):
            edges=model.sample(numbers,0,spin,rng=rng)
            degree=tree_degrees(edges,n)
            assert all(d<=ORGANIC_CAPS[z] for d,z in zip(degree,numbers))
            assert all(degree[i]==1 for i,z in enumerate(numbers) if z in model.terminal_species)
    edges=model.sample([6,6,6,6]+[1]*10,0,1,rng=np.random.default_rng(32807))
    loss=-model.log_prob(edges,[6,6,6,6]+[1]*10,0,1);loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    perm=np.random.default_rng(32808).permutation(14);inverse=np.argsort(perm);numbers=[6,6,6,6]+[1]*10
    transformed=[(int(inverse[i]),int(inverse[j])) for i,j in edges]
    torch.testing.assert_close(model.log_prob(edges,numbers,0,1),model.log_prob(transformed,[numbers[i] for i in perm],0,1))


def test_actual_vector_field_learns_product_velocity():
    from test_latent_tree_context import small_model,prepare_graph
    from cfm_mol.latent_tree_context import patch_latent_tree_context,tree_features,attach_context
    from cfm_mol.clamped_density import deterministic_field,position_velocity
    from cfm_mol.tree_manifold import cap_angular
    model=small_model();graph,nbi,uem=prepare_graph();patch_latent_tree_context(model,hidden=8)
    numbers=[6,6,1,1];edges=[(0,1),(0,2),(1,3)]
    attach_context(graph,[tree_features(4,edges)],nbi)
    y0,u0,b,inv,length=sample_product_source(edges,numbers,generator=torch.Generator().manual_seed(32809))
    y1,u1,*_=sample_product_source(edges,numbers,generator=torch.Generator().manual_seed(32810))
    y,u,dy,du=product_path(y0,u0,y1,u1,.4);x=to_coordinates(y,u,inv,length)
    with deterministic_field(model.vector_field):v=position_velocity(model,graph,x.float(),torch.tensor([.4]),nbi,uem,parameterization='displacement')
    edge=b@v.double()/length[:,None];radial=(edge*u).sum(-1)
    predicted=radial[:,None]*u+cap_angular(edge-radial[:,None]*u)
    loss=(predicted-(dy[:,None]*u+du)).square().mean();loss.backward()
    grad=model.vector_field.latent_tree_adapter[-1].weight.grad
    assert torch.isfinite(grad).all() and grad.norm()>0
