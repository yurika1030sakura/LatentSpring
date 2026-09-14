import itertools
import math
import numpy as np
import torch
from cfm_mol.tree_mixture_prior import (log_tree_partition,radial_log_density,
    tree_mixture_log_prob,weighted_tree,sample_tree_coordinates,TreeMixturePrior)


def trees(n):
    for code in itertools.product(range(n),repeat=max(0,n-2)):
        degree=[1]*n
        for v in code:degree[v]+=1
        edges=[]
        for v in code:
            leaf=next(i for i,d in enumerate(degree) if d==1)
            edges.append((min(leaf,v),max(leaf,v)));degree[leaf]-=1;degree[v]-=1
        left=[i for i,d in enumerate(degree) if d==1]
        edges.append(tuple(left));yield tuple(sorted(edges))


def test_partition_and_gradients_against_tree_enumeration():
    for n in [2,3,4,5]:
        raw=torch.randn(n,n,generator=torch.Generator().manual_seed(n),dtype=torch.float64,requires_grad=True)
        w=(raw+raw.T)/2
        explicit=torch.logsumexp(torch.stack([sum(w[i,j] for i,j in edges) for edges in trees(n)]),0)
        actual=log_tree_partition(w)
        torch.testing.assert_close(actual,explicit,atol=1e-12,rtol=0)
        g1=torch.autograd.grad(actual,raw,retain_graph=True)[0]
        g2=torch.autograd.grad(explicit,raw)[0]
        torch.testing.assert_close(g1,g2,atol=1e-12,rtol=0)
        assert abs(float(g1.sum())-(n-1))<1e-12


def test_extreme_log_weight_range_and_vertex_permutation():
    w=torch.tensor([[0.,-1000.,-2000.,-3000.],[-1000.,0.,-300.,-20.],[-2000.,-300.,0.,-700.],[-3000.,-20.,-700.,0.]],dtype=torch.float64,requires_grad=True)
    actual=log_tree_partition(w)
    explicit=torch.logsumexp(torch.stack([sum(w[i,j] for i,j in e) for e in trees(4)]),0)
    torch.testing.assert_close(actual,explicit,atol=1e-10,rtol=0)
    permutation=torch.tensor([2,0,3,1])
    torch.testing.assert_close(actual,log_tree_partition(w[permutation][:,permutation]),atol=1e-10,rtol=0)
    assert torch.isfinite(torch.autograd.grad(actual,w)[0]).all()


def test_normalized_edge_law_and_intrinsic_density_factor():
    from scipy.integrate import quad
    length=1.3;width=.2
    integral=quad(lambda r:4*math.pi*r*r*math.exp(float(radial_log_density(torch.tensor(r,dtype=torch.float64),torch.tensor(length),width))),.001,8,epsabs=1e-10)[0]
    assert abs(integral-1)<1e-7
    from cfm_mol.nonequilibrium import centered_orthonormal_basis
    n=4;basis=centered_orthonormal_basis(n)
    incidence=torch.zeros(n-1,n,dtype=torch.float64)
    for row,(i,j) in enumerate([(0,1),(1,2),(1,3)]):incidence[row,i]=1;incidence[row,j]=-1
    jacobian=torch.kron(incidence@basis,torch.eye(3,dtype=torch.float64))
    assert abs(float(torch.linalg.slogdet(jacobian)[1])-1.5*math.log(n))<1e-12
    x=torch.tensor([[.4,0.,0.],[-.1,.7,0.],[-.3,-.7,0.]],dtype=torch.float64)
    a=torch.tensor([[0.,.7,-.3],[.7,0.,.2],[-.3,.2,0.]],dtype=x.dtype)
    lengths=torch.full((3,3),length,dtype=x.dtype)
    terms=[]
    for edges in trees(3):
        terms.append(sum(a[i,j]+radial_log_density((x[i]-x[j]).norm(),lengths[i,j],width) for i,j in edges))
    expected=1.5*math.log(3)+torch.logsumexp(torch.stack(terms),0)-log_tree_partition(a)
    torch.testing.assert_close(tree_mixture_log_prob(x,a,lengths,width),expected,atol=1e-12,rtol=0)


def test_wilson_sampling_matches_weighted_tree_probabilities():
    a=np.array([[0.,.7,-.3],[.7,0.,.2],[-.3,.2,0.]])
    all_trees=list(trees(3));counts={t:0 for t in all_trees};rng=np.random.default_rng(998)
    for _ in range(10000):
        edges=tuple(sorted(tuple(sorted(e)) for e in weighted_tree(a,rng)));counts[edges]+=1
    expected=np.array([math.exp(sum(a[i,j] for i,j in t)) for t in all_trees]);expected/=expected.sum()
    observed=np.array([counts[t]/10000 for t in all_trees])
    assert np.max(abs(expected-observed))<.02


def test_prior_geometry_symmetry_and_likelihood_gradient():
    model=TreeMixturePrior(mode='pair').double()
    with torch.no_grad():model.head[-1].weight.fill_(.01)
    numbers=[6,1,1,8,1]
    x,edges=model.sample(numbers,0,2,rng=np.random.default_rng(301),generator=torch.Generator().manual_seed(302))
    assert len(edges)==4 and x.mean(0).abs().max()<1e-12
    x.requires_grad_(True);value=model.log_prob(x,numbers,0,2)
    q,_=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype,generator=torch.Generator().manual_seed(88)))
    permutation=torch.tensor([2,3,0,4,1])
    other=model.log_prob((x@q)[permutation],np.array(numbers)[permutation].tolist(),0,2)
    torch.testing.assert_close(value,other,atol=1e-10,rtol=0)
    gx,gw=torch.autograd.grad(value,[x,model.head[-1].weight])
    assert torch.isfinite(gx).all() and torch.isfinite(gw).all()
    assert gx.mean(0).abs().max()<1e-10
    direction=torch.randn(x.shape,dtype=x.dtype,generator=torch.Generator().manual_seed(89));direction-=direction.mean(0)
    eps=1e-5
    fd=(model.log_prob(x+eps*direction,numbers,0,2)-model.log_prob(x-eps*direction,numbers,0,2))/(2*eps)
    torch.testing.assert_close(fd,(gx*direction).sum(),atol=1e-6,rtol=1e-6)


def test_non_gaussian_prior_in_clamped_density_and_fm_interface():
    from types import SimpleNamespace
    from test_clamped_density import graph_batch, LinearHead, Schedule
    from cfm_mol.clamped_density import log_density_clamped_flow
    from cfm_mol.clamped_fm import clamped_fm_path
    graph,nbi,uem=graph_batch();head=LinearHead(a=.1,endpoint=False)
    model=SimpleNamespace(vector_field=head,_research_prior_kind='fixed');prior=TreeMixturePrior('fixed').double()
    def logp(x,g,idx):
        return torch.stack([prior.log_prob(x[idx==i],[6]*int((idx==i).sum()),0,1) for i in range(g.batch_size)])
    actual=log_density_clamped_flow(model,graph,nbi,uem,n_ode_steps=16,n_hutchinson=0,
        terminal_time=.8,parameterization='velocity',solver='rk4',for_training=True,prior_log_prob=logp)
    initial=graph.ndata['x_1_true']*torch.exp(-head.a*.8)
    expected=logp(initial,graph,nbi)-3*(graph.batch_num_nodes()-1)*head.a*.8
    torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-8)
    ga=torch.autograd.grad(actual.sum(),head.a,retain_graph=True)[0]
    ge=torch.autograd.grad(expected.sum(),head.a)[0]
    torch.testing.assert_close(ga,ge,atol=1e-6,rtol=1e-7)
    supplied=[]
    for n in graph.batch_num_nodes():
        x,_=prior.sample([6]*int(n),0,1,rng=np.random.default_rng(int(n)),generator=torch.Generator().manual_seed(int(n)))
        supplied.append(x)
    supplied=torch.cat(supplied)
    _,_,_,info=clamped_fm_path(graph,nbi,Schedule(),terminal_time=1.,parameterization='displacement',
        prior_positions=supplied,generator=torch.Generator().manual_seed(40))
    torch.testing.assert_close(info['x0'],supplied,atol=1e-12,rtol=0)


def test_declared_prior_cannot_silently_use_gaussian_sampling_or_density():
    import pytest
    from types import SimpleNamespace
    from test_clamped_density import graph_batch,LinearHead
    from cfm_mol.radial_reference import prepare_research_backbone
    from cfm_mol.clamped_density import sample_clamped_flow,log_density_clamped_flow
    from cfm_mol.clamped_fm import clamped_fm_loss
    g,nbi,uem=graph_batch();model=SimpleNamespace(vector_field=LinearHead(endpoint=False))
    prepare_research_backbone(model,dict(source_prior_kind='pair'))
    with pytest.raises(ValueError,match='non-Gaussian'):
        sample_clamped_flow(model,g,nbi,uem,parameterization='velocity')
    with pytest.raises(ValueError,match='non-Gaussian'):
        log_density_clamped_flow(model,g,nbi,uem,parameterization='velocity')
    with pytest.raises(ValueError,match='non-Gaussian'):
        clamped_fm_loss(model,g,nbi,uem,parameterization='displacement')
