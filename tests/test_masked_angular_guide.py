import torch
from cfm_mol.masked_angular_guide import MaskedAngularGuide,masked_angular_context,angular_log_score,angular_surface_score


def example():
    torch.manual_seed(131);x=torch.randn(2,6,3,dtype=torch.float64)
    bonds=torch.zeros(2,6,6,dtype=x.dtype)
    for i,j in [(0,1),(0,2),(0,3),(1,4),(1,5)]:bonds[:,i,j]=bonds[:,j,i]=1
    return x,bonds,torch.tensor([6,16,1,9,1,9]),torch.tensor([0.,1.,.02585],dtype=x.dtype),torch.tensor([[2,0],[4,1]])


def test_context_is_unchanged_after_leaf_motion_com_recentering_and_atom_permutation():
    model=MaskedAngularGuide().double();x,b,z,e,roots=example()
    with torch.no_grad():model.vector_weight.weight.normal_();model.tensor_weight.weight.normal_()
    original=model(x,b,z,e,roots);y=x.clone()
    for i,(leaf,anchor) in enumerate(roots.tolist()):
        radius=(x[i,leaf]-x[i,anchor]).norm();u=torch.randn(3,dtype=x.dtype);u=u/u.norm()
        y[i,leaf]=y[i,anchor]+radius*u
    y=y-y.mean(1,keepdim=True)
    for before,after in zip(masked_angular_context(x,roots),masked_angular_context(y,roots)):
        torch.testing.assert_close(before,after,atol=1e-12,rtol=0)
    for before,after in zip(original,model(y,b,z,e,roots)):torch.testing.assert_close(before,after,atol=1e-10,rtol=0)
    perm=torch.tensor([4,1,5,0,3,2]);inv=torch.argsort(perm)
    for before,after in zip(original,model(x[:,perm],b[:,perm][:,:,perm],z[perm],e,inv[roots])):
        torch.testing.assert_close(before,after,atol=1e-10,rtol=0)
    for before,after in zip(original,model(x[:1],b[:1],z,e,roots[:1])):
        torch.testing.assert_close(before[:1],after,atol=1e-10,rtol=0)


def test_vector_tensor_covariance_surface_derivative_and_bound():
    model=MaskedAngularGuide().double();x,b,z,e,roots=example()
    with torch.no_grad():model.vector_weight.weight.normal_();model.tensor_weight.weight.normal_()
    eta,a,envelope=model(x,b,z,e,roots)
    q=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype))[0];q[:,0]*=-1
    et,at,mt=model(x@q+17,b,z,e,roots)
    torch.testing.assert_close(et,eta@q,atol=1e-10,rtol=0)
    torch.testing.assert_close(at,q.T@a@q,atol=1e-10,rtol=0);torch.testing.assert_close(mt,envelope)
    assert bool((envelope<=64+1e-10).all())
    torch.testing.assert_close(a.diagonal(dim1=1,dim2=2).sum(1),torch.zeros(2,dtype=x.dtype),atol=1e-12,rtol=0)
    for _ in range(16):
        u=torch.randn(2,3,dtype=x.dtype);u=u/u.norm(dim=1,keepdim=True);u.requires_grad_()
        value=angular_log_score(u,eta,a)
        assert bool((value<=envelope+1e-10).all())
        gradient=torch.autograd.grad(value.sum(),u,retain_graph=True)[0]
        projected=gradient-(gradient*u).sum(1,keepdim=True)*u
        torch.testing.assert_close(projected,angular_surface_score(u,eta,a))


def test_zero_spatial_context_and_planar_tensor_symmetry():
    model=MaskedAngularGuide().double()
    with torch.no_grad():model.vector_weight.bias.fill_(4);model.tensor_weight.bias.fill_(4)
    x=torch.tensor([[[0.,0.,0.],[1.,0.,0.]]],dtype=torch.float64)
    eta,a,m=model(x,torch.tensor([[[0.,1.],[1.,0.]]],dtype=x.dtype),torch.tensor([1,1]),torch.tensor([0.,1.,.1],dtype=x.dtype),torch.tensor([[1,0]]))
    torch.testing.assert_close(eta,torch.zeros_like(eta));torch.testing.assert_close(a,torch.zeros_like(a))
    # A traceless tensor can encode two symmetric polar modes without selecting
    # an arbitrary oriented normal to a plane.
    matrix=torch.diag(torch.tensor([-1.,-1.,2.],dtype=x.dtype))[None]
    north=torch.tensor([[0.,0.,1.]],dtype=x.dtype);east=torch.tensor([[1.,0.,0.]],dtype=x.dtype)
    torch.testing.assert_close(angular_log_score(north,eta,matrix),angular_log_score(-north,eta,matrix))
    assert angular_log_score(north,eta,matrix)>angular_log_score(east,eta,matrix)


def test_capped_unknown_normalizer_cancels_only_with_invariant_context():
    # Exact finite analogue of the uniform-sphere/score/support rejection loop.
    pi=torch.tensor([.2,.3,.5],dtype=torch.float64);score=torch.tensor([-.2,-1.,-.6],dtype=pi.dtype)
    def kernel(scores):
        weights=scores.exp()/4  # Fourth proposal state is outside hard support.
        z=weights.sum(1);factor=sum((1-z)**j for j in range(7));q=factor[:,None]*weights
        # Deliberately omit factor(Z)'s ratio, as the valid constant-context kernel does.
        logratio=pi.log()[None]-pi.log()[:,None]+scores.T-scores
        flow=q*logratio.clamp_max(0).exp();flow=flow-torch.diag(flow.diagonal())
        return flow+torch.diag(1-flow.sum(1))
    invariant=score[None].expand(3,-1);p=kernel(invariant)
    torch.testing.assert_close(pi@p,pi,atol=1e-12,rtol=0)
    leaking=invariant.clone();leaking[0,1]-=2;leaking[1,2]-=1
    assert float((pi@kernel(leaking)-pi).abs().max())>1e-3
