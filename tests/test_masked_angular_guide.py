import torch
from cfm_mol.masked_angular_guide import MaskedAngularGuide,masked_angular_context,angular_log_score,angular_surface_score
from cfm_mol.masked_angular_sampler import capped_directions,masked_angular_transition
from cfm_mol.chemical_sampler import ChemicalTarget


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


def test_capped_sphere_sampler_and_mh_preserve_a_known_supported_target():
    g=torch.Generator().manual_seed(132);n=4096
    eta=torch.tensor([0.,0.,2.],dtype=torch.float64).expand(n,-1);matrix=torch.zeros(n,3,3,dtype=eta.dtype)
    envelope=torch.full((n,),2.,dtype=eta.dtype)
    valid=lambda u,index:u[:,2]>0
    y,ok,trace=capped_directions(eta,matrix,envelope,valid,max_trials=64,generator=g)
    assert int(ok.sum())>4090
    expected=1/(1-torch.exp(torch.tensor(-2.,dtype=eta.dtype)))-.5
    assert abs(float(y[ok,2].mean()-expected))<.025
    raw=torch.randn(n,3,dtype=eta.dtype,generator=g);raw[:,2]=raw[:,2].abs();x=raw/raw.norm(dim=1,keepdim=True)
    for _ in range(8):
        y,ok,_=capped_directions(eta,matrix,envelope,valid,max_trials=64,generator=g)
        ratio=angular_log_score(x,eta,matrix)-angular_log_score(y,eta,matrix)
        take=ok&(torch.rand(n,dtype=eta.dtype,generator=g).log()<ratio.clamp_max(0))
        x=torch.where(take[:,None],y,x)
    assert abs(float(x[:,2].mean())-.5)<.025
    assert abs(float(x[:,2].square().mean())-1/3)<.025
    _,failed,records=capped_directions(eta[:2],matrix[:2],envelope[:2],lambda u,index:torch.zeros(len(u),dtype=torch.bool),max_trials=3,generator=g)
    assert not failed.any() and len(records)==3


def test_molecular_capped_transition_keeps_context_and_scores_only_one_candidate():
    class Oracle:
        evaluated=0
        def evaluate_chunked(self,x,max_request):
            self.evaluated+=len(x);return .5*x.square().sum((1,2)),-x
    target=ChemicalTarget(Oracle(),dict(numbers=[6,1,1,1,9],charge=0,spin_multiplicity=1),.5,.1)
    x=torch.tensor([[0.,0.,0.],[1.,1.,1.],[1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.]],dtype=torch.float64)
    x[1:4]*=1.09/3**.5;x[4]*=1.35/3**.5;x-=x.mean(0)
    states=target.evaluate([target.coordinate_state(x)],phase='initial');model=MaskedAngularGuide().double()
    with torch.no_grad():model.vector_weight.weight.normal_(0,.1);model.tensor_weight.weight.normal_(0,.1)
    g=torch.Generator().manual_seed(133);scored=0
    for _ in range(8):
        old=states[0];states,rows,search=masked_angular_transition(target,states,model,max_trials=16,generator=g,phase='test')
        row=rows[0];scored+=row['valid']
        assert len(search['geometry_checks'])<=16
        if row['valid']:
            new=target.states[row['new_state_id']]
            ratio=-float(new['potential_eV']-old['potential_eV'])/.5+row['old_angular_score']-row['new_angular_score']
            assert abs(row['log_acceptance_ratio']-ratio)<1e-10
            assert row['accepted']==(row['log_uniform']<min(0.,ratio))
        else:assert row['exhausted'] and not row['accepted']
    assert target.oracle.evaluated==2*(1+scored)
