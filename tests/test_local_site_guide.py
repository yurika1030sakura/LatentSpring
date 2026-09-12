import pytest
import torch
from cfm_mol.local_site_guide import LocalSiteGuide,confinement_parameter
from cfm_mol.masked_angular_guide import masked_angular_context


def fixture():
    rng=torch.Generator().manual_seed(24801)
    x=torch.randn(2,7,3,dtype=torch.float64,generator=rng);x-=x.mean(1,keepdim=True)
    numbers=torch.tensor([6,16,9,1,8,15,17]);roots=torch.tensor([[2,0],[3,0]])
    bonds=torch.zeros(2,7,7,dtype=x.dtype)
    for a,b in [(0,1),(0,2),(0,3),(1,4),(1,5),(1,6)]:bonds[:,a,b]=bonds[:,b,a]=1
    electronic=x.new_tensor([0.,1.,.025851999786435])
    return x,bonds,numbers,roots,electronic,rng


def test_confinement_angular_coefficient_matches_full_centered_energy_derivative():
    x,_,_,roots,electronic,_=fixture();masked,radius,_=masked_angular_context(x,roots)
    eta=confinement_parameter(masked,radius,electronic[2].expand(len(x)),.1)
    for i,(leaf,anchor) in enumerate(roots.tolist()):
        u=((x[i,leaf]-x[i,anchor])/radius[i]).detach().requires_grad_()
        y=masked[i].clone();y[leaf]=radius[i]*u;y=y-y.mean(0)
        log_target=-.05*y.square().sum()/electronic[2]
        grad=torch.autograd.grad(log_target,u)[0]
        tangent=grad-(grad*u).sum()*u
        expected=eta[i]-(eta[i]*u).sum()*u
        torch.testing.assert_close(tangent,expected,atol=1e-10,rtol=1e-10)


def test_nonzero_local_residual_is_unchanged_by_distant_spectators():
    # Representation probe outside the chemical support; it is not a molecular
    # performance test. Only the residual is local; the known restraint changes.
    x,bonds,numbers,roots,electronic,rng=fixture()
    torch.manual_seed(24802);model=LocalSiteGuide(cutoff=4.).double()
    with torch.no_grad():model.center_weight.weight.normal_(0,.3);model.center_weight.bias.fill_(.2)
    first=model.components(x,bonds,numbers,electronic,roots)
    assert first[2].norm()>1e-6
    far=torch.randn(2,8,3,dtype=x.dtype,generator=rng)+40
    enlarged=torch.cat([x,far],1);enlarged-=enlarged.mean(1,keepdim=True)
    bb=torch.zeros(2,15,15,dtype=x.dtype);bb[:,:7,:7]=bonds
    z=torch.cat([numbers,torch.tensor([6,1]*4)])
    second=model.components(enlarged,bb,z,electronic,roots)
    torch.testing.assert_close(first[0],second[0],atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(first[2],second[2],atol=1e-10,rtol=1e-10)
    assert not torch.allclose(first[1],second[1])


def test_masking_covariance_permutation_and_zero_head_control():
    x,bonds,numbers,roots,electronic,rng=fixture();torch.manual_seed(24803)
    model=LocalSiteGuide().double();site,harmonic,residual=model.components(x,bonds,numbers,electronic,roots)
    assert torch.equal(residual,torch.zeros_like(residual))
    p,w=model(x,bonds,numbers,electronic,roots);torch.testing.assert_close(p[:,0],site+harmonic)
    with torch.no_grad():model.center_weight.weight.normal_(0,.3)
    first,_=model(x,bonds,numbers,electronic,roots)
    moved=x.clone()
    for i,(leaf,anchor) in enumerate(roots.tolist()):
        v=torch.randn(3,dtype=x.dtype,generator=rng);v/=v.norm()
        moved[i,leaf]=x[i,anchor]+(x[i,leaf]-x[i,anchor]).norm()*v
    moved-=moved.mean(1,keepdim=True)
    after,_=model(moved,bonds,numbers,electronic,roots)
    torch.testing.assert_close(after,first,atol=1e-10,rtol=1e-10)
    perm=torch.tensor([6,3,1,4,0,2,5]);lookup=torch.argsort(perm)
    rotation=torch.linalg.qr(torch.randn(3,3,dtype=x.dtype,generator=rng))[0]
    rotation[:,0]*=-torch.linalg.det(rotation)
    rotated,_=model(x[:,perm]@rotation,bonds[:,perm][:,:,perm],numbers[perm],electronic,lookup[roots])
    torch.testing.assert_close(rotated,first@rotation,atol=1e-10,rtol=1e-10)
    with pytest.raises(ValueError):model(x,bonds,numbers,electronic*0,roots)


def test_stiffness_can_exceed_old_cap_without_rotating_intrinsic_mode():
    import math
    from cfm_mol.local_site_guide import StiffnessSiteGuide
    from cfm_mol.spherical_proposal import vmf_sample,vmf_log_prob
    x,bonds,numbers,roots,electronic,_=fixture()
    model=StiffnessSiteGuide().double()
    d0,k0,h=model.intrinsic_parameters(x,bonds,numbers,electronic,roots)
    torch.testing.assert_close(k0,torch.full_like(k0,10.),atol=1e-5,rtol=0)
    with torch.no_grad():model.concentration_head.bias.fill_(math.log(512/(2048-512)))
    d1,k1,_=model.intrinsic_parameters(x,bonds,numbers,electronic,roots)
    torch.testing.assert_close(d0,d1);torch.testing.assert_close(k1,torch.full_like(k1,512.))
    eta,_=model(x,bonds,numbers,electronic,roots)
    direction,_=vmf_sample(eta[:,0],generator=torch.Generator().manual_seed(24911))
    assert torch.isfinite(vmf_log_prob(direction,eta[:,0])).all()
    # Concentrated-vMF angular second moment has exact E[cos(theta)] below.
    e=torch.tensor([[0.,0.,1024.]],dtype=torch.float64).expand(8192,-1)
    draws,_=vmf_sample(e,generator=torch.Generator().manual_seed(24912))
    assert abs(float(draws[:,2].mean())-(1-1/1024))<5e-5
    # Likelihood training through the stable high-concentration normalizer.
    loss=-vmf_log_prob(direction.detach(),eta[:,0]).mean();loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
