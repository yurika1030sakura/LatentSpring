import copy

import torch

from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.pair_precision import pair_precision_matrix,LearnedPairPrecision
from cfm_mol.matrix_gaussian import precision_cholesky,gaussian_precision_sample,gaussian_precision_log_density


def test_spectral_bounds_rigid_rotations_and_geometric_equivariance():
    torch.manual_seed(9073);n=5;x=torch.randn(2,n,3,dtype=torch.float64);x-=x.mean(1,keepdim=True)
    model=LearnedPairPrecision().double();numbers=torch.tensor([6,1,8,1,7]);times=torch.tensor([.3,.8],dtype=torch.float64)
    with torch.no_grad():model.network[-1].weight.normal_(std=.2)
    precision=model(x,times,numbers,0,2,.025852);eigen=torch.linalg.eigvalsh(precision)
    assert float(eigen.min())>=1-1e-10 and float(eigen.max())<=33+1e-10
    basis=centered_orthonormal_basis(n);tangent=torch.kron(basis.contiguous(),torch.eye(3,dtype=torch.float64))
    rotation,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64));perm=torch.tensor([2,4,0,3,1])
    change=tangent.T@torch.kron(torch.eye(n,dtype=torch.float64)[perm],rotation.contiguous())@tangent
    changed=model(x[:,perm]@rotation.T+3.,times,numbers[perm],0,2,.025852)
    torch.testing.assert_close(changed,change@precision@change.T,atol=2e-12,rtol=2e-12)
    omega=torch.tensor([.3,-.7,.4],dtype=torch.float64).expand_as(x)
    rotation_vector=(torch.cross(omega,x,dim=-1).flatten(1)@tangent)
    torch.testing.assert_close((precision@rotation_vector[...,None]).squeeze(-1),rotation_vector,rtol=1e-11,atol=1e-11)
    collapsed=torch.zeros_like(x);unit=model(collapsed,times,numbers,0,2,.025852)
    torch.testing.assert_close(unit,torch.eye(12,dtype=torch.float64)[None].expand(2,-1,-1),atol=0,rtol=0)


def test_trace_control_preserves_total_variance_and_parameters_get_gradients():
    torch.manual_seed(9074);x=torch.randn(2,4,3,dtype=torch.float64,requires_grad=True);numbers=[6,1,8,1]
    full=LearnedPairPrecision().double();trace=copy.deepcopy(full);trace.mode='trace'
    p=full(x,.6,numbers,0,1,.025852);q=trace(x,.6,numbers,0,1,.025852)
    torch.testing.assert_close(torch.linalg.inv(p).diagonal(dim1=-2,dim2=-1).sum(-1),torch.linalg.inv(q).diagonal(dim1=-2,dim2=-1).sum(-1),rtol=1e-12,atol=1e-12)
    loss=torch.linalg.slogdet(p)[1].sum();loss.backward()
    assert torch.isfinite(x.grad).all() and float(x.grad.norm())>0
    assert float(full.network[-1].weight.grad.norm())>0
    # Independent coordinate finite differences include the learned geometry metric.
    direction=torch.randn_like(x);autodiff=float((x.grad*direction).sum());h=1e-5
    plus=torch.linalg.slogdet(full(x.detach()+h*direction,.6,numbers,0,1,.025852))[1].sum()
    minus=torch.linalg.slogdet(full(x.detach()-h*direction,.6,numbers,0,1,.025852))[1].sum()
    torch.testing.assert_close(torch.tensor(autodiff),((plus-minus)/(2*h)).float(),atol=2e-5,rtol=2e-5)


def test_full_precision_gaussian_matches_distribution_api_and_covariance():
    generator=torch.Generator().manual_seed(9075);matrix=torch.randn(3,4,4,dtype=torch.float64,generator=generator)
    precision=matrix@matrix.transpose(-1,-2)+torch.eye(4,dtype=torch.float64);chol=precision_cholesky(precision)
    mean=torch.randn(3,4,dtype=torch.float64,generator=generator);value=torch.randn(3,4,dtype=torch.float64,generator=generator);std=.3
    expected=torch.distributions.MultivariateNormal(mean,precision_matrix=precision/std**2).log_prob(value)
    torch.testing.assert_close(gaussian_precision_log_density(value,mean,chol,std),expected,atol=1e-11,rtol=1e-11)
    count=65536;noise=torch.randn(count,4,dtype=torch.float64,generator=generator)
    draws=gaussian_precision_sample(mean[:1].expand(count,-1),chol[:1].expand(count,-1,-1),std,noise)
    covariance=torch.cov((draws-mean[:1]).T)
    torch.testing.assert_close(covariance,std**2*torch.linalg.inv(precision[0]),rtol=.03,atol=.001)
