import math

import pytest
import torch

from cfm_mol.rotation_mixture import matrix_fisher_normalizer,random_rotations,RotatedGaussianMixture,symmetry_templates
from cfm_mol.nonequilibrium import centered_orthonormal_basis


def test_isotropic_matrix_fisher_against_bessel_closed_form():
    s=torch.tensor([0.,.1,1.,10.,100.],dtype=torch.float64)
    f=s[:,None,None]*torch.eye(3,dtype=torch.float64)
    value,mean,diag=matrix_fisher_normalizer(f)
    expected=3*s+torch.log(torch.special.i0e(2*s)-torch.special.i1e(2*s))
    torch.testing.assert_close(value,expected,rtol=1e-12,atol=1e-11)
    torch.testing.assert_close(mean,torch.diag_embed(mean.diagonal(dim1=-2,dim2=-1)),atol=1e-12,rtol=0)
    assert diag['last_log_refinement_change']<=1e-10


def test_mean_rotation_matches_finite_difference_including_negative_determinant():
    f=torch.randn(4,3,3,dtype=torch.float64,generator=torch.Generator().manual_seed(5))
    _,mean,_=matrix_fisher_normalizer(f)
    h=1e-5
    for i in range(3):
        for j in range(3):
            delta=torch.zeros_like(f);delta[:,i,j]=h
            plus=matrix_fisher_normalizer(f+delta)[0];minus=matrix_fisher_normalizer(f-delta)[0]
            torch.testing.assert_close((plus-minus)/(2*h),mean[:,i,j],rtol=1e-7,atol=1e-9)


def test_normalizer_and_mean_against_independent_haar_monte_carlo():
    f=torch.tensor([[[.4,.2,-.1],[.3,-.5,.1],[.2,.1,.6]]],dtype=torch.float64)
    rotations=random_rotations(150000,torch.Generator().manual_seed(333))
    exponent=(rotations*f).sum((1,2))
    reference=torch.logsumexp(exponent,0)-math.log(len(rotations))
    reference_mean=torch.einsum('b,bij->ij',torch.softmax(exponent,0),rotations)
    value,mean,_=matrix_fisher_normalizer(f)
    torch.testing.assert_close(value[0],reference,rtol=0,atol=.006)
    torch.testing.assert_close(mean[0],reference_mean,rtol=0,atol=.006)


def test_diatomic_rotated_gaussian_density_integrates_to_one():
    model=RotatedGaussianMixture(torch.tensor([[1.3,0.,0.]],dtype=torch.float64),.45)
    radius=torch.linspace(0,8,4001,dtype=torch.float64)
    x=torch.stack([radius,torch.zeros_like(radius),torch.zeros_like(radius)],-1)
    density=model(x).log_value.exp()
    integral=torch.trapz(4*math.pi*radius.square()*density,radius)
    torch.testing.assert_close(integral,torch.tensor(1.,dtype=torch.float64),rtol=0,atol=1e-10)


def test_density_rotation_invariance_and_analytic_score():
    g=torch.Generator().manual_seed(12)
    centers=torch.randn(3,9,dtype=torch.float64,generator=g)
    model=RotatedGaussianMixture(centers,.6)
    x=torch.randn(6,9,dtype=torch.float64,generator=g)
    result=model(x);r=random_rotations(1,g)[0]
    rotated=(x.reshape(6,-1,3)@r).reshape_as(x);other=model(rotated)
    torch.testing.assert_close(result.log_value,other.log_value,rtol=0,atol=1e-10)
    torch.testing.assert_close((result.score.reshape(6,-1,3)@r).reshape_as(x),other.score,rtol=1e-10,atol=1e-10)
    for axis in range(9):
        delta=torch.zeros_like(x);delta[:,axis]=1e-5
        finite_difference=(model(x+delta).log_value-model(x-delta).log_value)/2e-5
        torch.testing.assert_close(finite_difference,result.score[:,axis],rtol=1e-6,atol=1e-8)


def test_haar_sampler_diatomic_moments():
    model=RotatedGaussianMixture(torch.tensor([[1.3,0.,0.]],dtype=torch.float64),.45)
    x,_=model.sample(70000,torch.Generator().manual_seed(42))
    torch.testing.assert_close(x.mean(0),torch.zeros(3,dtype=torch.float64),rtol=0,atol=.012)
    expected=(.45**2+1.3**2/3)*torch.eye(3,dtype=torch.float64)
    torch.testing.assert_close(x.T@x/len(x),expected,rtol=0,atol=.015)


def test_unconverged_density_is_rejected():
    with pytest.raises(RuntimeError,match='did not converge'):
        matrix_fisher_normalizer(100*torch.eye(3,dtype=torch.float64),start_order=2,max_order=4,tolerance=1e-12)


def test_full_identical_atom_permutations_and_reflections_preserve_density():
    g=torch.Generator().manual_seed(82);basis=centered_orthonormal_basis(4)
    centers=torch.randn(2,9,dtype=torch.float64,generator=g)
    augmented,meta=symmetry_templates(centers,basis,[1,8,1,8],reflect=True)
    assert meta['used_permutations']==4 and meta['all_permutations_enumerated']
    assert len(augmented)==16
    model=RotatedGaussianMixture(augmented,.7)
    x=torch.randn(5,9,dtype=torch.float64,generator=g)
    cartesian=torch.einsum('nk,bkd->bnd',basis,x.reshape(5,3,3))
    permuted=-cartesian[:,[2,3,0,1],:]
    transformed=torch.einsum('nk,bnd->bkd',basis,permuted).reshape_as(x)
    torch.testing.assert_close(model(x).log_value,model(transformed).log_value,atol=1e-10,rtol=0)


def test_capped_permutation_subset_does_not_claim_full_invariance():
    basis=centered_orthonormal_basis(8)
    centers=torch.ones(2,21,dtype=torch.float64)
    _,meta=symmetry_templates(centers,basis,[1]*8,max_permutations=3,reflect=False,seed=18)
    assert meta['used_permutations']==3
    assert not meta['all_permutations_enumerated']
