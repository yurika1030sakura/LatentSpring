import math

import numpy as np
import pytest
import torch

from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.triatomic_reference import gaussian_triatomic_shapes,centered_gaussian_mixture_log_density
from cfm_mol.triatomic_shape_proposal import (to_shape_coordinates,from_shape_coordinates,
    log_shape_volume,TriatomicShapeMixture,defensive_shape_log_density)


def test_shape_volume_matches_full_six_coordinate_jacobian():
    u=torch.tensor([math.log(2.4),math.log(2.7),math.atanh(.2),.3,.7,1.1],dtype=torch.float64,requires_grad=True)
    basis=centered_orthonormal_basis(3)
    def rz(a):
        zero=a*0;one=zero+1;c=a.cos();s=a.sin()
        return torch.stack([c,-s,zero,s,c,zero,zero,zero,one]).reshape(3,3)
    def ry(a):
        zero=a*0;one=zero+1;c=a.cos();s=a.sin()
        return torch.stack([c,zero,s,zero,one,zero,-s,zero,c]).reshape(3,3)
    def transform(v):
        r1,r2=v[:2].exp();c=v[2].tanh();zero=v[0]*0
        x=torch.stack([zero,zero,zero,r1,zero,zero,r2*c,r2*torch.sqrt(1-c*c),zero]).reshape(3,3)
        x=x-x.mean(0);rotation=rz(v[3])@ry(v[4])@rz(v[5])
        return (basis.T@x@rotation.T).reshape(-1)
    jacobian=torch.autograd.functional.jacobian(transform,u)
    actual=float(torch.linalg.slogdet(jacobian)[1])
    expected=float(log_shape_volume(u.detach().numpy()[None,:3])[0]-math.log(8*math.pi**2)+math.log(math.sin(float(u[4]))))
    assert actual==pytest.approx(expected,abs=1e-10)


def test_shape_roundtrip_and_identical_outer_atom_symmetry():
    x=gaussian_triatomic_shapes(7,1.2,911)
    np.testing.assert_allclose(from_shape_coordinates(to_shape_coordinates(x)),x,atol=1e-12)
    centers=to_shape_coordinates(x[:8]);centers=np.concatenate([centers,centers[:,[1,0,2]]])
    proposal=TriatomicShapeMixture(centers,(.3,.3,.6))
    np.testing.assert_allclose(proposal.log_density(x),proposal.log_density(x[:,[0,2,1]]),atol=1e-10)


def test_defensive_shape_quadrature_recovers_known_gaussian_normalizer_and_moment():
    local=TriatomicShapeMixture(np.array([[.3,.4,.2],[-.2,.8,-.4]]),(.5,.5,.8))
    normalizers=[];moments=[]
    for seed in [81,82,83,84]:
        x=np.concatenate([gaussian_triatomic_shapes(11,1.,seed),local.sample(11,seed+1000)])
        logq=defensive_shape_log_density(x,local,1.)
        weight=np.exp(centered_gaussian_mixture_log_density(x,[1.])-logq)
        normalizers.append(weight.mean());moments.append(np.mean(weight*(x*x).sum((1,2))))
    assert np.mean(normalizers)==pytest.approx(1.,abs=.012)
    assert np.mean(moments)==pytest.approx(6.,abs=.08)
