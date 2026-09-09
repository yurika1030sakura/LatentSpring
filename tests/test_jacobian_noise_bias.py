"""Exact quadrature verifies the interpretation of common-probe loss bias."""
import itertools
import math
import pytest
import torch


@pytest.mark.parametrize('gaussian',[False,True])
def test_group_noise_equals_symmetric_jacobian_dispersion(gaussian):
    dtype=torch.float64
    generator=torch.Generator().manual_seed(93)
    jac=torch.randn((3,3,3),generator=generator,dtype=dtype)
    symmetric=(jac+jac.transpose(-1,-2))/2
    if gaussian:
        # Three-point Gaussian quadrature integrates these quartics exactly.
        support=[(-math.sqrt(3),1/6),(0.,2/3),(math.sqrt(3),1/6)]
        operator=symmetric
    else:
        support=[(-1.,.5),(1.,.5)]
        operator=symmetric-torch.diag_embed(symmetric.diagonal(dim1=-2,dim2=-1))
    expected=2*(operator-operator.mean(0)).square().sum((-1,-2)).mean()
    expectation=0.
    for points in itertools.product(support,repeat=3):
        probe=torch.tensor([p[0] for p in points],dtype=dtype)
        weight=math.prod(p[1] for p in points)
        error=torch.einsum('d,idk,k->i',probe,jac,probe)-jac.diagonal(dim1=-2,dim2=-1).sum(-1)
        expectation+=weight*(error-error.mean()).square().mean()
    torch.testing.assert_close(expectation,expected,rtol=1e-12,atol=1e-12)
