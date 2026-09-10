import math

import numpy as np
from scipy.special import logsumexp

from cfm_mol.triatomic_box_quadrature import triatomic_box_nodes


def test_direct_box_volume_and_independent_full_gaussian_normalizer():
    lower,upper=2.,3.2
    x,logw=triatomic_box_nodes(4,lower,upper)
    expected=16*math.pi**2/(3*math.sqrt(3))*((upper**3-lower**3)/3)**2
    np.testing.assert_allclose(np.exp(logsumexp(logw)),expected,rtol=1e-13)
    np.testing.assert_allclose(x.mean(1),0,atol=1e-15)
    # Eight-sigma radial truncation has negligible Gaussian mass. This tests
    # the absolute six-dimensional volume, including the orientation factor.
    x,logw=triatomic_box_nodes(40,0.,8.)
    logz=logsumexp(logw-.5*np.sum(x*x,axis=(1,2)))
    np.testing.assert_allclose(logz,3*math.log(2*math.pi),atol=2e-6,rtol=0)
