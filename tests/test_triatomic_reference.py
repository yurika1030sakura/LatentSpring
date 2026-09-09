import math

import numpy as np
from scipy.special import logsumexp

from cfm_mol.triatomic_reference import gaussian_triatomic_shapes,centered_gaussian_mixture_log_density


def test_reduced_gaussian_shape_has_correct_intrinsic_moments():
    x=gaussian_triatomic_shapes(16,.8,43)
    norm2=(x*x).sum((1,2))/.8**2
    assert np.max(np.abs(x.sum(1)))<1e-13
    assert abs(norm2.mean()-6)<.003
    assert abs(norm2.var()-12)<.06
    for i,j in [(0,1),(0,2),(1,2)]:
        assert abs(((x[:,i]-x[:,j])**2).sum(-1).mean()-6*.8**2)<.003


def test_multiple_importance_reference_recovers_harmonic_partition():
    scales=[.5,1.,math.sqrt(10)]
    x=np.concatenate([gaussian_triatomic_shapes(14,s,74+i) for i,s in enumerate(scales)])
    log_target=-.05*(x*x).sum((1,2))
    logw=log_target-centered_gaussian_mixture_log_density(x,scales)
    logz=logsumexp(logw)-math.log(len(x))
    expected=3*math.log(2*math.pi/.1)
    assert abs(logz-expected)<.004
