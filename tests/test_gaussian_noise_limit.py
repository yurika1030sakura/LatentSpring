import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp

from cfm_mol.gaussian_noise_limit import gaussian_noise_ess_limit


def test_gaussian_mixture_optimum_matches_independent_density_integrals():
    target_variance=.25;noise_variance=1.
    limit=gaussian_noise_ess_limit([target_variance],noise_variance)['population_ess_fraction_limit']
    assert limit==pytest.approx(math.sqrt(7)/4)
    def second_moment(means,weights):
        def integrand(x):
            logp=-.5*x*x/target_variance-.5*math.log(2*math.pi*target_variance)
            logq=logsumexp(np.log(weights)-.5*(x-means)**2/noise_variance)-.5*math.log(2*math.pi*noise_variance)
            return math.exp(2*logp-logq)
        return quad(integrand,-np.inf,np.inf,epsabs=1e-10)[0]
    assert second_moment(np.array([0.]),np.array([1.]))==pytest.approx(1/limit,rel=1e-10)
    for means,weights in [([-.5,.5],[.5,.5]),([-1.,0.,2.],[.2,.6,.2]),([.1],[1.])]:
        assert second_moment(np.array(means),np.array(weights))>1/limit
    # Soft directions can match the target exactly through the mixing law.
    assert gaussian_noise_ess_limit([.25,2.,3.],1.)['population_ess_fraction_limit']==pytest.approx(limit)
    assert gaussian_noise_ess_limit([2.,3.],1.)['population_ess_fraction_limit']==1.
