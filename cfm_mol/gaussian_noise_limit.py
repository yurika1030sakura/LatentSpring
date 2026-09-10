"""Population-ESS ceiling for Gaussian targets and a fixed final Gaussian noise."""
import math


def gaussian_noise_ess_limit(target_variances,noise_variance):
    """Sharp over all Gaussian location mixtures; not a finite-sample ESS bound.

    Target covariance eigenvalues must be positive. Molecular Hessian-derived
    values only describe a local harmonic surrogate, not the full target.
    """
    values=list(target_variances)
    if not values or any(not math.isfinite(v) or v<=0 for v in values):
        raise ValueError('Positive finite target covariance eigenvalues required')
    if not math.isfinite(noise_variance) or noise_variance<=0:raise ValueError('Positive finite noise variance required')
    ratios=[max(1.,noise_variance/v) for v in values]
    log_limit=sum(.5*math.log(2*r-1)-math.log(r) for r in ratios)
    return {'population_ess_fraction_limit':math.exp(log_limit),'log_limit':log_limit,
        'unresolved_directions':sum(r>1 for r in ratios)}
