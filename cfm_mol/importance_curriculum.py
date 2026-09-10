"""Explicit empirical weight controls for a prospective work-weighted FM study.

Linear damping preserves a known endpoint mixture in the population identity.
Power tempering of latent path weights generally has a different endpoint law.
Neither operation establishes finite-pool calibration of the final target.
"""
import math

import torch


def weight_controls(log_weights,minimum_ess_fraction=.5):
    values=torch.as_tensor(log_weights).detach().double()
    if values.ndim!=1 or len(values)<2 or not torch.isfinite(values).all():raise ValueError('Finite log weights for at least two samples required')
    if not math.isfinite(minimum_ess_fraction) or not 0<minimum_ess_fraction<=1:raise ValueError('ESS fraction must lie in (0,1]')
    n=len(values);raw=torch.softmax(values,0);uniform=torch.full_like(raw,1/n)
    raw_square=float(raw.square().sum());target=minimum_ess_fraction*n
    excess=max(0.,raw_square-1/n)
    eta=1. if excess<1e-15 else min(1.,math.sqrt(max(0.,1/target-1/n)/excess))
    linear=(1-eta)*uniform+eta*raw
    if 1/raw_square>=target-1e-10:power=1.
    else:
        low,high=0.,1.
        for _ in range(64):
            middle=(low+high)/2;weights=torch.softmax(middle*values,0)
            if float(weights.square().sum().reciprocal())>=target:low=middle
            else:high=middle
        power=low
    tempered=torch.softmax(power*values,0)
    return {'uniform':uniform,'full':raw,'linear':linear,'power':tempered},{
        'particles':n,'raw_ess':1/raw_square,'minimum_ess_fraction':minimum_ess_fraction,
        'linear_target_fraction':eta,'power_exponent':power,
        'ess':{name:float(w.square().sum().reciprocal()) for name,w in [('uniform',uniform),('full',raw),('linear',linear),('power',tempered)]}}
