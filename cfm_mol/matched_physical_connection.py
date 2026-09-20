"""A common endpoint-displacement interpretation for FM and GAGA residuals.

The head predicts A in Angstrom per unit normalized progress. Its implied
endpoint displacement is (1-progress)*A. For FM this adds A to the velocity;
for GAGA it adds -alpha*(1-progress)*A/sigma to predicted noise. The latter
changes the decoded clean coordinates by exactly the same displacement.
This is a sampler-specific adaptation, not a common continuous-time dynamics.
"""
import math
import torch
from . import matched_egnn as base


def endpoint_and_progress(model, x, t, value, spec):
    if model.norm_values[0] != 1.:
        raise ValueError('This adapter uses coordinates in Angstrom')
    if spec['kind'] in ('harmonic_fm', 'gaussian_fm'):
        progress=t[:, 0]
        endpoint=x+(1-progress[:, None, None])*value
    elif spec['kind'] in ('gaga', 'edm'):
        gamma=model.gamma(t)
        endpoint=(x-model.sigma(gamma,x)*value)/model.alpha(gamma,x)
        maximum=spec['gaga_max_t']/model.T if spec['kind']=='gaga' else 1.
        progress=1-t[:, 0]/maximum
    else:
        raise ValueError(spec['kind'])
    if not torch.isfinite(endpoint).all() or (progress < -1e-6).any() or (progress > 1+1e-6).any():
        raise FloatingPointError('Invalid provisional endpoint or sampling progress')
    return base.center(endpoint), progress.clamp(0,1)


def native_correction(model, x, t, correction, spec):
    if spec['kind'] in ('harmonic_fm', 'gaussian_fm'):
        return correction
    if spec['kind'] not in ('gaga', 'edm'):raise ValueError(spec['kind'])
    gamma=model.gamma(t)
    maximum=spec['gaga_max_t']/model.T if spec['kind']=='gaga' else 1.
    remaining=t[...,None]/maximum
    return -model.alpha(gamma,x)*remaining*correction/model.sigma(gamma,x)


def head_inputs(x, endpoint, numbers, progress, atomic_numbers):
    batch,n,_=x.shape
    lookup=torch.full((84,),-1,dtype=torch.long,device=x.device)
    lookup[torch.tensor(atomic_numbers,device=x.device)]=torch.arange(len(atomic_numbers),device=x.device)
    types=lookup[numbers.reshape(-1)]
    if (types<0).any():raise ValueError('Unsupported atom type')
    i,j=torch.where(~torch.eye(n,dtype=torch.bool,device=x.device))
    offsets=torch.arange(batch,device=x.device)[:,None]*n
    return (x.reshape(-1,3),endpoint.reshape(-1,3),types,progress,
            torch.arange(batch,device=x.device).repeat_interleave(n),
            (i[None]+offsets).reshape(-1),(j[None]+offsets).reshape(-1))


class PhysicalFieldTransform:
    """Apply the head after the final parent prediction, leaving the parent frozen."""
    def __init__(self,model,spec,head,strength=1.,*,strength_limit=1.):
        # The original protocol retains its [0,1] restriction. A separately
        # declared calibration may raise the limit; the vector bound scales
        # by the actual strength and is not the original unit-strength bound.
        if not math.isfinite(strength_limit) or strength_limit<=0 or not 0 <= strength <= strength_limit:
            raise ValueError('Strength must be finite and within the declared limit')
        self.model=model;self.spec=spec;self.head=head;self.strength=strength;self.calls=0
        model.eval().requires_grad_(False)

    def __call__(self,x,t,numbers,value):
        self.calls+=1
        endpoint,progress=endpoint_and_progress(self.model,x,t,value,self.spec)
        inputs=head_inputs(x,endpoint,numbers,progress,self.head.configuration['atomic_numbers'])
        correction=self.head(*inputs).reshape_as(x)*self.strength
        result=value+native_correction(self.model,x,t,correction,self.spec)
        if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite physical correction')
        return result
