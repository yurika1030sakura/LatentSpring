"""A common endpoint-displacement interpretation for FM and GAGA residuals.

The head predicts A in Angstrom per unit normalized progress. Its implied
endpoint displacement is (1-progress)*A. For FM this adds A to the velocity;
for GAGA it adds -alpha*(1-progress)*A/sigma to predicted noise. The latter
changes the decoded clean coordinates by exactly the same displacement.
This is a sampler-specific adaptation, not a common continuous-time dynamics.
"""
import torch
from . import matched_egnn as base


def endpoint_and_progress(model, x, t, value, spec):
    if model.norm_values[0] != 1.:
        raise ValueError('This adapter uses coordinates in Angstrom')
    if spec['kind'] in ('harmonic_fm', 'gaussian_fm'):
        progress=t[:, 0]
        endpoint=x+(1-progress[:, None, None])*value
    elif spec['kind']=='gaga':
        gamma=model.gamma(t)
        endpoint=(x-model.sigma(gamma,x)*value)/model.alpha(gamma,x)
        progress=1-t[:, 0]/(spec['gaga_max_t']/model.T)
    else:
        raise ValueError(spec['kind'])
    if not torch.isfinite(endpoint).all() or (progress < -1e-6).any() or (progress > 1+1e-6).any():
        raise FloatingPointError('Invalid provisional endpoint or sampling progress')
    return base.center(endpoint), progress.clamp(0,1)


def native_correction(model, x, t, correction, spec):
    if spec['kind'] in ('harmonic_fm', 'gaussian_fm'):
        return correction
    if spec['kind']!='gaga':raise ValueError(spec['kind'])
    gamma=model.gamma(t)
    remaining=t[...,None]/(spec['gaga_max_t']/model.T)
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
    def __init__(self,model,spec,head,strength=1.):
        if not 0 <= strength <= 1:raise ValueError('Strength must be in [0,1]')
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
