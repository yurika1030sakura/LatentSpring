"""Reuse predicted geometry between calls of the independently trained EGNN.

This defines a stateful sampler, not the original memoryless two-pass ODE.
One bootstrap and one Euler interval, then midpoint intervals, use exactly
the declared even call budget. No physical oracle, supplied bonds or data
coordinates are used. Geometry-feedback reuse is an established idea; its
utility for these frozen models must be evaluated.
"""
import numpy as np
import torch
from . import matched_egnn as base
from .connectivity_feedback import use_context


@torch.no_grad()
def sample_cached(model,numbers,spec,source,context,seed,batch,calls=128):
    if spec['kind'] not in ['harmonic_fm','gaussian_fm'] or not spec['two_pass']:
        raise ValueError('Cached sampler requires a two-pass-trained flow model')
    if calls<2 or calls%2:raise ValueError('Positive even network-call budget required')
    device=next(model.parameters()).device;z=torch.tensor(numbers,device=device)[None].expand(batch,-1)
    if spec['kind']=='harmonic_fm':
        rng=np.random.default_rng(seed);x=torch.stack([source.sample(numbers,rng) for _ in range(batch)]).to(device).float()
    else:
        rng=torch.Generator(device=device).manual_seed(seed)
        x=base.center(torch.randn((batch,len(numbers),3),generator=rng,device=device))
    initial=x.clone();used=0;steps=calls//2;dt=1/steps
    def field(state,t,geometry):
        nonlocal used
        with use_context(model,context(base.center(geometry),z)):
            value=base.vector(model,state,t,z,spec)
        used+=1;return value
    t=x.new_zeros(batch,1);first=field(x,t,x);geometry=base.center(x+first)
    for step in range(steps):
        t=x.new_full((batch,1),step/steps);v=field(x,t,geometry)
        geometry=base.center(x+(1-t[...,None])*v)
        if step:
            middle=base.center(x+.5*dt*v);tmid=t+.5*dt
            v=field(middle,tmid,geometry)
            geometry=base.center(middle+(1-tmid[...,None])*v)
        x=base.center(x+dt*v)
    assert used==calls
    if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite cached-feedback output')
    return x*model.norm_values[0],initial
