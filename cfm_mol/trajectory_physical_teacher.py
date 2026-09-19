"""Local force/work moments at the frozen generator's provisional endpoints.

The target is a Gaussian restrained energy tilt on a fixed centered ball,
not a global molecular Boltzmann law. Translation escorts have determinant one.
Antithetic pairs eliminate proposal noise from the unweighted force control.
"""
import torch
from .escorted_thermal_teacher import local_work_log_weights


def candidates(anchor,force,*,kT,particles,max_sigma,max_shift,seed):
    if particles<2 or particles%2 or min(kT,max_sigma,max_shift)<=0:
        raise ValueError('Even particle count and positive physical scales required')
    force=force-force.mean(-2,keepdim=True)
    norm=force.square().sum((-1,-2)).sqrt()
    sigma=torch.minimum(torch.full_like(norm,max_sigma),torch.sqrt(max_shift*kT/norm.clamp_min(1e-12)))
    shift=sigma[:,None,None].square()*force/kT
    noise=torch.randn((len(anchor),particles//2,*anchor.shape[1:]),dtype=anchor.dtype,
                      generator=torch.Generator().manual_seed(seed))
    noise-=noise.mean(-2,keepdim=True);noise=torch.cat([noise,-noise],dim=1)
    source=anchor[:,None]+sigma[:,None,None,None]*noise
    return source,source+shift[:,None],sigma,shift


def targets(anchor,source,proposal,sigma,shift,anchor_energy,proposal_energy,*,kT,radius):
    if radius<=0:raise ValueError('Positive target support radius required')
    support=(proposal-anchor[:,None]).square().sum((-1,-2))<=radius**2
    logw=local_work_log_weights(anchor,source,proposal,anchor_energy,proposal_energy,sigma,kT,support)
    eligible=support.any(1);weights=torch.zeros_like(logw)
    weights[eligible]=logw[eligible].softmax(-1)
    work=(weights[:,:,None,None]*(proposal-anchor[:,None])).sum(1)
    # Retain unsupported states with a declared force fallback; never drop them.
    work[~eligible]=shift[~eligible];work-=work.mean(-2,keepdim=True)
    ess=torch.zeros_like(sigma);ess[eligible]=1/weights[eligible].square().sum(-1)
    return dict(force_shift=shift,work_shift=work,support=support,eligible=eligible,
                weights=weights,log_weights=logw,ess=ess)


def velocity_target(shift,t,*,velocity_scale,gate_power):
    if ((t<=0)|(t>=1)).any() or velocity_scale<=0 or gate_power<1:
        raise ValueError('Interior flow times and positive velocity gate required')
    shift=shift-shift.mean(-2,keepdim=True)
    velocity=shift/(1-t[:,None,None]);bound=velocity_scale*t.pow(gate_power)
    scale=(bound/velocity.norm(dim=-1).amax(-1).clamp_min(1e-12)).clamp_max(1.)
    return velocity*scale[:,None,None],scale
