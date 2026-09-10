"""Fixed-path Gaussian likelihood factors for standard off-policy balance losses.

States are observations here and are deliberately detached. This computes
parameter scores of path probabilities, not pathwise gradients through sampling
and not a marginal endpoint likelihood. All time slices share two batched drift
calls, whose rows must remain independent.
"""
import math

import torch

from cfm_mol.nonequilibrium import _states,_schedule,gaussian_log_density
from cfm_mol.path_work import gaussian_reference_step


def fixed_path_log_factors(states,forward_drift,backward_drift,times,noise_scale,*,prior_std=1.,
                          terminal_std=None,max_drift_norm=None,mean_parameterization='reference',noise_annealing_power=0.):
    grid=_schedule(times,'times');x=states.detach().double()
    if x.ndim!=3 or x.shape[1]!=len(grid):raise ValueError('Require [batch, times, dimension] observed paths')
    _states(x.flatten(0,1))
    if any(not math.isfinite(v) or v<=0 for v in [noise_scale,prior_std]):raise ValueError('Invalid path scales')
    if terminal_std is not None and (not math.isfinite(terminal_std) or terminal_std<=0):raise ValueError('Invalid terminal scale')
    if max_drift_norm is not None and (not math.isfinite(max_drift_norm) or max_drift_norm<=0):raise ValueError('Invalid drift bound')
    if noise_annealing_power<0 or not math.isfinite(noise_annealing_power):raise ValueError('Invalid noise schedule')
    if mean_parameterization not in ['reference','native']:raise ValueError('Unknown mean parameterization')
    batch,_,dimension=x.shape;steps=len(grid)-1
    left=x[:,:-1].reshape(-1,dimension);right=x[:,1:].reshape_as(left)
    coefficients=x.new_tensor([gaussian_reference_step(a,b,noise_scale,prior_std,terminal_std,noise_annealing_power) for a,b in zip(grid,grid[1:])])
    dt,af,ab,sf,sb=coefficients.repeat(batch,1).unbind(-1)
    t_left=x.new_tensor(grid[:-1]).repeat(batch);t_right=x.new_tensor(grid[1:]).repeat(batch)
    vf=forward_drift(left,t_left);vb=backward_drift(right,t_right)
    if vf.shape!=left.shape or vb.shape!=right.shape or not torch.isfinite(vf).all() or not torch.isfinite(vb).all():raise ValueError('Invalid batched path drift')
    if mean_parameterization=='native':
        vf=vf+((1-af)/dt)[:,None]*left;vb=vb+((1-ab)/dt)[:,None]*right
    if max_drift_norm is not None:
        vf=vf*max_drift_norm/torch.sqrt(max_drift_norm**2+vf.square().sum(-1,keepdim=True))
        vb=vb*max_drift_norm/torch.sqrt(max_drift_norm**2+vb.square().sum(-1,keepdim=True))
    mean_f=af[:,None]*left+dt[:,None]*vf;mean_b=ab[:,None]*right+dt[:,None]*vb
    def log_normal(value,mean,std):
        return -.5*((value-mean)/std[:,None]).square().sum(-1)-dimension*torch.log(std)-.5*dimension*math.log(2*math.pi)
    forward=log_normal(right,mean_f,sf).reshape(batch,steps).sum(-1)
    backward=log_normal(left,mean_b,sb).reshape(batch,steps).sum(-1)
    initial=gaussian_log_density(x[:,0],torch.zeros_like(x[:,0]),prior_std)
    return initial,forward,backward


def log_variance_balance(work):
    """Per-condition empirical log-variance objective; no cross-condition log-Z mixing."""
    if work.ndim!=1 or len(work)<2 or not torch.isfinite(work).all():raise ValueError('At least two finite same-condition path works required')
    return (work-work.mean()).square().mean()


def fixed_backward_residuals(states,backward_drift,times,noise_scale,*,prior_std=1.,
                             terminal_std=None,max_drift_norm=None,mean_parameterization='reference',noise_annealing_power=0.):
    """Per-transition squared residuals in reference backward-noise units."""
    grid=_schedule(times,'times');x=states.detach().double()
    if x.ndim!=3 or x.shape[1]!=len(grid):raise ValueError('Require observed [batch,times,dimension] paths')
    _states(x.flatten(0,1));batch,_,dimension=x.shape;steps=len(grid)-1
    if any(not math.isfinite(v) or v<=0 for v in [noise_scale,prior_std]):raise ValueError('Invalid path scales')
    if terminal_std is not None and (not math.isfinite(terminal_std) or terminal_std<=0):raise ValueError('Invalid terminal scale')
    if max_drift_norm is not None and (not math.isfinite(max_drift_norm) or max_drift_norm<=0):raise ValueError('Invalid drift bound')
    if mean_parameterization not in ['reference','native'] or noise_annealing_power<0 or not math.isfinite(noise_annealing_power):raise ValueError('Invalid path convention')
    left=x[:,:-1].reshape(-1,dimension);right=x[:,1:].reshape_as(left)
    coefficients=x.new_tensor([gaussian_reference_step(a,b,noise_scale,prior_std,terminal_std,noise_annealing_power) for a,b in zip(grid,grid[1:])])
    dt,_,ab,_,sb=coefficients.repeat(batch,1).unbind(-1)
    drift=backward_drift(right,x.new_tensor(grid[1:]).repeat(batch))
    if drift.shape!=right.shape or not torch.isfinite(drift).all():raise ValueError('Invalid backward drift')
    if mean_parameterization=='native':drift=drift+((1-ab)/dt)[:,None]*right
    if max_drift_norm is not None:drift=drift*max_drift_norm/torch.sqrt(max_drift_norm**2+drift.square().sum(-1,keepdim=True))
    mean=ab[:,None]*right+dt[:,None]*drift
    residual=((left-mean)/sb[:,None]).square().sum(-1).reshape(batch,steps)
    log_base_normalization=-dimension*(sb.log()+.5*math.log(2*math.pi)).reshape(batch,steps)
    return residual,log_base_normalization


def backward_log_probability(residual,log_base_normalization,dimension,variance_ratios=None):
    """Normalized per-time Gaussian factors with optional positive variance ratios."""
    if residual.shape!=log_base_normalization.shape or residual.ndim!=2:raise ValueError('Invalid residual shape')
    ratio=residual.new_ones(residual.shape[1]) if variance_ratios is None else torch.as_tensor(variance_ratios,device=residual.device,dtype=residual.dtype)
    if ratio.shape!=(residual.shape[1],) or not torch.isfinite(ratio).all() or (ratio<=0).any():raise ValueError('Positive per-time variance ratios required')
    return (-.5*residual/ratio+log_base_normalization-.5*dimension*ratio.log()).sum(-1)
