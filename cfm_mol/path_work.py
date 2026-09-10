"""Differentiable finite Gaussian paths for mean generalized-work training.

This is the standard path-space reverse-KL/SNF objective, not a new identity.
Unlike an exponentiated noisy CNF likelihood, every discrete proposal factor
is explicitly evaluated. Learned forward/backward drifts must act independently
on rows. External energy forces provide first derivatives only.
"""
from dataclasses import dataclass
import math

import torch
from torch.autograd.function import once_differentiable
from torch.utils.checkpoint import checkpoint

from cfm_mol.nonequilibrium import _states,_schedule,gaussian_log_density
from cfm_mol.matrix_gaussian import precision_cholesky,gaussian_precision_sample,gaussian_precision_log_density


class ExternalEnergy(torch.autograd.Function):
    """Attach a frozen oracle's exact-value/force pair to its input coordinates.

    The caller must evaluate this very coordinate tensor with the declared
    charge/spin and potential. Higher energy derivatives are not available.
    """
    @staticmethod
    def forward(ctx,positions,energy,force):
        if energy.shape!=(len(positions),) or force.shape!=positions.shape:
            raise ValueError('Oracle energy/force shapes disagree with positions')
        if not torch.isfinite(energy).all() or not torch.isfinite(force).all():
            raise ValueError('Non-finite oracle energy/force')
        ctx.save_for_backward(force.detach().to(positions))
        return energy.detach().to(device=positions.device,dtype=torch.float64)

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        force,=ctx.saved_tensors
        shape=(len(force),)+(1,)*(force.ndim-1)
        return -force*gradient.to(force).reshape(shape),None,None


def external_energy(positions,energy,force):
    return ExternalEnergy.apply(positions,energy,force)


@dataclass
class TrainablePath:
    terminal: torch.Tensor
    log_initial: torch.Tensor
    log_forward_minus_backward: torch.Tensor
    states: torch.Tensor | None = None

    def work(self,reduced_terminal_energy):
        if reduced_terminal_energy.shape!=self.log_initial.shape:
            raise ValueError('One terminal energy required per path')
        value=reduced_terminal_energy.double()+self.log_initial+self.log_forward_minus_backward
        if not torch.isfinite(value).all():raise FloatingPointError('Non-finite generalized work')
        return value


def gaussian_reference_step(left,right,noise_scale,prior_std,terminal_std,noise_annealing_power=0.):
    """Scalar transition coefficients shared by simulation and fixed-path scoring."""
    dt=right-left;step_noise=noise_scale*(1-left)**noise_annealing_power
    std=step_noise*math.sqrt(dt)
    if terminal_std is None:return dt,1.,1.,std,std
    left_scale=prior_std*(terminal_std/prior_std)**left
    right_scale=prior_std*(terminal_std/prior_std)**right
    correlation=math.exp(-.5*step_noise**2*dt)
    relative_noise=math.sqrt(-math.expm1(-step_noise**2*dt))
    return dt,correlation*right_scale/left_scale,correlation*left_scale/right_scale,right_scale*relative_noise,left_scale*relative_noise


def gaussian_training_path(x0,forward_drift,backward_drift,times,noise_scale,generator,*,
                           prior_std=1.,checkpoint_steps=False,terminal_std=None,max_drift_norm=None,
                           forward_energy_only=False,mean_parameterization='reference',noise_annealing_power=0.,retain_states=False,
                           forward_precision=None,backward_precision=None):
    """Simulate reparameterized paths and retain complete first-order gradients.

    x0 must be an independent draw from the stated isotropic Gaussian.
    With terminal_std=None, fixed noise_scale*sqrt(dt) is used in both
    directions (the original Euler control). Otherwise exact Gaussian reference
    kernels connect geometric standard deviations from prior_std to terminal_std.
    Optional precision callbacks replace scalar covariances by std^2 P^-1
    using fully normalized Gaussian factors. State-dependent P does not retain
    the Gaussian-reference reversibility property; the finite-path identity
    still applies. Both covariance and mean path derivatives are retained.
    Their scalar reference correlation is exp(-noise_scale^2*dt/2). Neural drifts add residual
    mean shifts. With zero residual, the reference marginal and its reverse
    conditional are exact at any step count.
    Backward drift is an auxiliary normalized kernel, not physical time reversal.
    Noise is generated once and reused exactly during checkpoint recomputation.
    forward_energy_only is an intentional gradient ablation: work values are
    unchanged, but path-factor gradients only train the backward parameters.
    Forward parameters then receive only terminal-energy gradients. This mode
    is not the full mean-work gradient and requires disjoint parameter sets.
    With mean_parameterization='native', subtract the reference mean dilation
    before bounding the residual. Without a bound the forward mean is exactly
    x+dt*v; with a bound it approximates that mean in the central region while
    the far-tail residual stays bounded. Noise and evaluated kernel densities
    are unchanged. This is an initialization choice, not a new work identity.
    A positive noise_annealing_power multiplies noise_scale by (1-t_left)^power
    at each step. Every finite-step variance stays positive. The Gaussian
    reference/reverse factors use that same actual scale.
    """
    _states(x0);grid=_schedule(times,'times')
    if (forward_precision is None)!=(backward_precision is None):raise ValueError('Supply both precision callbacks or neither')
    if mean_parameterization not in {'reference','native'}:raise ValueError('Unknown kernel mean parameterization')
    if not math.isfinite(noise_annealing_power) or noise_annealing_power<0:raise ValueError('Nonnegative finite noise annealing power required')
    if not math.isfinite(noise_scale) or noise_scale<=0 or not math.isfinite(prior_std) or prior_std<=0:
        raise ValueError('Positive finite noise and prior scales required')
    if terminal_std is not None and (not math.isfinite(terminal_std) or terminal_std<=0):raise ValueError('Invalid terminal reference scale')
    if max_drift_norm is not None and (not math.isfinite(max_drift_norm) or max_drift_norm<=0):raise ValueError('Invalid drift bound')
    log_initial=gaussian_log_density(x0,torch.zeros_like(x0),prior_std)
    x=x0;ratio=x0.new_zeros(len(x0),dtype=torch.float64)
    saved=[x.detach().clone()] if retain_states else None
    for left,right in zip(grid,grid[1:]):
        dt,af,ab,std_forward,std_backward=gaussian_reference_step(left,right,noise_scale,prior_std,terminal_std,noise_annealing_power)
        noise=torch.randn(x.shape,dtype=x.dtype,device=x.device,generator=generator)
        def step(state,epsilon,left=left,right=right,dt=dt,af=af,ab=ab,sf=std_forward,sb=std_backward):
            drift=forward_drift(state,left)
            if drift.shape!=state.shape or not torch.isfinite(drift).all():raise ValueError('Invalid forward drift')
            if mean_parameterization=='native':drift=drift+(1.-af)/dt*state
            if max_drift_norm is not None:
                drift=drift*max_drift_norm/torch.sqrt(max_drift_norm**2+drift.square().sum(-1,keepdim=True))
            mean=af*state+dt*drift
            forward_cholesky=None if forward_precision is None else precision_cholesky(forward_precision(state,left))
            terminal=mean+sf*epsilon if forward_cholesky is None else gaussian_precision_sample(mean,forward_cholesky,sf,epsilon)
            reverse_input=terminal.detach() if forward_energy_only else terminal
            reverse_state=state.detach() if forward_energy_only else state
            reverse_drift=backward_drift(reverse_input,right)
            if reverse_drift.shape!=terminal.shape or not torch.isfinite(reverse_drift).all():raise ValueError('Invalid backward drift')
            if mean_parameterization=='native':reverse_drift=reverse_drift+(1.-ab)/dt*reverse_input
            if max_drift_norm is not None:
                reverse_drift=reverse_drift*max_drift_norm/torch.sqrt(max_drift_norm**2+reverse_drift.square().sum(-1,keepdim=True))
            reverse_mean=ab*reverse_input+dt*reverse_drift
            log_forward=gaussian_log_density(terminal,mean,sf) if forward_cholesky is None else gaussian_precision_log_density(terminal,mean,forward_cholesky,sf)
            if forward_energy_only:log_forward=log_forward.detach()
            backward_cholesky=None if backward_precision is None else precision_cholesky(backward_precision(reverse_input,right))
            log_backward=gaussian_log_density(reverse_state,reverse_mean,sb) if backward_cholesky is None else gaussian_precision_log_density(reverse_state,reverse_mean,backward_cholesky,sb)
            change=log_forward-log_backward
            return terminal,change
        if checkpoint_steps:
            x,increment=checkpoint(step,x,noise,use_reentrant=False)
        else:x,increment=step(x,noise)
        ratio=ratio+increment
        if saved is not None:saved.append(x.detach().clone())
    _states(x)
    return TrainablePath(x,log_initial,ratio,None if saved is None else torch.stack(saved,dim=1))
