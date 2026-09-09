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

    def work(self,reduced_terminal_energy):
        if reduced_terminal_energy.shape!=self.log_initial.shape:
            raise ValueError('One terminal energy required per path')
        value=reduced_terminal_energy.double()+self.log_initial+self.log_forward_minus_backward
        if not torch.isfinite(value).all():raise FloatingPointError('Non-finite generalized work')
        return value


def gaussian_training_path(x0,forward_drift,backward_drift,times,noise_scale,generator,*,
                           prior_std=1.,checkpoint_steps=False):
    """Simulate reparameterized paths and retain complete first-order gradients.

    x0 must be an independent draw from the stated isotropic Gaussian.
    Fixed standard deviation noise_scale*sqrt(dt) is used in each direction.
    Backward drift is an auxiliary normalized kernel, not physical time reversal.
    Noise is generated once and reused exactly during checkpoint recomputation.
    """
    _states(x0);grid=_schedule(times,'times')
    if not math.isfinite(noise_scale) or noise_scale<=0 or not math.isfinite(prior_std) or prior_std<=0:
        raise ValueError('Positive finite noise and prior scales required')
    log_initial=gaussian_log_density(x0,torch.zeros_like(x0),prior_std)
    x=x0;ratio=x0.new_zeros(len(x0),dtype=torch.float64)
    for left,right in zip(grid,grid[1:]):
        dt=right-left;std=noise_scale*math.sqrt(dt)
        noise=torch.randn(x.shape,dtype=x.dtype,device=x.device,generator=generator)
        def step(state,epsilon,left=left,right=right,dt=dt,std=std):
            drift=forward_drift(state,left)
            if drift.shape!=state.shape or not torch.isfinite(drift).all():raise ValueError('Invalid forward drift')
            mean=state+dt*drift
            terminal=mean+std*epsilon
            reverse_drift=backward_drift(terminal,right)
            if reverse_drift.shape!=terminal.shape or not torch.isfinite(reverse_drift).all():raise ValueError('Invalid backward drift')
            reverse_mean=terminal+dt*reverse_drift
            change=gaussian_log_density(terminal,mean,std)-gaussian_log_density(state,reverse_mean,std)
            return terminal,change
        if checkpoint_steps:
            x,increment=checkpoint(step,x,noise,use_reentrant=False)
        else:x,increment=step(x,noise)
        ratio=ratio+increment
    _states(x)
    return TrainablePath(x,log_initial,ratio)
