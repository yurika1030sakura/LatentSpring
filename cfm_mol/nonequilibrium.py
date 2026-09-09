"""Finite-step work weights and an AIS teacher for conditional flow matching.

Each density/drift callback must act independently on each row; a callback
coupling particles would require joint-cloud weights instead of these weights.
These are implementations of existing path importance sampling / AIS identities,
not new estimators. States are [particles, Euclidean coordinates]. Molecular
callers must use an orthonormal COM-free coordinate system and fix composition,
charge, spin and a normalizable target. Projecting full-dimensional Gaussian
noise and then using its ambient density is invalid.

All log weights are accumulated in float64. The supplied log densities must be
actual densities of the implemented proposals, not noisy CNF log estimates.
The returned particles need their weights; they are not exact iid target draws.
"""
from dataclasses import dataclass
import math
from typing import Callable

import torch
from torch import Tensor

LogDensity = Callable[[Tensor], Tensor]
Drift = Callable[[Tensor, float], Tensor]


def centered_orthonormal_basis(n_atoms: int, *, dtype=torch.float64, device=None) -> Tensor:
    """Helmert basis [N,N-1]; Cartesian tensor product uses Lebesgue measure on H."""
    if n_atoms < 2:
        raise ValueError('At least two atoms are required for a nonempty positional state')
    basis = torch.zeros((n_atoms, n_atoms-1), dtype=dtype, device=device)
    for j in range(n_atoms-1):
        denominator = math.sqrt((j+1)*(j+2))
        basis[:j+1,j] = 1/denominator
        basis[j+1,j] = -(j+1)/denominator
    return basis


def _vector(value: Tensor, n: int, name: str) -> Tensor:
    if value.shape != (n,) or not torch.isfinite(value).all():
        raise ValueError(f'{name} must contain {n} finite log densities')
    return value.to(torch.float64)


def _states(x: Tensor) -> None:
    if x.ndim != 2 or min(x.shape) < 1 or not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError('States must be a finite nonempty floating [particles, dimensions] tensor')


def _schedule(values, name: str) -> list[float]:
    result = [float(v) for v in values]
    if (len(result) < 2 or result[0] != 0 or result[-1] != 1
            or not all(math.isfinite(v) for v in result)
            or any(b <= a for a, b in zip(result, result[1:]))):
        raise ValueError(f'{name} must increase strictly from 0 to 1')
    return result


def normalized_weights(log_weights: Tensor) -> Tensor:
    """Normalize one fixed-condition cloud; never normalize across compositions."""
    if (log_weights.ndim != 1 or not log_weights.numel()
            or torch.isnan(log_weights).any() or torch.isposinf(log_weights).any()
            or not torch.isfinite(log_weights).any()):
        raise ValueError('Weights need at least one finite value and no NaN or +inf')
    return torch.softmax(log_weights.to(torch.float64), dim=0)


@dataclass
class WeightedPaths:
    positions: Tensor
    log_weights: Tensor
    diagnostics: dict

    @property
    def dimensionless_work(self) -> Tensor:
        return -self.log_weights

    def summary(self) -> dict:
        weights = normalized_weights(self.log_weights)
        n = weights.numel()
        return {
            'particles': n,
            'ess': float(1 / weights.square().sum()),
            'ess_fraction': float(1 / (n * weights.square().sum())),
            'maximum_weight': float(weights.max()),
            # log of an unbiased Z estimator is not itself unbiased.
            'log_mean_weight': float(torch.logsumexp(self.log_weights.double(), 0) - math.log(n)),
            **self.diagnostics,
        }


def path_log_weight(log_initial: Tensor, log_target: Tensor,
                    log_forward: Tensor, log_backward: Tensor) -> Tensor:
    """log gamma(x_K) - log q0(x_0) + sum_k log L_k - log K_k.

    K_k(y|x) and L_k(x|y) must each be normalized, with target path support
    covered by the forward path. Transition arrays have shape [steps, batch].
    E_forward[exp(log_weight) f(x_K)] = integral gamma(x) f(x) dx.
    """
    n = log_initial.numel()
    initial = _vector(log_initial, n, 'log_initial')
    target = _vector(log_target, n, 'log_target')
    if (log_forward.ndim != 2 or log_forward.shape != log_backward.shape
            or log_forward.shape[1] != n or not torch.isfinite(log_forward).all()
            or not torch.isfinite(log_backward).all()):
        raise ValueError('Forward/backward log transitions must be finite [steps, particles] arrays')
    return target - initial + (log_backward.double() - log_forward.double()).sum(0)


def gaussian_log_density(value: Tensor, mean: Tensor, std: float) -> Tensor:
    if not math.isfinite(std) or std <= 0:
        raise ValueError('Gaussian standard deviation must be finite and positive')
    if value.shape != mean.shape or value.ndim != 2:
        raise ValueError('Gaussian value and mean must have equal [particles, dimensions] shapes')
    residual = (value.double() - mean.double()) / std
    return -0.5 * residual.square().sum(-1) - value.shape[-1] * math.log(std * math.sqrt(2 * math.pi))


@torch.no_grad()
def gaussian_path_sample(x0: Tensor, log_initial: LogDensity, log_target: LogDensity,
                         forward_drift: Drift, backward_drift: Drift, times,
                         noise_scale: float, generator: torch.Generator) -> WeightedPaths:
    """Euler Gaussian path with its exact *discrete* proposal density ratio.

    Forward mean: x + dt*b_forward(x,t); backward mean:
    y + dt*b_backward(y,t+dt). Both use std=noise_scale*sqrt(dt).
    Backward drift is an explicit normalized auxiliary kernel, not a claim
    that a learned drift is the physical time reverse. No trace is evaluated.
    Arbitrary drifts remain valid proposals, but may give unusably small ESS.
    """
    _states(x0)
    grid = _schedule(times, 'times')
    if not math.isfinite(noise_scale) or noise_scale <= 0:
        raise ValueError('noise_scale must be finite and positive')
    n = x0.shape[0]
    initial = _vector(log_initial(x0), n, 'log_initial')
    x = x0.clone()
    ratio = torch.zeros(n, dtype=torch.float64, device=x.device)
    for t0, t1 in zip(grid, grid[1:]):
        dt = t1 - t0
        std = noise_scale * math.sqrt(dt)
        b = forward_drift(x, t0)
        if b.shape != x.shape or not torch.isfinite(b).all():
            raise ValueError('Invalid forward drift')
        mean_forward = x + dt * b
        y = mean_forward + std * torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        _states(y)
        back = backward_drift(y, t1)
        if back.shape != y.shape or not torch.isfinite(back).all():
            raise ValueError('Invalid backward drift')
        mean_backward = y + dt * back
        ratio += gaussian_log_density(x, mean_backward, std) - gaussian_log_density(y, mean_forward, std)
        x = y
    logw = _vector(log_target(x), n, 'log_target') - initial + ratio
    _vector(logw, n, 'log_weights')
    return WeightedPaths(x, logw, {'method': 'finite_gaussian_path', 'steps': len(grid)-1,
                                  'drift_evaluations_per_particle': 2*(len(grid)-1),
                                  'target_evaluations_per_particle': 1})


@torch.no_grad()
def random_walk_ais(x0: Tensor, log_initial: LogDensity, log_target: LogDensity,
                    betas, proposal_std: float, generator: torch.Generator,
                    moves_per_stage: int = 1) -> WeightedPaths:
    """Value-only AIS; update work before an MH move invariant to each bridge.

    gamma_beta = q0**(1-beta) * gamma_target**beta. This implementation requires
    finite endpoint log densities on every queried point (strictly positive
    targets). q0 is normalized and x0 must actually be iid draws from q0.
    Exact invariant MH kernels do not require equilibration at each stage.
    No resampling is performed; terminal samples keep their AIS weights.
    """
    _states(x0)
    grid = _schedule(betas, 'betas')
    if not math.isfinite(proposal_std) or proposal_std <= 0 or not isinstance(moves_per_stage, int) or moves_per_stage < 1:
        raise ValueError('proposal_std must be positive; moves_per_stage must be a positive integer')
    x = x0.clone()
    n = len(x)
    l0 = _vector(log_initial(x), n, 'log_initial')
    l1 = _vector(log_target(x), n, 'log_target')
    logw = torch.zeros_like(l0)
    accepted = 0
    for prev_beta, beta in zip(grid, grid[1:]):
        logw += (beta - prev_beta) * (l1 - l0)
        for _ in range(moves_per_stage):
            y = x + proposal_std * torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            p0 = _vector(log_initial(y), n, 'proposal log_initial')
            p1 = _vector(log_target(y), n, 'proposal log_target')
            log_accept = (1-beta)*(p0-l0) + beta*(p1-l1)
            uniform = torch.rand(n, device=x.device, dtype=torch.float64, generator=generator)
            accept = torch.log(uniform) < torch.minimum(log_accept, torch.zeros_like(log_accept))
            x = torch.where(accept[:, None], y, x)
            l0 = torch.where(accept, p0, l0)
            l1 = torch.where(accept, p1, l1)
            accepted += int(accept.sum())
    proposals = (len(grid)-1)*moves_per_stage
    _vector(logw, n, 'log_weights')
    return WeightedPaths(x, logw, {'method': 'random_walk_ais', 'steps': len(grid)-1,
                                  'moves_per_stage': moves_per_stage,
                                  'acceptance_fraction': accepted/(n*proposals),
                                  'target_evaluations_per_particle': 1+proposals})


def weighted_flow_matching_loss(predicted_velocity: Tensor, target_velocity: Tensor,
                                 log_weights: Tensor) -> Tensor:
    """Distill one fixed-condition weighted teacher cloud with independent CFM pairs.

    The caller pairs each weighted endpoint with an independent Gaussian start
    and time, using (x1-x0) as the linear CFM target. Teacher weights are detached.
    Self-normalization has finite-particle bias; this does not preserve exact
    target sampling after distillation. Do not pool different conditions here.
    """
    if (predicted_velocity.shape != target_velocity.shape or predicted_velocity.ndim != 2
            or log_weights.shape != (len(predicted_velocity),)):
        raise ValueError('Expected velocities [particles, dimensions] and one log weight per particle')
    if not torch.isfinite(predicted_velocity).all() or not torch.isfinite(target_velocity).all():
        raise ValueError('Non-finite flow matching velocity')
    weights = normalized_weights(log_weights.detach())
    errors = (predicted_velocity - target_velocity).square().mean(-1)
    return (weights * errors.double()).sum()
