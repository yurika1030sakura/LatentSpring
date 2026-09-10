"""Gaussian-noise coordinates and a normalized global auxiliary posterior.

This is a diagnostic implementation of standard Gaussian conditioning and
Gauss--Newton/Laplace approximation, not a new sampler identity. Reparameterizing
an existing path does not change its endpoint distribution. All auxiliary
parameters must depend only on the endpoint, never on its generating noise.
"""
from dataclasses import dataclass
import math

import torch

from cfm_mol.nonequilibrium import _schedule, _states, gaussian_log_density
from cfm_mol.path_work import gaussian_reference_step


class GaussianInnovationMap:
    """Original scalar Gaussian path as y=F(z)+s*epsilon, z~N(0,I).

    z contains initial Gaussian coordinates and all but the last innovation.
    The map returns the last conditional mean. Its density is never treated as
    a bijective-flow likelihood. The transformation from z to intermediate
    path states is triangular with a known, constant Jacobian determinant.
    """
    def __init__(self, drift, dimension, times, noise_scale, *, prior_std=1.,
                 terminal_std=None, max_drift_norm=None,
                 mean_parameterization='reference', noise_annealing_power=0.):
        if not isinstance(dimension, int) or dimension < 1:
            raise ValueError('Positive integer dimension required')
        if any(not math.isfinite(v) or v <= 0 for v in [noise_scale, prior_std]):
            raise ValueError('Positive finite scales required')
        if terminal_std is not None and (not math.isfinite(terminal_std) or terminal_std <= 0):
            raise ValueError('Invalid terminal scale')
        if max_drift_norm is not None and (not math.isfinite(max_drift_norm) or max_drift_norm <= 0):
            raise ValueError('Invalid drift bound')
        if mean_parameterization not in ['reference', 'native']:
            raise ValueError('Unknown mean convention')
        if not math.isfinite(noise_annealing_power) or noise_annealing_power < 0:
            raise ValueError('Invalid noise schedule')
        self.drift = drift
        self.dimension = dimension
        self.grid = _schedule(times, 'times')
        self.steps = len(self.grid)-1
        self.latent_dimension = self.steps*dimension
        self.prior_std = prior_std
        self.bound = max_drift_norm
        self.native = mean_parameterization == 'native'
        self.coefficients = [gaussian_reference_step(a, b, noise_scale, prior_std,
            terminal_std, noise_annealing_power) for a, b in zip(self.grid, self.grid[1:])]
        self.terminal_noise_std = self.coefficients[-1][3]
        self.log_state_jacobian = dimension*(math.log(prior_std)+sum(math.log(c[3]) for c in self.coefficients[:-1]))
        self.map_calls = 0

    def mean(self, x, step):
        dt, af, _, _, _ = self.coefficients[step]
        v = self.drift(x, self.grid[step])
        if v.shape != x.shape or not torch.isfinite(v).all():
            raise ValueError('Invalid independent drift')
        if self.native:
            v = v+(1-af)/dt*x
        if self.bound is not None:
            v = v*self.bound/torch.sqrt(self.bound**2+v.square().sum(-1, keepdim=True))
        return af*x+dt*v

    def __call__(self, z, *, retain_states=False):
        _states(z)
        if z.shape[1] != self.latent_dimension:
            raise ValueError('Innovation dimension differs from path')
        self.map_calls += 1
        innovations = z.reshape(len(z), self.steps, self.dimension)
        x = innovations[:, 0]*self.prior_std
        states = [x] if retain_states else None
        for k in range(self.steps):
            mean = self.mean(x, k)
            if k < self.steps-1:
                x = mean+self.coefficients[k][3]*innovations[:, k+1]
                if retain_states:
                    states.append(x)
        return (mean, torch.stack(states, dim=1)) if retain_states else mean

    def from_states(self, states):
        if states.ndim != 3 or states.shape[1:] != (self.steps+1, self.dimension):
            raise ValueError('Complete original paths required')
        _states(states.flatten(0, 1))
        innovations = [states[:, 0]/self.prior_std]
        for k in range(self.steps-1):
            innovations.append((states[:, k+1]-self.mean(states[:, k], k))/self.coefficients[k][3])
        return torch.stack(innovations, dim=1).flatten(1)


def independent_map_jacobian(function, z):
    """Jacobian [batch,output,input]; caller guarantees independent map rows."""
    with torch.enable_grad():
        z = z.detach().double().requires_grad_(True)
        value = function(z)
        _states(value)
        if len(value) != len(z):
            raise ValueError('Map changed particle count')
        rows = [torch.autograd.grad(value[:, j].sum(), z, retain_graph=j < value.shape[1]-1)[0]
                for j in range(value.shape[1])]
    jacobian = torch.stack(rows, dim=1).detach()
    if not torch.isfinite(jacobian).all():
        raise FloatingPointError('Non-finite map Jacobian')
    return value.detach(), jacobian


@dataclass
class GaussianAuxiliary:
    """L(z|y)=N(mean,(I+J^T J/s^2)^-1), fully normalized in latent space."""
    mean: torch.Tensor
    jacobian: torch.Tensor
    noise_std: float

    def log_prob(self, z, *, metric_scale=1.):
        if z.shape != self.mean.shape or self.jacobian.ndim != 3 or self.jacobian.shape[::2] != z.shape:
            raise ValueError('Auxiliary shapes disagree')
        if not math.isfinite(self.noise_std) or self.noise_std <= 0 or not math.isfinite(metric_scale) or metric_scale < 0:
            raise ValueError('Invalid auxiliary metric')
        if not all(torch.isfinite(v).all() for v in [z, self.mean, self.jacobian]):
            raise ValueError('Finite auxiliary parameters required')
        batch, out, _ = self.jacobian.shape
        scaled = self.jacobian*(math.sqrt(metric_scale)/self.noise_std)
        small = torch.eye(out, dtype=z.dtype, device=z.device).expand(batch, out, out)+scaled@scaled.transpose(-1, -2)
        factor = torch.linalg.cholesky(small)
        logdet = 2*factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        delta = z-self.mean
        projected = (scaled@delta[..., None]).squeeze(-1)
        return -.5*z.shape[1]*math.log(2*math.pi)+.5*logdet-.5*delta.square().sum(-1)-.5*projected.square().sum(-1)


def fit_gauss_newton_auxiliary(function, endpoint, latent_dimension, noise_std, *, iterations=4, backtracks=8, initial_mean=None):
    """Deterministic approximate posterior mode and Gauss--Newton metric.

    The optional initial_mean must be a deterministic function of the endpoint,
    never the observed generating latent. With no initial_mean it starts at zero.
    Finite iterations, rejected moves and local curvature are all approximation
    limitations. The returned Gaussian is normalized regardless of convergence.
    No claim that it equals the posterior or has finite importance variance.
    """
    _states(endpoint)
    if min(latent_dimension, iterations, backtracks) < 1 or not math.isfinite(noise_std) or noise_std <= 0:
        raise ValueError('Positive dimensions, counts and noise required')
    endpoint = endpoint.detach().double()
    mean = endpoint.new_zeros(len(endpoint), latent_dimension)
    if initial_mean is not None:
        if initial_mean.shape != mean.shape or not torch.isfinite(initial_mean).all():
            raise ValueError('Invalid endpoint-derived initial mean')
        mean = initial_mean.detach().to(mean).clone()
    sigma2 = noise_std**2
    history = []
    identity = torch.eye(endpoint.shape[1], dtype=endpoint.dtype, device=endpoint.device)
    def objective(z, value):
        return .5*z.square().sum(-1)+.5*(value-endpoint).square().sum(-1)/sigma2
    for iteration in range(iterations):
        value, jacobian = independent_map_jacobian(function, mean)
        current = objective(mean, value)
        small = jacobian@jacobian.transpose(-1, -2)+sigma2*identity
        rhs = endpoint-value+(jacobian@mean[..., None]).squeeze(-1)
        local_mean = (jacobian.transpose(-1, -2)@torch.linalg.solve(small, rhs[..., None])).squeeze(-1)
        direction = local_mean-mean
        accepted = torch.zeros(len(mean), dtype=torch.bool, device=mean.device)
        proposed = mean.clone()
        next_objective = current.clone()
        with torch.no_grad():
            for k in range(backtracks):
                candidate = mean+direction*(.5**k)
                candidate_value = function(candidate)
                candidate_objective = objective(candidate, candidate_value)
                good = (~accepted) & torch.isfinite(candidate_objective) & (candidate_objective <= current)
                proposed[good] = candidate[good]
                next_objective[good] = candidate_objective[good]
                accepted |= good
                if accepted.all():
                    break
        mean = proposed
        history.append({'iteration': iteration+1, 'accepted': accepted.cpu().tolist(),
            'objective_before': current.cpu().tolist(), 'objective_after': next_objective.cpu().tolist()})
    value, jacobian = independent_map_jacobian(function, mean)
    gradient = mean+(jacobian.transpose(-1, -2)@((value-endpoint)/sigma2)[..., None]).squeeze(-1)
    diagnostics = {'history': history, 'final_objective': objective(mean, value).cpu().tolist(),
        'endpoint_residual_norm': (value-endpoint).norm(dim=-1).cpu().tolist(),
        'posterior_gradient_norm': gradient.norm(dim=-1).cpu().tolist(),
        'initialization': 'zero' if initial_mean is None else 'caller supplied endpoint-derived mean'}
    return GaussianAuxiliary(mean.detach(), jacobian, noise_std), diagnostics


def innovation_log_joint(z, endpoint, conditional_mean, noise_std):
    """Normalized Q(z,y), where y|z has the actual last-step Gaussian noise."""
    _states(z)
    if endpoint.shape != conditional_mean.shape or len(endpoint) != len(z):
        raise ValueError('Invalid joint Gaussian-map shapes')
    return gaussian_log_density(z, torch.zeros_like(z), 1.)+gaussian_log_density(endpoint, conditional_mean, noise_std)
