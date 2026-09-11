"""Frozen source + exact-entropy adapter on the standard particle benchmarks.

The adapter construction needs a frozen source whose density it never has to
evaluate during training. For a benchmark we nonetheless want the source density
in closed form so that importance weights require no density approximation.
ESS is still a finite-sample diagnostic, not known population coverage.
The source here is a centred isotropic Gaussian on the
centre-of-mass-free subspace. Because the adapter supplies an exact log-volume,
the pushforward density is exact too:

    log q_T(T(x)) = log q_0(x) - logdet DT(x)

Energy expectations and ESS below retain Monte Carlo uncertainty.
"""
import math

import torch

from cfm_mol.benchmarks.particle_systems import reduced_energy


class CentredGaussianSource:
    """Isotropic Gaussian restricted to the COM-free subspace of R^(P x 3)."""

    def __init__(self, n_particles, scale, *, dtype=torch.float64, device='cpu'):
        if n_particles < 2 or not math.isfinite(scale) or scale <= 0:
            raise ValueError('Require at least two particles and a positive scale')
        self.n_particles = int(n_particles)
        self.scale = float(scale)
        self.dtype, self.device = dtype, device
        self.dimension = 3*(self.n_particles-1)

    def sample(self, n, generator=None):
        z = torch.randn(n, self.n_particles, 3, generator=generator, dtype=self.dtype)
        z = z - z.mean(1, keepdim=True)
        return (self.scale*z).to(self.device)

    def log_density(self, x):
        """Exact log density on the subspace, in its own orthonormal coordinates."""
        if float(x.mean(1).abs().max()) > 1e-7:
            raise ValueError('Source density is defined on centred configurations')
        quadratic = x.square().sum((1, 2))/(2*self.scale**2)
        constant = .5*self.dimension*math.log(2*math.pi*self.scale**2)
        return -quadratic-constant


def effective_sample_size(log_weights):
    """Normalised ESS in [0, 1]: (sum w)^2 / (N sum w^2), computed stably."""
    if (log_weights.ndim != 1 or not len(log_weights) or torch.isnan(log_weights).any()
            or torch.isposinf(log_weights).any() or not torch.isfinite(log_weights).any()):
        raise ValueError('Require defined log weights with at least one finite weight')
    n = log_weights.numel()
    shifted = log_weights-log_weights.max()
    w = shifted.exp()
    return float(w.sum().square()/(n*w.square().sum()))


def reverse_diagnostics(name, source, adapter, n, *, generator=None, device='cpu', samples_out=None):
    """Reverse (model-sample) weights for the base and for the adapted model.

    Weights are pi(y)/q(y) up to the target's unknown normaliser, which cancels
    in ESS. The base row is the same quantity with the identity map, so the two
    rows are directly comparable and the base is not a different estimator.

    Reverse ESS cannot see mode collapse, and the training objective here is
    mode-seeking reverse KL, so weight-concentration diagnostics are reported
    alongside it and must not be dropped when quoting the ESS.
    """
    with torch.no_grad():
        x = source.sample(n, generator=generator).to(device)
        log_q0 = source.log_density(x)
        base_energy = reduced_energy(name, x)
        base_lw = -base_energy-log_q0
        chunks = [adapter(part) for part in x.split(128)]
        y = torch.cat([part[0] for part in chunks]); volume = torch.cat([part[1] for part in chunks])
        log_qT = log_q0-volume
        adapted_energy = reduced_energy(name, y)
        adapted_lw = -adapted_energy-log_qT
        if samples_out is not None:
            torch.save({key:value.cpu() for key,value in dict(parent_positions=x, positions=y,
                base_energy=base_energy, adapted_energy=adapted_energy, log_q0=log_q0,
                log_qT=log_qT, log_volume=volume).items()}, samples_out)
    # Forward ESS needs target samples, which we do not have for these systems
    # without an independent sampler; what we CAN report cheaply is the weight
    # dispersion and the fraction of mass in the largest weight, both of which
    # expose the concentration that reverse ESS alone can hide.
    def _concentration(lw):
        w = (lw-lw.max()).exp(); w = w/w.sum()
        top = w.sort(descending=True).values
        return {'max_weight': float(top[0]),
                'mass_in_top_1pct': float(top[:max(1, len(top)//100)].sum()),
                'log_weight_range': float(lw.max()-lw.min())}

    return {
        'base': {'ess': effective_sample_size(base_lw),
                 'concentration': _concentration(base_lw),
                 'mean_reduced_energy': float(base_energy.mean()),
                 'log_weight_std': float(base_lw.std())},
        'adapted': {'ess': effective_sample_size(adapted_lw),
                    'concentration': _concentration(adapted_lw),
                    'mean_reduced_energy': float(adapted_energy.mean()),
                    'log_weight_std': float(adapted_lw.std()),
                    'mean_log_volume': float(volume.mean())},
        'n_samples': int(n),
        'target_energy_evaluations': 2*int(n),
        'weight_scope': 'exact density ratios on sampled labelled configurations; ESS is a finite-sample diagnostic',
    }


def reverse_kl_objective(name, source, adapter, x):
    """E_q0[ u(T x) - logdet DT(x) ]: reverse KL to the target up to a constant.

    This is the same objective the molecular runs use, with the reduced energy in
    place of (U - U_0)/kT. A population reverse-KL interpretation additionally
    requires integrability; finite minibatches do not establish it for singular LJ.
    """
    y, volume = adapter(x)
    return (reduced_energy(name, y)-volume).mean(), y, volume
