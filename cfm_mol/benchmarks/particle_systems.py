"""DW-4 and LJ-13 target energies on the centre-of-mass-free subspace.

These are the benchmarks the equivariant-flow literature reports on, so the
parameter values here are fixed by that literature and must not be tuned. Each
energy is written for a batch of centred configurations and returns a reduced
(dimensionless) energy, i.e. already divided by the reference temperature, which
is the convention those papers use.

The exact constants are pinned in `PARAMETERS` and are checked against the
published definitions in `tests/test_particle_systems.py`; do not edit them to
make a result look better.
"""
import torch


PARAMETERS = {
    'dw4': {'a': 0.0, 'b': -4.0, 'c': 0.9, 'd0': 4.0, 'tau': 1.0, 'n_particles': 4},
    'lj13': {'epsilon': 1.0, 'r_m': 1.0, 'tau': 1.0, 'oscillator': 0.5, 'n_particles': 13},
}


def _pairwise_distance(x):
    """Upper-triangular pair distances for a centred batch, shape (B, P*(P-1)/2)."""
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError('Require (batch, particles, 3) coordinates')
    n = x.shape[1]
    i, j = torch.triu_indices(n, n, offset=1, device=x.device)
    d = (x[:, i] - x[:, j]).square().sum(-1)
    return d.sqrt()


def dw4_energy(x, *, a=None, b=None, c=None, d0=None, tau=None):
    """Double-well reduced energy, Koehler-Klein-Noe (ICML 2020) form.

        u(x) = 1/tau sum_{i<j} [ a (d_ij - d0) + b (d_ij - d0)^2 + c (d_ij - d0)^4 ]

    The reference's ordered-pair sum has a factor 1/2; the unordered sum does not.
    """
    p = PARAMETERS['dw4']
    a = p['a'] if a is None else a
    b = p['b'] if b is None else b
    c = p['c'] if c is None else c
    d0 = p['d0'] if d0 is None else d0
    tau = p['tau'] if tau is None else tau
    delta = _pairwise_distance(x) - d0
    return (a * delta + b * delta.square() + c * delta.pow(4)).sum(-1) / tau


def lj13_energy(x, *, epsilon=None, r_m=None, tau=None, oscillator=None):
    """Lennard-Jones reduced energy with the harmonic centre-of-mass term.

        u(x) = epsilon/(2 tau) sum_{i,j} [ (r_m/d_ij)^12 - 2 (r_m/d_ij)^6 ]
               + oscillator * sum_i || x_i - mean(x) ||^2

    The pair sum runs over ORDERED pairs in the reference implementation, so the
    equivalent unordered form used here carries 1/tau rather than 1/(2 tau).
    Writing 1/(2 tau) together with an unordered sum would halve the energy.

    The oscillator term is what keeps the cluster bound; the literature includes
    it in the target, so it is part of the benchmark and not a modelling choice
    of ours. This target has no distance floor: coincident particles have
    positive infinite energy and zero target weight. Numerical failures must
    be retained rather than silently changing the target potential.
    """
    p = PARAMETERS['lj13']
    epsilon = p['epsilon'] if epsilon is None else epsilon
    r_m = p['r_m'] if r_m is None else r_m
    tau = p['tau'] if tau is None else tau
    oscillator = p['oscillator'] if oscillator is None else oscillator
    d = _pairwise_distance(x)
    ratio = (r_m / d).pow(6)
    pair = epsilon * ((ratio-1).square()-1)
    centred = x - x.mean(1, keepdim=True)
    # The harmonic term sits OUTSIDE the epsilon/(2 tau) prefactor in the reference
    # implementation: it is neither scaled by epsilon nor divided by tau.
    return pair.sum(-1) / tau + oscillator * centred.square().sum((1, 2))


ENERGIES = {'dw4': dw4_energy, 'lj13': lj13_energy}


def reduced_energy(name, x):
    if name not in ENERGIES:
        raise ValueError(f'Unknown benchmark {name!r}')
    return ENERGIES[name](x)
