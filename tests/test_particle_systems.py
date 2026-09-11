"""Pin the DW-4 and LJ-13 constants to the published reference implementations.

Both were got wrong once while building this benchmark: the LJ-13 harmonic
coefficient (1.0 instead of 0.5) and then its placement (inside the
epsilon/(2 tau) prefactor instead of outside). These are silent errors -- they
produce a plausible number against the wrong target density -- so the constants
and the pair convention are asserted here rather than trusted to a comment.

Reference: E-ACF (Midgley et al., NeurIPS 2023), eacf/targets/target_energy/.
"""
import torch

from cfm_mol.benchmarks.particle_systems import PARAMETERS, dw4_energy, lj13_energy


def _centred(batch, n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, n, 3, generator=g, dtype=torch.float64)
    return x - x.mean(1, keepdim=True)


def test_dw4_constants():
    p = PARAMETERS['dw4']
    assert (p['a'], p['b'], p['c'], p['d0'], p['tau']) == (0.0, -4.0, 0.9, 4.0, 1.0)
    assert p['n_particles'] == 4


def test_lj13_constants():
    p = PARAMETERS['lj13']
    assert (p['epsilon'], p['r_m'], p['tau']) == (1.0, 1.0, 1.0)
    assert p['oscillator'] == 0.5, 'E-ACF uses 0.5, not 1.0'
    assert p['n_particles'] == 13


def test_dw4_matches_ordered_pair_reference():
    """Reference sums ORDERED pairs with 1/(2 tau); ours sums unordered with 1/tau."""
    x = _centred(4, 4, 1)
    a, b, c, d0, tau = 0.0, -4.0, 0.9, 4.0, 1.0
    d = (x[:, :, None] - x[:, None, :]).square().sum(-1).clamp_min(1e-24).sqrt()
    off = ~torch.eye(4, dtype=torch.bool)
    delta = d[:, off].reshape(len(x), -1) - d0            # all 12 ordered pairs
    reference = (a*delta + b*delta.square() + c*delta.pow(4)).sum(-1)/tau/2
    torch.testing.assert_close(dw4_energy(x), reference, atol=1e-10, rtol=1e-10)


def test_lj13_harmonic_term_sits_outside_the_prefactor():
    """Scaling tau must not rescale the harmonic term."""
    x = _centred(3, 13, 2)
    base = lj13_energy(x)
    doubled = lj13_energy(x, tau=2.0)
    centred = x - x.mean(1, keepdim=True)
    harmonic = 0.5*centred.square().sum((1, 2))
    # pair part halves, harmonic part does not
    torch.testing.assert_close(doubled, (base - harmonic)/2 + harmonic, atol=1e-8, rtol=1e-8)


def test_lj13_matches_ordered_pair_reference():
    x = _centred(3, 13, 3)
    d = (x[:, :, None] - x[:, None, :]).square().sum(-1).clamp_min(1e-24).sqrt()
    off = ~torch.eye(13, dtype=torch.bool)
    d = d[:, off].reshape(len(x), -1)                      # 156 ordered pairs
    ratio = (1.0/d.clamp_min(1e-4)).pow(6)
    pair = (ratio.square() - 2*ratio).sum(-1)/2            # epsilon/(2 tau), ordered
    centred = x - x.mean(1, keepdim=True)
    reference = pair + 0.5*centred.square().sum((1, 2))
    torch.testing.assert_close(lj13_energy(x), reference, atol=1e-6, rtol=1e-9)


def test_lj_repulsive_core_is_not_silently_capped():
    x = torch.zeros(1, 13, 3, dtype=torch.float64)
    x[0, :, 0] = torch.arange(13, dtype=torch.float64)
    x[0, 1, 0] = 5e-5
    energy = lj13_energy(x)
    assert float(energy[0]) > 1e50
    x[0, 1] = x[0, 0]
    assert torch.isposinf(lj13_energy(x)).all()


def test_zero_target_weights_have_defined_finite_sample_ess():
    from cfm_mol.benchmarks.harness import effective_sample_size
    import pytest
    assert effective_sample_size(torch.tensor([0., -torch.inf])) == .5
    with pytest.raises(ValueError):
        effective_sample_size(torch.tensor([-torch.inf, -torch.inf]))
