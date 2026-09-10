import math

import numpy as np
import pytest
import torch

from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.parity_refinement import evaluate_even_potential, parity_work_change, randomize_inversion


def test_even_potential_force_and_physical_query_accounting():
    oracle = object.__new__(EnergyOracle)
    oracle.evaluated = 0
    bias = torch.tensor([[.2, -.7, .9], [-.1, .8, .3]], dtype=torch.float64)
    def evaluate(x):
        oracle.evaluated += len(x)
        return (.5*x.square()+bias*x).sum((1, 2)), -x-bias
    oracle.evaluate = evaluate
    x = torch.randn(37, 2, 3, dtype=torch.float64)
    energy, force, parts = evaluate_even_potential(oracle, x, max_request=16)
    torch.testing.assert_close(energy, .5*x.square().sum((1, 2)))
    torch.testing.assert_close(force, -x)
    torch.testing.assert_close(parts['odd_energy_eV'], (bias*x).sum((1, 2)))
    torch.testing.assert_close(parts['odd_force_eV_A'], -bias.expand_as(x))
    assert oracle.evaluated == 74
    h = 1e-4
    direction = torch.randn_like(x)
    plus, _, _ = evaluate_even_potential(oracle, x+h*direction)
    minus, _, _ = evaluate_even_potential(oracle, x-h*direction)
    torch.testing.assert_close((plus-minus)/(2*h), -(force*direction).sum((1, 2)), atol=1e-10, rtol=1e-10)


def test_inversion_lineage_and_dedicated_rng_replay():
    x = torch.ones(100, 2, 3, dtype=torch.float64)
    x[:, 1] = -x[:, 0]
    a, signs = randomize_inversion(x, generator=torch.Generator().manual_seed(9201))
    b, repeated = randomize_inversion(x, generator=torch.Generator().manual_seed(9201))
    assert set(signs.tolist()) == {-1, 1}
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(signs, repeated)
    torch.testing.assert_close(a*signs[:, None, None], x, atol=0, rtol=0)
    torch.testing.assert_close(a.mean(1), torch.zeros(100, 3, dtype=torch.float64))


def test_entropy_identity_for_noninvariant_source_after_symmetry_mixture():
    # The source is shifted Gaussian and is NOT inversion-invariant. The
    # symmetrized source is a known two-component mixture. Gauss-Hermite
    # integration verifies the change and the additional source-mixture gain.
    nodes, weights = np.polynomial.hermite.hermgauss(64)
    x = torch.tensor(1.2+math.sqrt(2)*.6*nodes, dtype=torch.float64)
    weights = torch.tensor(weights/math.sqrt(math.pi), dtype=torch.float64)
    def log_gaussian(v, mean, sigma):
        return -.5*((v-mean)/sigma)**2-math.log(sigma)-.5*math.log(2*math.pi)
    def log_mixture(v, scale=1.):
        return torch.logsumexp(torch.stack([log_gaussian(v, scale*1.2, scale*.6),
            log_gaussian(v, -scale*1.2, scale*.6)]), 0)-math.log(2)
    scale = .7
    original = log_gaussian(x, 1.2, .6)
    mixture = log_mixture(x)
    transported = log_mixture(scale*x, scale)
    torch.testing.assert_close(transported, mixture-math.log(scale), atol=1e-12, rtol=1e-12)
    direct_delta = transported+.5*(scale*x)**2-(mixture+.5*x**2)
    formula = .5*(scale**2-1)*x**2-math.log(scale)
    torch.testing.assert_close(direct_delta, formula, atol=1e-12, rtol=1e-12)
    expected = .5*(scale**2-1)*(1.2**2+.6**2)-math.log(scale)
    torch.testing.assert_close(weights@formula, torch.tensor(expected, dtype=torch.float64), atol=1e-12, rtol=1e-12)
    mixture_gain = float(weights@(original-mixture))
    assert 0 < mixture_gain < math.log(2)
    # Refiner's formula excludes this nonnegative source-symmetrization gain.
    total = weights@(transported+.5*(scale*x)**2-(original+.5*x**2))
    torch.testing.assert_close(total, weights@formula-mixture_gain, atol=1e-12, rtol=1e-12)


def test_uniform_auxiliary_sign_work_correction():
    # Choose either sign. The proposal and target auxiliary each have1/2,
    # so they cancel even for an asymmetric original forward/reverse law.
    raw_energy = torch.tensor([1., 3.], dtype=torch.float64)
    even_energy = torch.tensor([2., 2.5], dtype=torch.float64)
    log_q = torch.tensor([-.3, -1.7], dtype=torch.float64)
    log_r = torch.tensor([-1.1, -.2], dtype=torch.float64)
    kT = .4
    raw_work = raw_energy/kT+log_q-log_r
    actual = parity_work_change(raw_work, raw_energy, even_energy, kT=kT)
    joint = even_energy/kT+(log_q-math.log(2))-(log_r-math.log(2))
    torch.testing.assert_close(actual, joint)
    with pytest.raises(ValueError):
        parity_work_change(raw_work, raw_energy, even_energy, kT=-1.)
