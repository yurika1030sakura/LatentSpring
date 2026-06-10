"""Unit tests for cfm_mol.bgfm_loss.

Tests:
  1. divergence_exact_atomwise agrees with finite-difference on small toy.
  2. divergence_hutchinson converges to exact as n_samples increases.
  3. score_from_fm_velocity matches analytical score on 1D Gaussian.
  4. force_loss and energy_loss_variance produce sensible values.
  5. bgfm_total_loss schedule gives expected weights.

Run:
    cd /n/holylabs/ryl_lab/Lab/yulili_cfm_mol
    source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
    conda activate envs/flowmol
    python -m pytest tests/test_bgfm_loss.py -v
"""
from __future__ import annotations

import math

import pytest
import torch

from cfm_mol.bgfm_loss import (
    divergence_exact_atomwise,
    divergence_hutchinson,
    score_from_fm_velocity,
    force_loss,
    force_direction_loss,
    score_force_cosine,
    energy_loss_variance,
    bgfm_total_loss,
)


# ---------------------------------------------------------------------------
# Divergence tests
# ---------------------------------------------------------------------------

def test_divergence_exact_linear_vf():
    """For v(x) = A x with constant matrix A, div(v) = trace(A).

    Use a 2-atom system (6-dim) with known A.
    """
    torch.manual_seed(0)
    # A is 6x6; set trace to a known value
    A = torch.randn(6, 6)
    diag_target = torch.tensor([0.3, -0.1, 0.5, -0.2, 0.4, 0.1])
    A.diagonal().copy_(diag_target)
    expected_trace = diag_target.sum().item()  # ~1.0

    def v_fn(x: torch.Tensor) -> torch.Tensor:
        # x: (2, 3); flatten, multiply, reshape
        return (A @ x.flatten()).reshape(2, 3)

    x = torch.randn(2, 3, requires_grad=True)
    n_atoms = torch.tensor([2], dtype=torch.long)
    div = divergence_exact_atomwise(v_fn, x, n_atoms)
    # (1,) tensor
    assert div.shape == (1,)
    assert abs(div.item() - expected_trace) < 1e-5, \
        f"div={div.item()} vs expected_trace={expected_trace}"


def test_divergence_exact_matches_finite_diff():
    """divergence_exact_atomwise == sum_i sum_c (v(x + eps e_ic) - v(x-eps e_ic))[i,c] / (2eps)
    on a nonlinear v_fn.
    """
    torch.manual_seed(1)

    # Nonlinear v: v(x) = x^3 element-wise + pairwise coupling
    def v_fn(x: torch.Tensor) -> torch.Tensor:
        # coupling: each v_i = x_i^3 + 0.1 * sum_j x_j
        return x ** 3 + 0.1 * x.sum(dim=0, keepdim=True).expand_as(x)

    x = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    n_atoms = torch.tensor([3], dtype=torch.long)
    div_autograd = divergence_exact_atomwise(v_fn, x, n_atoms).item()

    # Finite-difference reference
    eps = 1e-4
    div_fd = 0.0
    for i in range(3):
        for c in range(3):
            xp = x.detach().clone()
            xm = x.detach().clone()
            xp[i, c] += eps
            xm[i, c] -= eps
            div_fd += (v_fn(xp)[i, c] - v_fn(xm)[i, c]).item() / (2 * eps)

    assert abs(div_autograd - div_fd) < 1e-3, \
        f"autograd={div_autograd} vs fd={div_fd}"


def test_divergence_hutchinson_converges_to_exact():
    """Hutchinson with enough samples converges to exact trace."""
    torch.manual_seed(2)

    def v_fn(x):
        return x ** 3 + 0.1 * x.sum(dim=0, keepdim=True).expand_as(x)

    x = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    n_atoms = torch.tensor([3], dtype=torch.long)
    div_exact = divergence_exact_atomwise(v_fn, x, n_atoms).item()

    # Average 1000 Hutchinson samples -- should converge to within ~0.3
    torch.manual_seed(42)
    divs = []
    for _ in range(50):
        d = divergence_hutchinson(
            v_fn,
            x.detach().clone().requires_grad_(True),
            n_atoms,
            n_samples=20,
            rademacher=True,
        ).item()
        divs.append(d)
    mean_hutch = sum(divs) / len(divs)
    assert abs(mean_hutch - div_exact) < 0.5, \
        f"hutch={mean_hutch}, exact={div_exact}"


# ---------------------------------------------------------------------------
# Score derivation tests
# ---------------------------------------------------------------------------

def test_score_from_fm_velocity_one_d_gaussian():
    """For a 1D Gaussian target p_1 = N(mu, sigma^2), the flow matching
    conditional velocity u*(x, t) = (x_1 - x_0) has known marginal form.

    Let's use a toy where p_0 = N(0, 1), p_1 = delta(x_1 = mu), so
    x_t = (1-t) x_0 + t mu, and the true score is:
        s*(x_t, t) = -(x_t - t mu) / (1 - t)^2

    We should recover this from the FM velocity
        v*(x_t, t) = mu - x_0 = mu - (x_t - t mu) / (1 - t) = (mu - x_t) / (1-t)
    via score_from_fm_velocity.
    """
    mu = torch.tensor([[2.0]])
    t = torch.tensor([0.5])
    prior_std = 1.0
    # Pick x_t consistent with x_0 ~ N(0, 1)
    x_0 = torch.tensor([[0.3]])
    x_t = (1 - t) * x_0 + t * mu  # = 0.5*0.3 + 0.5*2.0 = 1.15

    # True conditional velocity from x_t to x_1 = mu:
    v_true = (mu - x_t) / (1 - t)

    s_pred = score_from_fm_velocity(v_true, x_t, t, prior_std)
    # Analytical score: s* = -(x_t - t * mu) / ((1-t)^2 * prior_std^2)
    s_analytical = -(x_t - t * mu) / ((1 - t) ** 2 * prior_std ** 2)
    # The two should agree (possibly up to a sign convention -- to verify)
    # TODO yuli-review: the sign of score_from_fm_velocity vs the
    # analytical formula. Current impl returns (t v - x) / ((1-t) std^2).
    # Plug in v = (mu - x_t)/(1-t): numerator becomes t(mu - x_t)/(1-t) - x_t
    # = [t*mu - t*x_t - (1-t)*x_t] / (1-t) = [t*mu - x_t] / (1-t)
    # So s_pred = [t*mu - x_t] / ((1-t)^2 * std^2).
    # Analytical s* = -(x_t - t*mu) / ((1-t)^2 * std^2) = [t*mu - x_t] / ...
    # So they SHOULD be equal. Let's check.
    assert torch.allclose(s_pred, s_analytical, atol=1e-5), \
        f"s_pred={s_pred}, s_analytical={s_analytical}"


# ---------------------------------------------------------------------------
# Force/energy loss sanity checks
# ---------------------------------------------------------------------------

def test_force_loss_zero_when_match():
    s = torch.randn(10, 3)
    forces = 0.1 * s  # if s = F/kT with kT=0.1
    loss = force_loss(s, forces, kT=0.1)
    assert loss.item() < 1e-6


def test_force_loss_positive_when_mismatch():
    s = torch.ones(5, 3)
    forces = torch.zeros(5, 3)  # mismatch
    loss = force_loss(s, forces, kT=1.0)
    assert loss.item() > 0


def test_score_force_cosine_aligned():
    forces = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    s = forces / 0.5
    cos = score_force_cosine(s, forces, kT=0.5)
    assert abs(cos.item() - 1.0) < 1e-6


def test_score_force_cosine_opposed():
    forces = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    s = -forces
    cos = score_force_cosine(s, forces, kT=1.0)
    assert abs(cos.item() + 1.0) < 1e-6


def test_score_force_cosine_ignores_zero_force_rows():
    forces = torch.tensor([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    s = torch.tensor([[100.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    cos = score_force_cosine(s, forces, kT=0.5)
    assert abs(cos.item() - 1.0) < 1e-6


def test_force_direction_loss_aligned_is_zero():
    forces = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    s = 10.0 * forces
    loss = force_direction_loss(s, forces, mode="cosine")
    assert loss.item() < 1e-6


def test_force_direction_loss_opposed_is_large():
    forces = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    s = -forces
    loss = force_direction_loss(s, forces, mode="cosine")
    assert loss.item() > 1.9


def test_force_direction_norm_mse_ignores_magnitude():
    forces = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    s = torch.tensor([[100.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
    loss = force_direction_loss(s, forces, mode="norm_mse")
    assert loss.item() < 1e-6


def test_energy_loss_variance_zero_when_constant():
    """If log_p + E/kT is constant across batch, loss is 0."""
    energies = torch.tensor([1.0, 2.0, 3.0, 4.0])
    kT = 1.0
    log_p = -energies / kT + 5.0  # constant offset
    loss = energy_loss_variance(log_p, energies, kT)
    assert loss.item() < 1e-6


def test_energy_loss_variance_positive_when_inconsistent():
    energies = torch.tensor([1.0, 2.0, 3.0, 4.0])
    kT = 1.0
    log_p = torch.tensor([1.0, 1.0, 1.0, 1.0])  # flat, but E varies
    loss = energy_loss_variance(log_p, energies, kT)
    assert loss.item() > 0


# ---------------------------------------------------------------------------
# Schedule tests
# ---------------------------------------------------------------------------

def test_bgfm_schedule_warmup():
    fm = torch.tensor(1.0)
    fl = torch.tensor(5.0)
    el = torch.tensor(10.0)
    # During warmup: only FM, lambdas = 0
    total, _ = bgfm_total_loss(fm, fl, el, 0.5, 0.1, epoch_frac=0.05,
                                warmup_frac=0.1, ramp_frac=0.2)
    assert abs(total.item() - 1.0) < 1e-6


def test_bgfm_schedule_mid_ramp():
    fm = torch.tensor(1.0)
    fl = torch.tensor(2.0)
    el = torch.tensor(3.0)
    # Halfway through ramp: lambdas at 50% of full
    total, comps = bgfm_total_loss(fm, fl, el, 0.4, 0.2, epoch_frac=0.2,
                                    warmup_frac=0.1, ramp_frac=0.2)
    # epoch_frac=0.2, warmup=0.1, ramp=0.2 -> (0.2-0.1)/0.2 = 0.5 ramp
    # lambda_1 = 0.4 * 0.5 = 0.2; lambda_2 = 0.2 * 0.5 = 0.1
    expected = 1.0 + 0.2 * 2.0 + 0.1 * 3.0  # = 1.7
    assert abs(total.item() - expected) < 1e-6, \
        f"total={total.item()} vs expected={expected}"


def test_bgfm_schedule_full():
    fm = torch.tensor(1.0)
    fl = torch.tensor(2.0)
    el = torch.tensor(3.0)
    # Past warmup + ramp: full lambdas
    total, _ = bgfm_total_loss(fm, fl, el, 0.5, 0.1, epoch_frac=0.5,
                                warmup_frac=0.1, ramp_frac=0.2)
    expected = 1.0 + 0.5 * 2.0 + 0.1 * 3.0  # = 2.3
    assert abs(total.item() - expected) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
