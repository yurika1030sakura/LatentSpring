"""Unit tests for cfm_mol.bgfm_density (energy-consistency / Boltzmann term).

Validates the FFJORD-style log-density estimator, especially the SIGN
CONVENTION of the divergence integral, which is the easiest thing to get
wrong. Uses mock velocity fields with analytic densities.

Run:
  cd /n/holylabs/ryl_lab/Lab/yulili_cfm_mol
  conda activate envs/flowmol
  python -m pytest tests/test_bgfm_density.py -v
"""
from __future__ import annotations

import math

import pytest
import torch

from cfm_mol.bgfm_density import (
    gaussian_prior_log_density,
    log_density_via_flow,
    within_group_variance_loss,
    _n_atoms_per_graph,
)


# ---------------------------------------------------------------------------
# Gaussian prior log-density
# ---------------------------------------------------------------------------

def test_gaussian_prior_matches_torch_distributions():
    """gaussian_prior_log_density should match the analytic isotropic
    Gaussian on the full 3N dimension when com_free=False."""
    torch.manual_seed(0)
    # one graph, 4 atoms
    x0 = torch.randn(4, 3)
    n_atoms = torch.tensor([4])
    std = 1.3

    got = gaussian_prior_log_density(x0, n_atoms, prior_std=std, com_free=False)

    # analytic: sum over all 12 dims of N(0, std^2) log-prob
    d = 12
    sq = (x0 ** 2).sum()
    expected = -0.5 * sq / std**2 - 0.5 * d * math.log(2 * math.pi * std**2)
    assert torch.allclose(got, expected.unsqueeze(0), atol=1e-4), \
        f"got={got.item()} expected={expected.item()}"


# ---------------------------------------------------------------------------
# Mock model for flow density tests
# ---------------------------------------------------------------------------

class _MockVF:
    """Mock vector_field returning a controllable position velocity."""
    def __init__(self, mode: str, coeff: float = 0.0):
        self.mode = mode
        self.coeff = coeff

    def __call__(self, g, t_scalar, node_batch_idx=None, upper_edge_mask=None):
        x = g.ndata['x_t']
        if self.mode == "constant":
            v = torch.full_like(x, self.coeff)       # div = 0
        elif self.mode == "linear":
            v = self.coeff * x                        # div = coeff per coord
        else:
            raise ValueError(self.mode)
        return {"x": v}


class _MockModel:
    def __init__(self, vf):
        self.vector_field = vf


class _MockGraph:
    """Minimal DGL-like graph holding ndata + batch_size."""
    def __init__(self, x1, n_atoms_per_graph):
        self._ndata = {"x_1_true": x1.clone(), "x_t": x1.clone()}
        self.batch_size = len(n_atoms_per_graph)
        self.device = x1.device
        self._n = n_atoms_per_graph

    @property
    def ndata(self):
        return self._ndata


def test_near_zero_divergence():
    """Near-zero linear coefficient => divergence ~ 0, so
    log p_1(x_1) ~ log p_0(x_0). Uses a tiny linear field (a=1e-4) so
    autograd has a valid grad path while divergence stays negligible.

    (A literally constant field has a disconnected autograd graph; that
    edge case is handled in divergence_hutchinson via allow_unused, but
    is not exercised by the real GVP model which always depends on x.)
    """
    torch.manual_seed(1)
    n_atoms = torch.tensor([3])
    node_batch_idx = torch.zeros(3, dtype=torch.long)
    x1 = torch.randn(3, 3)
    a = 1e-4

    g = _MockGraph(x1, n_atoms)
    model = _MockModel(_MockVF("linear", coeff=a))

    logp = log_density_via_flow(
        model, g, node_batch_idx, upper_edge_mask=None,
        n_ode_steps=20, n_hutchinson=4, prior_std=1.0)

    # x_0 ~ x_1 (tiny contraction), div integral ~ 3*3*a ~ 9e-4 ~ 0
    x0_expected = x1 * math.exp(-a)
    logp0_expected = gaussian_prior_log_density(x0_expected, n_atoms, prior_std=1.0)
    assert torch.allclose(logp, logp0_expected, atol=1e-2), \
        f"logp={logp.item()} expected={logp0_expected.item()}"


def test_linear_velocity_divergence_sign():
    """Linear velocity v = a*x has divergence 3N*a (per graph, summed over
    real coords). Reverse ODE: dx/dt = a*x integrated 1->0.

    Forward density change: d/dt log p_t = -div(v) = -3N*a (constant).
    => log p_1(x_1) = log p_0(x_0) - 3N*a.

    We check the SIGN: for a > 0 (expanding flow), density should DROP from
    prior to data (log p_1 < log p_0(x_0)); estimator must reflect that.
    """
    torch.manual_seed(2)
    n_atoms = torch.tensor([2])
    node_batch_idx = torch.zeros(2, dtype=torch.long)
    x1 = torch.randn(2, 3)
    a = 0.3

    g = _MockGraph(x1, n_atoms)
    model = _MockModel(_MockVF("linear", coeff=a))

    logp = log_density_via_flow(
        model, g, node_batch_idx, upper_edge_mask=None,
        n_ode_steps=50, n_hutchinson=8, prior_std=1.0)

    # Reverse-time exact: x(t) = x_1 * exp(a (t - 1)); x_0 = x_1 exp(-a).
    x0 = x1 * math.exp(-a)
    logp0 = gaussian_prior_log_density(x0, n_atoms, prior_std=1.0)
    # divergence integral over [0,1] = 3*N*a (full coords; mock uses 3*N)
    div_integral = 3 * 2 * a
    expected = logp0 - div_integral
    # Hutchinson + Euler => loose tolerance
    assert torch.allclose(logp, expected, atol=0.15), \
        f"logp={logp.item()} expected={expected.item()} (div_integral={div_integral})"


def test_n_atoms_per_graph_helper():
    nbi = torch.tensor([0, 0, 0, 1, 1, 2])
    out = _n_atoms_per_graph(None, nbi)
    assert out.tolist() == [3, 2, 1]


# ---------------------------------------------------------------------------
# within_group_variance_loss: the core of L_energy_per_mol
# ---------------------------------------------------------------------------

def test_within_group_variance_isolates_within_mol():
    """Construct residuals where within-parent variation is small (~0.01)
    but across-parent variation is huge (~100). The within-group loss should
    see only the small within-mol part, NOT the across-mol log-Z spread.

    This is the central correctness claim of the per-mol fix: bad cross-batch
    variance was ~(100)^2 = 1e4; within-group variance should be ~(0.01)^2 = 1e-4.
    """
    torch.manual_seed(0)
    # 3 parents, K=4 perturbations each
    parent_id = torch.tensor([0,0,0,0, 1,1,1,1, 2,2,2,2], dtype=torch.long)
    # Within-parent: small random jitter (~ N(0, 0.01))
    within = torch.randn(12) * 0.01
    # Across-parent: large offsets per parent (the per-mol log Z confound)
    across = torch.tensor([0.0]*4 + [100.0]*4 + [-50.0]*4)
    residual = within + across

    loss, diag = within_group_variance_loss(residual, parent_id)

    # Within-group var ~ 0.01^2 ~ 1e-4. Should not see the 100 / -50 offsets.
    assert loss.item() < 1e-2, \
        f"loss={loss.item()} too large -- across-mol spread is leaking in"
    assert diag["n_groups_used"] == 3
    assert diag["n_valid"] == 12

    # Sanity: if we'd instead taken cross-batch variance, it'd be ~5e3.
    cross_var = residual.var().item()
    assert cross_var > 100.0, "sanity: synthetic spread should be huge cross-batch"


def test_within_group_variance_drops_nan():
    """NaN energies (failed OMol25) should be silently skipped, not poison
    the loss or counts."""
    parent_id = torch.tensor([0,0,0,0, 1,1,1,1], dtype=torch.long)
    residual = torch.tensor([1.0, 1.0, float("nan"), 1.0,  # parent 0: one NaN
                             2.0, 2.0, 2.0, 2.0])           # parent 1: all valid
    loss, diag = within_group_variance_loss(residual, parent_id)
    # Both parents have all-equal valid residuals => loss = 0
    assert loss.item() == 0.0
    assert diag["n_valid"] == 7
    assert diag["n_groups_used"] == 2


def test_within_group_variance_skips_singleton_groups():
    """Parents with < 2 valid samples have no within-group variance to
    compute and must be excluded."""
    parent_id = torch.tensor([0,0,0,0, 1,1,1,1], dtype=torch.long)
    # parent 1 has only 1 valid -> excluded
    residual = torch.tensor([5.0, 10.0, 5.0, 10.0,
                             1.0, float("nan"), float("nan"), float("nan")])
    loss, diag = within_group_variance_loss(residual, parent_id)
    # Only parent 0 contributes; its mean = 7.5, var = ((5-7.5)^2*2 + (10-7.5)^2*2)/4 = 6.25
    assert abs(loss.item() - 6.25) < 1e-5
    assert diag["n_groups_used"] == 1


def test_within_group_variance_backprops():
    """Loss must be differentiable so the energy term can train the model.
    Use a leaf tensor for residual and verify .grad populates after backward."""
    parent_id = torch.tensor([0,0,0,0, 1,1,1,1], dtype=torch.long)
    residual = torch.tensor(
        [0.1, 0.3, -0.2, 0.5,  1.0, 0.9, 1.2, 0.8], requires_grad=True)
    loss, _ = within_group_variance_loss(residual, parent_id)
    loss.backward()
    assert residual.grad is not None
    assert torch.isfinite(residual.grad).all()
    assert residual.grad.abs().sum().item() > 0, "gradient should not be zero"


def test_within_group_variance_zero_when_each_parent_constant():
    """Each parent's residual is constant across perturbations => loss = 0
    (perfect Boltzmann condition modulo per-mol log Z)."""
    parent_id = torch.tensor([0,0,0,0, 1,1,1,1, 2,2,2,2], dtype=torch.long)
    residual = torch.tensor([1.5]*4 + [-200.7]*4 + [42.0]*4)
    loss, _ = within_group_variance_loss(residual, parent_id)
    assert loss.item() < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
