"""P1 -- analytic unit tests behind scripts/p1_estimator_precision.py.

These pin down three things the numerical study relies on:

  1. ``divergence_exact_atomwise`` returns the exact Jacobian trace (it is the
     reference against which the production estimator is measured).
  2. The 2-probe Rademacher Hutchinson estimator used in production is
     UNBIASED for that trace, with the analytic variance
         Var = (1/k) * sum_{i<j} (A_ij + A_ji)^2 .
  3. In ``log_density_via_flow`` the reverse trajectory is a deterministic
     function of the model and x_1: the Hutchinson probes never feed back into
     the state update (it runs under ``torch.no_grad()`` using only v).
     This is what makes repeat-scoring variance pure quadrature noise, with no
     accompanying trajectory jitter, and it is what licenses the
     shared-trajectory repeats used by TASK B of the study.

Run:
  cd /n/home04/yulili/bgfm
  /n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/bin/python -m pytest \
      tests/test_estimator_precision.py -v
"""
from __future__ import annotations

import math

import torch

from cfm_mol.bgfm_loss import divergence_exact_atomwise, divergence_hutchinson
from cfm_mol.bgfm_density import log_density_via_flow


def _linear_field(seed=0, n_atoms=6):
    torch.manual_seed(seed)
    d = 3 * n_atoms
    A = torch.randn(d, d, dtype=torch.float64) / math.sqrt(d)
    x = torch.randn(n_atoms, 3, dtype=torch.float64)
    n_apg = torch.tensor([n_atoms])

    def v_fn(z):
        return (A @ z.reshape(-1)).reshape(n_atoms, 3)

    return A, x, n_apg, v_fn


def test_exact_divergence_is_exact():
    A, x, n_apg, v_fn = _linear_field()
    got = float(divergence_exact_atomwise(
        v_fn, x.clone().requires_grad_(True), n_apg, create_graph=False)[0])
    want = float(torch.diagonal(A).sum())
    assert abs(got - want) < 1e-12, (got, want)


def test_hutchinson2_unbiased_with_analytic_variance():
    A, x, n_apg, v_fn = _linear_field()
    d = A.shape[0]
    tr = float(torch.diagonal(A).sum())
    var = sum(float((A[i, j] + A[j, i]) ** 2)
              for i in range(d) for j in range(i + 1, d)) / 2.0   # k = 2 probes

    torch.manual_seed(1234)
    n = 20000
    est = torch.tensor([
        float(divergence_hutchinson(v_fn, x.clone().requires_grad_(True), n_apg,
                                    n_samples=2, rademacher=True,
                                    create_graph=False)[0])
        for _ in range(n)], dtype=torch.float64)

    sd = float(est.std())
    sem = sd / math.sqrt(n)
    bias = float(est.mean()) - tr
    # unbiased: bias within 4 standard errors of zero
    assert abs(bias) < 4 * sem, (bias, sem)
    # variance matches the closed form to within 5%
    assert abs(sd - math.sqrt(var)) / math.sqrt(var) < 0.05, (sd, math.sqrt(var))


# ---------------------------------------------------------------------------
# trajectory determinism
# ---------------------------------------------------------------------------
class _MixingVF:
    """v = a*x + c*roll(x, 1, dim=-1): Jacobian has off-diagonal entries, so the
    Hutchinson trace estimate is genuinely stochastic (unlike v = a*x)."""

    def __init__(self, a=0.3, c=0.4):
        self.a, self.c = a, c

    def __call__(self, g, t_scalar, node_batch_idx=None, upper_edge_mask=None):
        x = g.ndata['x_t']
        return {"x": self.a * x + self.c * torch.roll(x, 1, dims=-1)}


class _MockModel:
    def __init__(self, vf):
        self.vector_field = vf


class _MockGraph:
    def __init__(self, x1, batch_size):
        self._ndata = {"x_1_true": x1.clone(), "x_t": x1.clone()}
        self.batch_size = batch_size
        self.device = x1.device

    @property
    def ndata(self):
        return self._ndata


def test_reverse_trajectory_is_probe_independent():
    """Two runs with different probe draws must leave an IDENTICAL state
    trajectory, while the returned log-density differs (quadrature noise only)."""
    torch.manual_seed(7)
    n = 5
    x1 = torch.randn(n, 3, dtype=torch.float64)
    nbi = torch.zeros(n, dtype=torch.long)

    outs, states = [], []
    for seed in (11, 12):
        torch.manual_seed(seed)
        g = _MockGraph(x1, batch_size=1)
        model = _MockModel(_MixingVF())
        lp = log_density_via_flow(model, g, nbi, upper_edge_mask=None,
                                  n_ode_steps=12, n_hutchinson=2, prior_std=1.0)
        outs.append(float(lp[0]))
        states.append(g.ndata['x_t'].detach().clone())

    assert torch.equal(states[0], states[1]), "trajectory depends on the probes"
    assert outs[0] != outs[1], "probes had no effect -- test field is not mixing"
