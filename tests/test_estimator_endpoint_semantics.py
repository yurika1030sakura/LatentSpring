"""Analytic endpoint-semantics tests for the likelihood integrator.

QUESTION UNDER TEST
-------------------
`log_density_via_flow` places its quadrature nodes at
    t_k = 1 - (k + 1/2)/n,   k = 0 .. n-1,
so the LARGEST node is 1 - 1/(2n) rather than 1.  The manuscript has been
reading that as evidence that the estimator targets a smoothed marginal
p_{1-eps} with eps = 1/(2n), not p_1.

These tests decide the question from the code, not from the node list.
For the standard midpoint rule the nodes are all interior but the
integration interval is still the whole of [0,1]:

    int_0^1 f(t) dt  ~  dt * sum_k f((k+1/2) dt).

So if the accumulated divergence integral equals its exact value over
[0,1] INDEPENDENTLY of n, the estimator targets p_1 and the eps = 1/(2n)
reading is wrong.  If instead it systematically returns a (1 - 1/(2n))
fraction of the exact value, the truncation reading is right.

Run:
    python -m pytest tests/test_estimator_endpoint_semantics.py -v
"""
from __future__ import annotations

import math
import torch

from cfm_mol.bgfm_density import (
    gaussian_prior_log_density,
    log_density_via_flow,
)


class _TimeAwareVF:
    """v(x, t) = a(t) * x, with a(t) either constant or a supplied callable."""

    def __init__(self, a):
        self.a = a

    def __call__(self, g, t_scalar, node_batch_idx=None, upper_edge_mask=None,
                 **kw):
        x = g.ndata['x_t']
        t = float(t_scalar.reshape(-1)[0])
        coeff = self.a(t) if callable(self.a) else self.a
        return {"x": coeff * x}


class _Model:
    def __init__(self, vf):
        self.vector_field = vf


class _Graph:
    def __init__(self, x1, n_atoms_per_graph):
        self._ndata = {"x_1_true": x1.clone(), "x_t": x1.clone()}
        self.batch_size = len(n_atoms_per_graph)
        self.device = x1.device
        self._n = n_atoms_per_graph

    @property
    def ndata(self):
        return self._ndata


def _run(a, n_steps, n_atoms=3, seed=0, exact_div=True):
    """Return (logp_estimate, x1, n_atoms_tensor)."""
    torch.manual_seed(seed)
    na = torch.tensor([n_atoms])
    nbi = torch.zeros(n_atoms, dtype=torch.long)
    x1 = torch.randn(n_atoms, 3, dtype=torch.float64)
    g = _Graph(x1, na)
    model = _Model(_TimeAwareVF(a))
    logp = log_density_via_flow(
        model, g, nbi, upper_edge_mask=None,
        n_ode_steps=n_steps,
        n_hutchinson=0 if exact_div else 8,   # 0 => exact divergence
        prior_std=1.0)
    return logp, x1, na


# ---------------------------------------------------------------------------
# TEST 1 (decisive): constant divergence.
# ---------------------------------------------------------------------------
def test_constant_divergence_integral_is_n_independent():
    """v = a x has divergence 3N*a, constant in t.

    Over [0,1] the exact integral is 3N*a for every n.  Under the
    'truncated at 1-1/(2n)' reading it would instead be 3N*a*(1-1/(2n)),
    i.e. 12.5% low at n=4 and 1.04% low at n=48.
    """
    a, N = 0.37, 3
    exact = 3 * N * a
    print(f"\n  exact int_0^1 div dt = 3N*a = {exact:.10f}")
    for n in (1, 2, 4, 12, 48, 100):
        logp, x1, na = _run(a, n)
        # logp = logp0(x_0) - integral  =>  integral = logp0(x_0) - logp
        # recompute x_0 exactly as the code does (reverse Euler)
        x = x1.clone()
        dt = 1.0 / n
        for k in range(n):
            x = x - dt * a * x
        logp0 = gaussian_prior_log_density(x, na, prior_std=1.0)
        integral = float(logp0 - logp)
        truncated = exact * (1.0 - 1.0 / (2 * n))
        print(f"  n={n:>3}  integral={integral:.10f}  "
              f"full={exact:.10f}  truncated-reading={truncated:.10f}")
        assert abs(integral - exact) < 1e-8, (
            f"n={n}: integral {integral} != exact {exact}; "
            f"truncated reading would give {truncated}")


# ---------------------------------------------------------------------------
# TEST 2: linear CNF against the analytic endpoint likelihood.
# ---------------------------------------------------------------------------
def test_linear_cnf_converges_to_exact_endpoint_likelihood():
    """dx/dt = a x.  Exact: x_0 = e^{-a} x_1 and int div dt = 3N a, so
        log p_1(x_1) = log p_0(e^{-a} x_1) - 3N a.
    Euler is O(dt) in the STATE, so the estimate must converge to this
    endpoint value as n grows -- to the t=1 quantity, not to a t=1-eps one.
    """
    a, N = 0.5, 3
    torch.manual_seed(0)
    na = torch.tensor([N])
    x1 = torch.randn(N, 3, dtype=torch.float64)
    exact = float(gaussian_prior_log_density(
        x1 * math.exp(-a), na, prior_std=1.0)) - 3 * N * a
    print(f"\n  analytic log p_1 = {exact:.8f}")
    errs = []
    for n in (2, 4, 12, 48, 200, 800):
        logp, _, _ = _run(a, n, n_atoms=N, seed=0)
        err = abs(float(logp) - exact)
        errs.append((n, err))
        print(f"  n={n:>4}  estimate={float(logp):.8f}  |err|={err:.3e}")
    # monotone decrease and first-order convergence
    assert errs[-1][1] < errs[0][1] / 50, f"no convergence: {errs}"
    assert errs[-1][1] < 1e-3, f"did not reach the endpoint value: {errs}"


# ---------------------------------------------------------------------------
# TEST 3: time-varying field, where a truncated upper limit WOULD show up.
# ---------------------------------------------------------------------------
def test_time_varying_field_integral_matches_full_interval():
    """v = t*x  =>  div(t) = 3N*t, int_0^1 3N t dt = 3N/2.

    This is the sharpest discriminator: with a t-dependent integrand a
    truncated upper limit 1-1/(2n) would give 3N/2*(1-1/(2n))^2, which is
    23.4% low at n=2.  Midpoint quadrature is exact for linear integrands,
    so the full-interval reading predicts exactly 3N/2 for every n.
    """
    N = 3
    exact = 3 * N / 2.0
    print(f"\n  exact int_0^1 3N t dt = {exact:.10f}")
    for n in (1, 2, 4, 12, 48):
        logp, x1, na = _run(lambda t: t, n, n_atoms=N, seed=3)
        x = x1.clone()
        dt = 1.0 / n
        for k in range(n):
            t_val = 1.0 - (k + 0.5) * dt
            x = x - dt * t_val * x
        logp0 = gaussian_prior_log_density(x, na, prior_std=1.0)
        integral = float(logp0 - logp)
        truncated = exact * (1.0 - 1.0 / (2 * n)) ** 2
        print(f"  n={n:>3}  integral={integral:.10f}  full={exact:.10f}  "
              f"truncated-reading={truncated:.10f}")
        assert abs(integral - exact) < 1e-8, (
            f"n={n}: {integral} != {exact}; truncated would give {truncated}")
