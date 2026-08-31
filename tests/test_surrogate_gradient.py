"""Unit tests for the full-gradient vs surrogate-gradient measurement (P1).

These pin the three facts the measurement in
`scripts/validate_surrogate_gradient.py` rests on:

  1. Detaching changes only the GRAPH, never the VALUE: the fully
     differentiable likelihood and the implemented one return identical
     log-densities for identical probes.
  2. The reference implementation reproduces the shipped one: with both
     paths detached, `log_density_full_gradient` gives exactly the gradient
     of `log_density_via_flow(..., for_training=True)`.
  3. The reference IS the gradient of the estimator: it matches central
     finite differences of the objective.
  4. The shipped surrogate is NOT that gradient (documents FINDING 2).

Run:
    python -m pytest tests/test_surrogate_gradient.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cfm_mol.bgfm_density import log_density_via_flow  # noqa: E402
from scripts.validate_surrogate_gradient import (  # noqa: E402
    ToyModel,
    compare,
    flat,
    grad_dict,
    log_density_full_gradient,
    loss_energy,
    make_xi_bank,
    synthetic_batch,
)

N_STEPS = 3
N_HUTCH = 0        # exact divergence -> deterministic


def _setup():
    model = ToyModel(hidden=16, seed=7).to(dtype=torch.float64)
    g, nbi, uem, E, pid = synthetic_batch(M=2, K=3, n_atoms=4, seed=11,
                                          dtype=torch.float64)
    named = [(n, p) for n, p in model.named_parameters()]
    xi_fn = make_xi_bank(tuple(g.ndata['x_1_true'].shape), N_STEPS, N_HUTCH,
                         5, g.ndata['x_1_true'].device, torch.float64)
    return model, g, nbi, uem, E, pid, named, xi_fn


def _logp(model, g, nbi, uem, xi_fn, variant):
    g.ndata['x_t'] = g.ndata['x_1_true'].clone()
    if variant == "impl":
        return log_density_via_flow(
            model, g, nbi, uem, n_ode_steps=N_STEPS, n_hutchinson=N_HUTCH,
            prior_std=1.0, for_training=True, xi_fn=xi_fn)
    return log_density_full_gradient(
        model, g, nbi, uem, n_ode_steps=N_STEPS, n_hutchinson=N_HUTCH,
        prior_std=1.0, xi_fn=xi_fn,
        detach_prior=(variant == "ref_surr"),
        detach_trajectory=(variant == "ref_surr"))


def test_values_identical_across_graph_variants():
    """Detaching cannot move the number: log p must agree exactly."""
    model, g, nbi, uem, E, pid, named, xi_fn = _setup()
    vals = {v: _logp(model, g, nbi, uem, xi_fn, v).detach()
            for v in ("impl", "ref_surr", "full")}
    for v, x in vals.items():
        dev = float((x - vals["impl"]).abs().max())
        print(f"  max |logp({v}) - logp(impl)| = {dev:.3e}")
        assert dev == 0.0


def test_reference_reproduces_shipped_surrogate_gradient():
    """Both paths detached => exactly the shipped implementation's gradient."""
    model, g, nbi, uem, E, pid, named, xi_fn = _setup()
    g_impl = grad_dict(loss_energy(_logp(model, g, nbi, uem, xi_fn, "impl"),
                                   E, pid), named)
    g_ref = grad_dict(loss_energy(_logp(model, g, nbi, uem, xi_fn, "ref_surr"),
                                  E, pid), named)
    names = [n for n, _ in named]
    c = compare(g_impl, g_ref, names)
    md = max(float((g_impl[n] - g_ref[n]).abs().max()) for n in names)
    print(f"  cos = {c['cos']:.12f}  rel_norm = {c['rel_norm']:.12f}  "
          f"max abs diff = {md:.3e}")
    assert md == 0.0
    assert abs(c["cos"] - 1.0) < 1e-12


def test_full_gradient_matches_finite_differences():
    """The fully differentiable path is the true gradient of the estimator."""
    model, g, nbi, uem, E, pid, named, xi_fn = _setup()
    names = [n for n, _ in named]
    g_full = grad_dict(loss_energy(_logp(model, g, nbi, uem, xi_fn, "full"),
                                   E, pid), named)
    torch.manual_seed(3)
    d = {n: torch.randn_like(p) for n, p in named}
    dn = flat(d, names).norm()
    d = {n: v / dn for n, v in d.items()}
    pred = float(flat(g_full, names) @ flat(d, names))

    h = 1e-5
    with torch.no_grad():
        for n, p in named:
            p += h * d[n]
    Lp = float(loss_energy(_logp(model, g, nbi, uem, xi_fn, "impl"), E, pid))
    with torch.no_grad():
        for n, p in named:
            p -= 2 * h * d[n]
    Lm = float(loss_energy(_logp(model, g, nbi, uem, xi_fn, "impl"), E, pid))
    with torch.no_grad():
        for n, p in named:
            p += h * d[n]
    fd = (Lp - Lm) / (2 * h)
    rel = abs(fd - pred) / (abs(pred) + 1e-30)
    print(f"  <g_full, d> = {pred:+.10f}   central FD = {fd:+.10f}   "
          f"rel err = {rel:.3e}")
    assert rel < 1e-6


def test_shipped_surrogate_is_not_the_full_gradient():
    """FINDING 2, made numerical: the shipped update is a surrogate."""
    model, g, nbi, uem, E, pid, named, xi_fn = _setup()
    names = [n for n, _ in named]
    g_full = grad_dict(loss_energy(_logp(model, g, nbi, uem, xi_fn, "full"),
                                   E, pid), named)
    g_surr = grad_dict(loss_energy(_logp(model, g, nbi, uem, xi_fn, "impl"),
                                   E, pid), named)
    c = compare(g_full, g_surr, names)
    print(f"  cos(g_full, g_surr) = {c['cos']:.6f}   "
          f"||g_surr||/||g_full|| = {c['rel_norm']:.6f}")
    assert c["cos"] < 0.999, "surrogate unexpectedly equals the full gradient"
    assert c["cos"] > 0.0, "surrogate is not even a descent direction here"
