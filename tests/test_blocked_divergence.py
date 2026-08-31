"""Tests for the optional `xi_provider` hook on divergence_hutchinson.

The global-ensemble eval (scripts/eval_global_ensemble.py) needs EXACT
divergences: the inter-basin Boltzmann signal is O(0.1) nat while the Hutchinson
error is O(6) nats. It gets them cheaply by exploiting the fact that a batched
DGL graph is block-diagonal -- a probe equal to sqrt(3N) * e_(i,c) replicated in
every graph returns, per graph, exactly 3N * J_(i,c),(i,c), so averaging the 3N
such probes yields the exact trace for EVERY graph at a cost of 3N backward
passes total, independent of batch size.

These tests pin that identity and the backward compatibility of the default
(no xi_provider) path.
"""
import math

import pytest
import torch

from cfm_mol.bgfm_loss import divergence_hutchinson, divergence_exact_atomwise


def _block_diagonal_field(B, N, seed=0):
    """A nonlinear velocity field where graphs do not interact."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(3, 3, generator=g) * 0.4
    A = torch.randn(N, N, generator=g) * 0.3

    def v_fn(x):
        xb = x.view(B, N, 3)
        out = torch.tanh(torch.einsum("ij,bjc->bic", A, xb)) @ W + 0.3 * xb ** 3
        return out.reshape(B * N, 3)

    return v_fn


def _blocked_provider(B, N):
    scale = math.sqrt(3.0 * N)

    def xi_provider(k, xx):
        base = torch.zeros(N, 3)
        base[k // 3, k % 3] = scale
        return base.repeat(B, 1)

    return xi_provider


@pytest.mark.parametrize("B,N", [(1, 4), (4, 5), (7, 3)])
def test_blocked_probes_give_exact_divergence(B, N):
    torch.manual_seed(0)
    v_fn = _block_diagonal_field(B, N)
    x = torch.randn(B * N, 3, requires_grad=True)
    napg = torch.full((B,), N, dtype=torch.long)

    exact = divergence_exact_atomwise(v_fn, x, napg, create_graph=False)
    blocked = divergence_hutchinson(v_fn, x, napg, n_samples=3 * N,
                                    create_graph=False,
                                    xi_provider=_blocked_provider(B, N))
    assert torch.allclose(exact, blocked, atol=1e-4), (exact, blocked)


def test_blocked_estimate_is_deterministic():
    """Zero stochastic error: repeated calls must agree bit-for-bit."""
    torch.manual_seed(0)
    B, N = 3, 4
    v_fn = _block_diagonal_field(B, N)
    x = torch.randn(B * N, 3, requires_grad=True)
    napg = torch.full((B,), N, dtype=torch.long)
    prov = _blocked_provider(B, N)
    a = divergence_hutchinson(v_fn, x, napg, n_samples=3 * N,
                              create_graph=False, xi_provider=prov)
    b = divergence_hutchinson(v_fn, x, napg, n_samples=3 * N,
                              create_graph=False, xi_provider=prov)
    assert torch.equal(a, b)


def test_default_path_unchanged_and_unbiased():
    """No xi_provider -> original i.i.d. Rademacher behaviour (training path)."""
    torch.manual_seed(0)
    B, N = 2, 4
    v_fn = _block_diagonal_field(B, N)
    x = torch.randn(B * N, 3, requires_grad=True)
    napg = torch.full((B,), N, dtype=torch.long)
    exact = divergence_exact_atomwise(v_fn, x, napg, create_graph=False)
    est = divergence_hutchinson(v_fn, x, napg, n_samples=4000,
                                create_graph=False)
    assert est.shape == exact.shape
    # Hutchinson is unbiased; 4000 samples should land close.
    assert torch.allclose(est, exact, rtol=0.15, atol=0.5), (est, exact)


def test_xi_provider_shape_is_validated():
    torch.manual_seed(0)
    B, N = 2, 4
    v_fn = _block_diagonal_field(B, N)
    x = torch.randn(B * N, 3, requires_grad=True)
    napg = torch.full((B,), N, dtype=torch.long)
    with pytest.raises(ValueError):
        divergence_hutchinson(v_fn, x, napg, n_samples=1, create_graph=False,
                              xi_provider=lambda k, xx: torch.zeros(N, 3))
