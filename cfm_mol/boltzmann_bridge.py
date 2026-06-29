"""Local Boltzmann bridge loss (BGFM-Native module).

This module implements the *Local Boltzmann bridge* — the seventh BGFM
loss term (lambda_7 * L_bridge). It is the algorithmic counterpart of the
density-energy variance loss in `bgfm_density`, but formulated as an
explicit KL match between the model's coordinate distribution and an
OMol25-induced Boltzmann distribution, both restricted to a finite
perturbation cloud around a parent molecule.

Setup. For each parent molecule m we have K perturbations
{r_m^(k)}_{k=1..K} with OMol25 energies {E_m^(k)}. Define

    w_m^(k) = softmax_k[ -beta * E_m^(k) ]                  (target)
    q_theta^(k) = softmax_k[ log p_theta^pos(r_m^(k) | c_m) ] (model)

Note the unknown partition function Z_c cancels inside the softmax over
the cloud (it shifts every cloud entry by the same constant), so the loss
is well-defined even though `log p_theta` is only known up to a per-
parent constant in practice.

The loss is

    L_bridge = mean_m  KL( w_m || q_theta^(m) )
             = mean_m  sum_k w_m^(k) * ( log w_m^(k) - log q_theta^(m,k) ).

Properties:
  - Translation/rotation invariance: inherits from the position log-density.
  - Per-parent constant invariance: log w and log q both shift by the same
    constant under (E -> E + c) or (log p -> log p + c'). Constants drop
    out of the softmax, so KL is unchanged.
  - Reduces to variance loss in the small-fluctuation limit: when
    log p - (-beta E) is approximately constant across the cloud,
    KL(w || q) ~ (1/2) Var_k[ log p + beta E ] + O(.^3). The bridge form
    is more numerically stable for large clouds and is easier to reason
    about as an explicit distribution-matching objective.

This is meant to be used together with — not as a replacement for — the
density-energy variance loss. The variance form is theoretically cleaner;
the bridge form is the discrete finite-K target reviewers more readily
recognize as a distribution-matching algorithm.

Reference: Cao et al. (2024) score distillation discussion;
Lipman et al. (2023) flow matching; Grathwohl et al. (2019) FFJORD.
"""
from __future__ import annotations

from typing import Tuple

import torch


def per_parent_softmax(
    scores: torch.Tensor,
    parent_id: torch.Tensor,
    n_parents: int,
) -> torch.Tensor:
    """Numerically-stable per-parent log-softmax over a flat (B*K,) tensor.

    Args:
        scores: (B*K,) per-virtual-mol log-scores (e.g., log p_theta or
            -beta * E). May contain large magnitudes; we subtract the
            per-parent max before exp.
        parent_id: (B*K,) int64 in 0..n_parents-1, mapping virtual mols
            to their parent.
        n_parents: B (number of parents).

    Returns:
        log_p: (B*K,) the log-normalized softmax weights, one per virtual
            mol. sum over each parent's cloud of exp(log_p) is 1.
    """
    device = scores.device
    dtype = scores.dtype
    very_neg = torch.full((n_parents,), -float("inf"), device=device, dtype=dtype)
    # Per-parent max via scatter_reduce (amax).
    max_per_parent = very_neg.scatter_reduce(
        0, parent_id, scores, reduce="amax", include_self=True,
    )
    shifted = scores - max_per_parent[parent_id]
    exps = shifted.exp()
    Z_per_parent = torch.zeros(n_parents, device=device, dtype=dtype)
    Z_per_parent.scatter_add_(0, parent_id, exps)
    # Guard against an empty parent (shouldn't happen but keeps autograd safe).
    log_Z = Z_per_parent.clamp_min(1e-30).log()
    return shifted - log_Z[parent_id]


def boltzmann_bridge_loss(
    log_p_coord: torch.Tensor,
    energies_eV: torch.Tensor,
    parent_id: torch.Tensor,
    n_parents: int,
    beta: float,
    symmetric: bool = False,
    return_diagnostics: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """Local Boltzmann bridge KL loss.

    For each parent m, computes
        KL( w_m || q_theta^(m) )
    with
        w_m^(k)        = softmax_k(-beta * E_m^(k))
        q_theta^(m,k)  = softmax_k( log p_theta^pos(r_m^(k) | c_m) ).

    Args:
        log_p_coord: (B*K,) model coordinate log-densities for each
            perturbed virtual mol. Typically produced by FFJORD through
            the BGFM vector field (the same tensor that feeds
            within_group_variance_loss).
        energies_eV: (B*K,) OMol25 energies in eV.
        parent_id: (B*K,) int64 in 0..B-1.
        n_parents: B.
        beta: inverse temperature in eV^{-1}. Pass 1/kT with kT in eV.
        symmetric: if True, return the symmetrized KL
            0.5 * (KL(w || q) + KL(q || w)). Default False (forward KL,
            which is mode-covering, matches the OMol25 "trust the teacher"
            interpretation).
        return_diagnostics: if True, include extra scalars (entropy of w
            and q, mean cloud size) for logging.

    Returns:
        loss: scalar tensor, mean over parents of the per-parent KL.
        diag: dict with logging scalars.
    """
    if log_p_coord.shape != energies_eV.shape:
        raise RuntimeError(
            f"shape mismatch: log_p_coord {log_p_coord.shape} vs "
            f"energies_eV {energies_eV.shape}"
        )
    if parent_id.shape != log_p_coord.shape:
        raise RuntimeError(
            f"parent_id shape {parent_id.shape} must match log_p_coord "
            f"{log_p_coord.shape}"
        )
    log_w = per_parent_softmax(-beta * energies_eV, parent_id, n_parents)
    log_q = per_parent_softmax(log_p_coord, parent_id, n_parents)
    # Forward KL(w || q) per virtual mol contribution: w * (log w - log q).
    kl_terms = log_w.exp() * (log_w - log_q)
    # Sum per parent => (B,), then mean over parents.
    kl_per_parent = torch.zeros(n_parents, device=log_p_coord.device, dtype=log_p_coord.dtype)
    kl_per_parent.scatter_add_(0, parent_id, kl_terms)

    if symmetric:
        kl_rev_terms = log_q.exp() * (log_q - log_w)
        kl_rev_per_parent = torch.zeros_like(kl_per_parent)
        kl_rev_per_parent.scatter_add_(0, parent_id, kl_rev_terms)
        loss = 0.5 * (kl_per_parent.mean() + kl_rev_per_parent.mean())
    else:
        loss = kl_per_parent.mean()

    diag: dict = {}
    if return_diagnostics:
        with torch.no_grad():
            ent_w_terms = -log_w.exp() * log_w
            ent_q_terms = -log_q.exp() * log_q
            ent_w_pp = torch.zeros_like(kl_per_parent)
            ent_q_pp = torch.zeros_like(kl_per_parent)
            ent_w_pp.scatter_add_(0, parent_id, ent_w_terms)
            ent_q_pp.scatter_add_(0, parent_id, ent_q_terms)
            count_pp = torch.zeros_like(kl_per_parent)
            count_pp.scatter_add_(0, parent_id, torch.ones_like(log_p_coord))
            diag = {
                "bridge_kl_per_parent_mean": float(kl_per_parent.mean().item()),
                "bridge_kl_per_parent_max": float(kl_per_parent.max().item()),
                "bridge_entropy_w_mean": float(ent_w_pp.mean().item()),
                "bridge_entropy_q_mean": float(ent_q_pp.mean().item()),
                "bridge_cloud_size_mean": float(count_pp.mean().item()),
            }
    return loss, diag


__all__ = [
    "per_parent_softmax",
    "boltzmann_bridge_loss",
]
