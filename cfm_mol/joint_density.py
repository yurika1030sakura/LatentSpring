"""Hybrid discrete-continuous joint density for BGFM.

The Boltzmann regularization in BGFM requires a log-density that
covers both the continuous coordinates and the discrete molecular
identity:

    log p_theta(x) = log p_theta(r | c) + log p_theta(c),

with

    log p_theta(c) = log p_theta(N) + sum_i log p_theta(a_i)
                                    + sum_i log p_theta(q_i),

where N is the atom count, a_i are per-atom type tokens, and q_i are
per-atom formal-charge tokens. The continuous term is the FFJORD
instantaneous change-of-variables integral along the coordinate flow
with c held fixed at the endpoint label. The discrete term is the
CTMC bridge path probability of the discrete state under the
FlowMol3 categorical interpolant.

This module provides the discrete log-probability; the continuous
log-density is computed by cfm_mol.bgfm_density.log_density_via_flow
and the two are summed at the call site (see
cfm_mol.bgfm_train_hook).

CTMC bridge log-probability
---------------------------

FlowMol3 parameterizes a categorical CTMC interpolant from a mask /
uniform prior at t=0 to the data category at t=1. For each token,
the per-step log-probability of the bridge is the cross-entropy of
the predicted simplex against the target one-hot endpoint, integrated
over the time discretization. We use the closed-form per-step
contribution
    log p(a_i = a^*_i) ~= log p_theta(a^*_i | x_t, t)
evaluated at the same time-grid used for the FFJORD integral, then
sum across the trajectory. For atom-count we use the marginal
distribution implied by the prior over node-count bins; see
\citet{stark2024flowmol3} for the parameterization.
"""
from __future__ import annotations

from typing import Optional

import torch


def log_prob_atom_count(
    n_atoms: torch.Tensor,
    n_count_logits: torch.Tensor,
) -> torch.Tensor:
    """Log-probability of the observed atom count per graph.

    Args:
        n_atoms: (B,) integer atom counts.
        n_count_logits: (B, N_max + 1) logits over count bins.

    Returns:
        (B,) log-probabilities log p(N = n_atoms).
    """
    log_probs = torch.log_softmax(n_count_logits, dim=-1)
    return log_probs.gather(-1, n_atoms.long().unsqueeze(-1)).squeeze(-1)


def log_prob_discrete_token_path(
    target_idx: torch.Tensor,
    per_t_logits: torch.Tensor,
    t_grid: torch.Tensor,
) -> torch.Tensor:
    """Path log-probability of a discrete CTMC bridge for one token.

    Args:
        target_idx: (N_tokens,) endpoint category for each token.
        per_t_logits: (N_tokens, T, K) per-time-step logits over K
            categories produced by the discrete flow at the same time
            grid used for the FFJORD integral.
        t_grid: (T,) time samples in [0, 1].

    Returns:
        (N_tokens,) summed log-probability along the bridge.
    """
    log_probs = torch.log_softmax(per_t_logits, dim=-1)
    target = target_idx.long().view(-1, 1, 1).expand(-1, log_probs.shape[1], 1)
    chosen = log_probs.gather(-1, target).squeeze(-1)  # (N_tokens, T)
    # Trapezoid-rule integral along t.
    if t_grid.numel() <= 1:
        return chosen.sum(-1)
    dt = (t_grid[1:] - t_grid[:-1]).to(chosen.dtype)
    mid = 0.5 * (chosen[..., 1:] + chosen[..., :-1])
    return (mid * dt).sum(-1)


def log_prob_discrete_state(
    atom_count_logp: torch.Tensor,
    atom_type_logp: torch.Tensor,
    charge_logp: torch.Tensor,
    node_batch_idx: torch.Tensor,
    n_graphs: int,
) -> torch.Tensor:
    """Sum per-token discrete log-probabilities to a per-graph scalar.

    Args:
        atom_count_logp: (B,) log p(N) per graph.
        atom_type_logp: (N_total,) per-atom type log-probability.
        charge_logp: (N_total,) per-atom charge log-probability.
        node_batch_idx: (N_total,) graph index per atom.
        n_graphs: number of graphs.

    Returns:
        (n_graphs,) log p_theta(c) per graph.
    """
    out = atom_count_logp.clone()
    out.scatter_add_(0, node_batch_idx, atom_type_logp)
    out.scatter_add_(0, node_batch_idx, charge_logp)
    return out


def joint_log_prob(
    coord_log_p: torch.Tensor,
    discrete_log_p: torch.Tensor,
) -> torch.Tensor:
    """Combine continuous and discrete contributions:

        log p_theta(x) = log p_theta(r | c) + log p_theta(c).

    Args:
        coord_log_p: (B,) FFJORD coordinate log-density per graph.
        discrete_log_p: (B,) discrete CTMC log-probability per graph.

    Returns:
        (B,) joint log-densities.
    """
    return coord_log_p + discrete_log_p
