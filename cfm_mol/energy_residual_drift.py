"""Energy-residual drift for BGFM.

BGFM upgrades the standard flow-matching velocity field

    v_theta(x_t, t)

into an energy-calibrated vector field

    v_BGFM(x_t, t) = v_theta(x_t, t) + alpha(t) * P_SE3[ -grad_r Delta E_psi(r_t, c) ],

where Delta E_psi is the coordinate-dependent residual of the
composition-conditioned strain head (see cfm_mol.energy_head; the
composition baseline b_phi(c) drops out of the gradient by
construction). The projection P_SE3 removes the center-of-mass drift
inside each graph so the drift respects translation equivariance; SO(3)
equivariance is inherited from the energy head being a rotation-
invariant scalar of pairwise distances.

The schedule alpha(t) is zero near t = 0 (so we never disturb the
early stages of the flow, where x_t is dominated by Gaussian noise)
and ramps to a maximum at late t where the molecular geometry is
already meaningful. The default schedule is a smoothed power ramp:

    alpha(t) = alpha_max * clip((t - t_on) / (1 - t_on), 0, 1) ** k

with t_on = 0.5 and k = 2, giving a smooth quadratic onset between
t = 0.5 and t = 1.

This module is intentionally side-effect free: it never mutates the
flow backbone, it only wraps the model's velocity output with an
additive drift derived from autograd through the trained energy head.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch


def alpha_schedule(t: torch.Tensor,
                   alpha_max: float = 0.1,
                   t_on: float = 0.5,
                   power: float = 2.0) -> torch.Tensor:
    """Late-time drift schedule.

    Args:
        t: (B,) or scalar time in [0, 1].
        alpha_max: peak drift strength.
        t_on: time at which the drift becomes nonzero.
        power: smooth-onset exponent (>= 1).

    Returns:
        Tensor of the same shape as t with alpha(t).
    """
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=torch.float32)
    norm = (t - t_on).clamp(min=0.0) / max(1e-6, 1.0 - t_on)
    return alpha_max * norm.clamp(max=1.0).pow(power)


def project_translation_free(field: torch.Tensor,
                             node_batch_idx: torch.Tensor,
                             n_graphs: int) -> torch.Tensor:
    """Subtract the per-graph mean of a per-atom vector field.

    Equivalent to projecting onto the subspace orthogonal to global
    translations of each molecule.

    Args:
        field: (N_total, 3) per-atom vector field.
        node_batch_idx: (N_total,) graph index per atom.
        n_graphs: number of graphs in the batch.
    """
    if field.numel() == 0:
        return field
    sums = torch.zeros(n_graphs, field.shape[-1], device=field.device,
                       dtype=field.dtype)
    counts = torch.zeros(n_graphs, device=field.device, dtype=field.dtype)
    sums.index_add_(0, node_batch_idx, field)
    counts.index_add_(0, node_batch_idx, torch.ones_like(node_batch_idx,
                                                         dtype=field.dtype))
    means = sums / counts.clamp_min(1.0).unsqueeze(-1)
    return field - means[node_batch_idx]


def bgfm_velocity(
    v_theta: torch.Tensor,
    energy_force_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    positions: torch.Tensor,
    t: torch.Tensor,
    node_batch_idx: torch.Tensor,
    n_graphs: int,
    alpha_max: float = 0.1,
    t_on: float = 0.5,
    power: float = 2.0,
    project_se3: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Combine the flow velocity with an energy-residual drift.

    .. math::
        v_{\\rm BGFM}(x_t, t) =
            v_\\theta(x_t, t)
            + \\alpha(t)\\, P_{\\rm SE(3)}\\bigl[-\\nabla_r\\,\\Delta E_\\psi(r_t, c)\\bigr].

    Args:
        v_theta: (N_total, 3) base velocity from the FlowMol3 backbone.
        energy_force_fn: callable taking positions (N_total, 3) and
            returning (E, F) where F is -grad_r Delta E_psi. Typically
            wraps cfm_mol.energy_head.energy_and_force on the strain
            term only (b_phi(c) has zero coordinate gradient).
        positions: (N_total, 3) current x_t coordinates.
        t: (B,) time per graph in [0, 1].
        node_batch_idx: (N_total,) graph index per atom.
        n_graphs: number of graphs.
        alpha_max, t_on, power: parameters of `alpha_schedule`.
        project_se3: subtract per-graph centroid drift if True.

    Returns:
        v_bgfm: (N_total, 3) calibrated velocity.
        alpha_per_atom: (N_total, 1) effective per-atom drift weight,
            useful for logging / diagnostics.
    """
    _, F_psi = energy_force_fn(positions)  # F_psi = -grad_r Delta E_psi
    if project_se3:
        F_psi = project_translation_free(F_psi, node_batch_idx, n_graphs)
    a = alpha_schedule(t, alpha_max=alpha_max, t_on=t_on, power=power)
    a_per_atom = a[node_batch_idx].unsqueeze(-1)
    drift = a_per_atom * F_psi
    return v_theta + drift, a_per_atom


def drift_stability_loss(
    v_drift: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Quadratic penalty on the drift magnitude to keep it bounded.

    Without a regularizer the gradient of the head can grow arbitrarily
    on out-of-distribution geometries. The stability term keeps the
    drift contribution at the same order of magnitude as the base
    velocity.
    """
    return v_drift.pow(2).mean() + eps
