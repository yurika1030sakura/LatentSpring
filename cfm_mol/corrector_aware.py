"""Corrector-aware training utilities for BGFM.

Standard BGFM trains the flow proposal and the energy head independently,
then runs the Langevin corrector only at sampling time. A reviewer can
fairly object that this is post-processing. Corrector-aware training
closes the gap: during the training step we randomly unroll a short
corrector chain on top of the data endpoint, and we add a small stability
penalty so the corrector does not push the geometry away from the data
manifold.

Specifically, on each training step we sample

    J ~ Uniform{0, 1, ..., J_max}

and update the data positions as

    r' = LangevinUnroll(J; r_1, c; eta, beta, P_SE3)

then add a soft penalty:

    L_stab = max(0, ||r' - r_1||^2 - delta_max^2)
           + max(0, E_psi(r', c) - E_psi(r_1, c)).

The first term keeps the corrector from drifting outside a small ball
around the data endpoint; the second prevents the corrector from
moving samples *uphill* on the strain landscape. Both terms are zero
when the corrector is well-behaved.

This module is intentionally cheap: J_max defaults to 3 and the unroll
runs with no autograd graph through positions, so the only autograd path
is the energy head. Total training overhead is roughly J_max additional
energy-head forward + backward passes per batch.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch


def _sample_J(J_max: int, device: torch.device) -> int:
    if J_max <= 0:
        return 0
    return int(torch.randint(0, J_max + 1, (1,), device=device).item())


def unrolled_corrector(
    energy_force_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    positions: torch.Tensor,
    J: int,
    eta: float,
    beta: float,
    node_batch_idx: torch.Tensor,
    n_graphs: int,
    recenter: bool = True,
) -> torch.Tensor:
    """Run J short Langevin steps from `positions` under the energy head.

    Args:
        energy_force_fn: callable returning (E, F = -grad E).
        positions: (N_total, 3) initial coordinates.
        J: number of refinement steps. Zero -> returns input unchanged.
        eta: step size in A^2 / eV.
        beta: inverse temperature in eV^{-1}.
        node_batch_idx: (N_total,) graph index per atom.
        n_graphs: number of graphs.
        recenter: subtract per-graph centroid after each step.

    Returns:
        (N_total, 3) refined coordinates. No autograd graph through r.
    """
    if J <= 0:
        return positions
    from cfm_mol.energy_residual_drift import project_translation_free
    r = positions.detach()
    sigma = (2.0 * eta / beta) ** 0.5
    for _ in range(J):
        r = r.detach().requires_grad_(True)
        _, F = energy_force_fn(r)
        F = F.detach()
        if recenter:
            F = project_translation_free(F, node_batch_idx, n_graphs)
        noise = torch.randn_like(r) * sigma
        r = (r.detach() + eta * F + noise)
        if recenter:
            r = project_translation_free(r, node_batch_idx, n_graphs)
    return r.detach()


def corrector_stability_loss(
    energy_force_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    r_data: torch.Tensor,
    r_refined: torch.Tensor,
    delta_max: float = 0.5,
) -> Tuple[torch.Tensor, dict]:
    """Stability penalty for a corrector-aware training step.

    L_stab = mean_atom max(0, ||r' - r_1||^2 - delta_max^2)
           + mean_graph max(0, E_psi(r') - E_psi(r_1)).

    Args:
        energy_force_fn: callable (positions -> (E, F)).
        r_data: (N_total, 3) endpoint geometry from the data batch.
        r_refined: (N_total, 3) post-corrector geometry.
        delta_max: tolerated per-atom displacement in Angstrom.

    Returns:
        (loss, diag) where diag has 'corrector_displacement_mean' and
        'corrector_energy_delta_mean'.
    """
    disp_sq = (r_refined - r_data).pow(2).sum(dim=-1)
    over = (disp_sq - delta_max ** 2).clamp_min(0.0).mean()
    E_data, _ = energy_force_fn(r_data)
    E_refined, _ = energy_force_fn(r_refined)
    energy_increase = (E_refined - E_data).clamp_min(0.0).mean()
    loss = over + energy_increase
    diag = {
        "corrector_displacement_mean": float(disp_sq.detach().sqrt().mean().item()),
        "corrector_energy_delta_mean": float((E_refined - E_data).detach().mean().item()),
    }
    return loss, diag
