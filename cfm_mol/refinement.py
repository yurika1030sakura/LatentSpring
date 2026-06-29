"""Learned-energy Langevin corrector for BGFM.

After the flow proposal returns a candidate molecule (c, r), BGFM
runs J short Langevin steps under the calibrated energy head E_psi:

    r^{j+1} = r^j - eta_j * grad_r E_psi(r^j, c) + sigma_j * xi^j

with sigma_j = sqrt(2 * eta_j / beta) so that the Langevin chain
targets exp(-beta * E_psi). Optional Metropolis correction can be
enabled. The discrete components c are held fixed throughout the
refinement; only coordinates are updated.

Honest compute accounting
-------------------------

This module returns an `Accounting` record alongside the refined
sample. Reported fields:

  flow_nfe:           ODE steps used by the flow proposal (filled by
                      the caller).
  energy_head_nfe:    total forward passes of E_psi (= J + 1 for the
                      initial energy on the proposal).
  oracle_nfe:         number of external OMol25 / xTB calls made
                      during refinement (zero by default; positive
                      only when --use_external_oracle is set, used
                      for diagnostic ablations).
  wall_clock_s:       per-sample wall-clock seconds.
  rejected:           number of Metropolis rejections (zero if
                      Metropolis is disabled).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple
import time

import torch


@dataclass
class Accounting:
    flow_nfe: int = 0
    energy_head_nfe: int = 0
    oracle_nfe: int = 0
    wall_clock_s: float = 0.0
    rejected: int = 0
    n_samples: int = 0

    def to_dict(self) -> dict:
        return {
            "flow_nfe": self.flow_nfe,
            "energy_head_nfe": self.energy_head_nfe,
            "oracle_nfe": self.oracle_nfe,
            "wall_clock_s": self.wall_clock_s,
            "rejected": self.rejected,
            "n_samples": self.n_samples,
        }


def _step_schedule(j: int, J: int, eta_init: float, eta_final: float) -> float:
    """Inverse-time step size schedule.

    eta_j = eta_init * (eta_final / eta_init) ** (j / max(J - 1, 1))
    """
    if J <= 1:
        return eta_init
    frac = j / (J - 1)
    return float(eta_init * (eta_final / eta_init) ** frac)


def langevin_corrector(
    energy_force_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    positions: torch.Tensor,
    n_steps: int,
    beta: float,
    eta_init: float = 1.0e-3,
    eta_final: float = 1.0e-4,
    metropolis: bool = False,
    accounting: Optional[Accounting] = None,
    recenter: bool = True,
) -> Tuple[torch.Tensor, Accounting]:
    """Run J Langevin steps targeting exp(-beta * E_psi).

    Args:
        energy_force_fn: callable taking positions (N_total, 3) and
            returning (E, F) where E is (n_graphs,) and F is
            (N_total, 3). Typically wraps cfm_mol.energy_head.energy_and_force.
        positions: (N_total, 3) initial coordinates from the flow
            proposal. Centered at the per-graph centroid; this routine
            does not re-center.
        n_steps: J in the method description; recommended 20-100.
        beta: inverse temperature 1 / kT in eV^{-1}.
        eta_init, eta_final: per-step coordinate step size at start
            and end of the schedule (Angstrom^2 / eV).
        metropolis: if True, accept each proposed step with
            probability min(1, exp(-beta * dE)). Requires an
            additional energy evaluation per step.
        accounting: an Accounting object to update in place; created
            fresh if not provided.
        recenter: if True, subtract the centroid after every proposal.
            This keeps single-molecule samples in the same translation
            gauge as the FlowMol training data. For multi-molecule batched
            refinement, call this routine one molecule at a time or extend
            it with node_batch_idx-aware recentering.

    Returns:
        positions: (N_total, 3) refined coordinates.
        accounting: updated Accounting record.
    """
    if accounting is None:
        accounting = Accounting()
    t0 = time.time()

    r = positions.detach().clone()
    # Initial energy/force evaluation used for the first step (and as
    # the reference if Metropolis is on).
    r.requires_grad_(True)
    E_prev, F_prev = energy_force_fn(r)
    accounting.energy_head_nfe += 1

    for j in range(n_steps):
        eta = _step_schedule(j, n_steps, eta_init, eta_final)
        sigma = (2.0 * eta / beta) ** 0.5
        noise = torch.randn_like(r) * sigma
        r_prop = r.detach() + eta * F_prev.detach() + noise
        if recenter:
            r_prop = r_prop - r_prop.mean(dim=0, keepdim=True)
        r_prop = r_prop.requires_grad_(True)

        if metropolis:
            E_prop, F_prop = energy_force_fn(r_prop)
            accounting.energy_head_nfe += 1
            log_ratio = -beta * (E_prop - E_prev)
            # Per-graph accept/reject; broadcast back to atoms.
            accept = torch.rand_like(log_ratio).log() < log_ratio
            # Build a per-atom accept mask: this requires knowing the
            # per-graph node membership; for simplicity the caller is
            # responsible for providing positions arranged so that the
            # mean force per graph is meaningful. Fallback: accept all
            # if accept shape doesn't broadcast.
            if accept.numel() == 1:
                if not bool(accept.item()):
                    r_prop = r.detach().requires_grad_(True)
                    E_prop, F_prop = E_prev, F_prev
                    accounting.rejected += 1
            else:
                raise NotImplementedError(
                    "Metropolis correction for batched multi-graph refinement "
                    "requires node_batch_idx-aware accept/reject. Call "
                    "langevin_corrector one molecule at a time or disable "
                    "metropolis.")
            r = r_prop
            E_prev, F_prev = E_prop, F_prop
        else:
            r = r_prop
            E_new, F_new = energy_force_fn(r)
            accounting.energy_head_nfe += 1
            E_prev, F_prev = E_new, F_new

    accounting.wall_clock_s += time.time() - t0
    return r.detach(), accounting
