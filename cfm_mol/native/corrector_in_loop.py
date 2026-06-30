"""Corrector-in-the-loop training loss for BGFM-Native."""
from __future__ import annotations

import random
import torch

from .strain_head import residual_energy_and_force
from .energy_drift import center_graph_vectors


def _eta_schedule(j: int, n_steps: int, eta_init: float, eta_final: float) -> float:
    if n_steps <= 1:
        return eta_init
    frac = j / max(n_steps - 1, 1)
    return float(eta_init * (eta_final / eta_init) ** frac)


def corrector_in_loop_loss(
    head,
    positions: torch.Tensor,
    atom_types: torch.Tensor,
    charges: torch.Tensor,
    node_batch_idx: torch.Tensor,
    n_graphs: int,
    beta: float,
    j_max: int = 3,
    eta_init: float = 1.0e-3,
    eta_final: float = 1.0e-4,
    delta_max: float = 0.25,
    energy_margin: float = 0.0,
    stochastic: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Unroll 0..j_max learned-energy steps and penalize unstable moves.

    The corrector is trained to be safe, not to create a new exact MCMC
    sampler.  We penalize excessive per-atom displacement and increases in
    the residual strain energy.
    """
    if j_max <= 0:
        z = positions.sum() * 0.0
        return z, {"native_corrector_steps": 0.0, "native_stab_energy_inc": 0.0}
    J = random.randint(0, int(j_max))
    if J == 0:
        z = positions.sum() * 0.0
        return z, {"native_corrector_steps": 0.0, "native_stab_energy_inc": 0.0}
    r0 = positions.detach()
    r = r0.clone().requires_grad_(True)
    with torch.no_grad():
        # Baseline strain only -- no force is needed here, so call
        # forward_parts directly. residual_energy_and_force runs
        # torch.autograd.grad internally which fails under no_grad.
        _, strain0, _ = head.forward_parts(
            r0, atom_types, charges, node_batch_idx, n_graphs,
        )
    for j in range(J):
        _, _, _, force = residual_energy_and_force(
            head, r, atom_types, charges, node_batch_idx, n_graphs,
            create_graph=True, detach_positions=False,
        )
        force = center_graph_vectors(force, node_batch_idx, n_graphs)
        eta = _eta_schedule(j, J, eta_init, eta_final)
        noise = 0.0
        if stochastic:
            noise = torch.randn_like(r) * (2.0 * eta / max(beta, 1e-8)) ** 0.5
        r = (r + eta * force + noise).requires_grad_(True)
        # recenter after every step
        disp = r - r0
        disp = center_graph_vectors(disp, node_batch_idx, n_graphs)
        r = (r0 + disp).requires_grad_(True)
    _, strainJ, _, _ = residual_energy_and_force(
        head, r, atom_types, charges, node_batch_idx, n_graphs,
        create_graph=True, detach_positions=False,
    )
    energy_inc = torch.relu(strainJ - strain0 + float(energy_margin)).mean()
    disp_norm2 = (r - r0).pow(2).sum(dim=-1)
    disp_loss = torch.relu(disp_norm2 - float(delta_max) ** 2).mean()
    loss = energy_inc + disp_loss
    return loss, {
        "native_corrector_steps": float(J),
        "native_stab_energy_inc": float(energy_inc.detach().item()),
        "native_stab_disp": float(disp_loss.detach().item()),
    }
