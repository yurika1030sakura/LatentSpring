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
    beta: float | None = None,
    j_max: int = 3,
    eta_init: float = 1.0e-3,
    eta_final: float = 1.0e-4,
    delta_max: float = 0.25,
    energy_margin: float = 0.0,
    stochastic: bool | None = None,  # legacy alias for noise_mode
    *,
    noise_mode: str | None = None,  # SPEC item D
    corrector_kT_eV: float | None = None,
    corrector_noise_mode: str | None = None,  # alias accepted from train_hook configs
) -> tuple[torch.Tensor, dict[str, float]]:
    """Unroll 0..j_max learned-energy steps and penalize unstable moves.

    The corrector is trained to be safe, not to create a new exact MCMC
    sampler.  We penalize excessive per-atom displacement and increases in
    the residual strain energy.

    Two reported variants (SPEC item D):
        noise_mode = 'deterministic' -> epsilon = 0 (BGFM-Native-det)
        noise_mode = 'physical_300K' -> epsilon ~ sqrt(2*eta/beta_physical) *
            N(0, I) with beta_physical = 1 / 0.02569 eV^-1

    corrector_kT_eV overrides `beta` for the Langevin noise term ONLY. The
    legacy `stochastic` flag is still honoured: True -> 'physical_300K',
    False -> 'deterministic'. If neither `stochastic` nor `noise_mode` is
    set, the default is 'physical_300K' to preserve previous behaviour.
    `corrector_noise_mode` is an accepted alias for `noise_mode` so that
    train_hook configs can forward the YAML key name verbatim.
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
    # Resolve noise mode. Precedence: corrector_noise_mode > noise_mode >
    # stochastic. If NONE are set, default to 'physical_300K' so the legacy
    # behaviour (noise on by default) is preserved.
    resolved_mode = corrector_noise_mode if corrector_noise_mode is not None else noise_mode
    if resolved_mode is None:
        if stochastic is True:
            resolved_mode = "physical_300K"
        elif stochastic is False:
            resolved_mode = "deterministic"
        else:
            resolved_mode = "physical_300K"
    else:
        # explicit mode wins; legacy `stochastic` reconciled (False forces det)
        if stochastic is False:
            resolved_mode = "deterministic"
    # beta used for the Langevin noise term ONLY. If corrector_kT_eV is set,
    # it takes precedence over `beta`; otherwise default to the physical
    # room-temperature anchor (0.02569 eV) so the diagnostic is mechanistic.
    if corrector_kT_eV is not None and corrector_kT_eV > 0.0:
        beta_noise = 1.0 / float(corrector_kT_eV)
    elif beta is not None and beta > 0.0:
        beta_noise = float(beta)
    else:
        beta_noise = 1.0 / 0.02569  # physical 300 K anchor
    for j in range(J):
        _, _, _, force = residual_energy_and_force(
            head, r, atom_types, charges, node_batch_idx, n_graphs,
            create_graph=True, detach_positions=False,
        )
        force = center_graph_vectors(force, node_batch_idx, n_graphs)
        eta = _eta_schedule(j, J, eta_init, eta_final)
        if resolved_mode == "physical_300K":
            noise = torch.randn_like(r) * (2.0 * eta / max(beta_noise, 1e-8)) ** 0.5
        else:  # deterministic
            noise = 0.0
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
        "native_corrector_noise_mode": 1.0 if resolved_mode == "physical_300K" else 0.0,
        "native_corrector_kT_eV": float(1.0 / beta_noise),
        "native_stab_energy_inc": float(energy_inc.detach().item()),
        "native_stab_disp": float(disp_loss.detach().item()),
    }
