"""Energy-coupled vector-field utilities for BGFM-Native.

This module patches a FlowMol3-style vector_field so its coordinate
velocity is no longer only the amortized transport output v_theta.  The
coordinate channel becomes

    v_BGFM = v_theta + alpha(t) * P[- beta * grad_r DeltaU_psi(r, c)],

where DeltaU_psi is the residual strain energy head and P removes
per-molecule center-of-mass drift.  The patch is intentionally local: it
does not change FlowMol data loading or the categorical CTMC component.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MethodType

import torch

from .strain_head import residual_energy_and_force


@dataclass
class EnergyDriftConfig:
    enabled: bool = True
    alpha_max: float = 0.2
    t_on: float = 0.65
    power: float = 2.0
    # drift_kT_eV is the AUTHORITATIVE source for beta in the energy-coupled
    # drift when set; beta is derived as 1 / drift_kT_eV. The room-temperature
    # anchor is 0.02569 eV (NOT 0.025). When drift_kT_eV is None (default),
    # `beta` is used as-is so legacy callers that pass beta=... still work.
    drift_kT_eV: float | None = None
    beta: float = 1.0 / 0.02569  # ~38.93 eV^-1; overridden by drift_kT_eV when set
    force_clip: float = 10.0
    normalize_force: bool = True
    detach_force: bool = False

    def effective_beta(self) -> float:
        """Authoritative beta for the drift. Reads drift_kT_eV first."""
        kt = float(self.drift_kT_eV) if self.drift_kT_eV is not None else None
        if kt is not None and kt > 0.0:
            return 1.0 / kt
        return float(self.beta)


def alpha_schedule(t: torch.Tensor | float, cfg: EnergyDriftConfig) -> torch.Tensor:
    if not torch.is_tensor(t):
        t = torch.tensor(float(t))
    out = torch.zeros_like(t, dtype=torch.float32)
    denom = max(1.0 - float(cfg.t_on), 1e-6)
    active = t >= float(cfg.t_on)
    frac = ((t.float() - float(cfg.t_on)) / denom).clamp(0.0, 1.0)
    out = torch.where(active, float(cfg.alpha_max) * frac.pow(float(cfg.power)), out)
    return out


def center_graph_vectors(vec: torch.Tensor, node_batch_idx: torch.Tensor, n_graphs: int) -> torch.Tensor:
    mean = torch.zeros(n_graphs, vec.shape[-1], device=vec.device, dtype=vec.dtype)
    cnt = torch.zeros(n_graphs, 1, device=vec.device, dtype=vec.dtype)
    mean.index_add_(0, node_batch_idx, vec)
    cnt.index_add_(0, node_batch_idx, torch.ones_like(vec[:, :1]))
    mean = mean / cnt.clamp_min(1.0)
    return vec - mean[node_batch_idx]


def _infer_atom_charge(g) -> tuple[torch.Tensor, torch.Tensor]:
    if "a_t" in g.ndata:
        atom_types = g.ndata["a_t"].argmax(dim=-1)
    elif "a_1_true" in g.ndata:
        atom_types = g.ndata["a_1_true"].argmax(dim=-1)
    else:
        raise KeyError("graph missing a_t/a_1_true for energy drift")
    if "c_t" in g.ndata:
        charges = g.ndata["c_t"].argmax(dim=-1)
    elif "c_1_true" in g.ndata:
        charges = g.ndata["c_1_true"].argmax(dim=-1)
    else:
        charges = torch.zeros_like(atom_types)
    return atom_types, charges


def compute_energy_drift(
    head,
    positions: torch.Tensor,
    atom_types: torch.Tensor,
    charges: torch.Tensor,
    node_batch_idx: torch.Tensor,
    n_graphs: int,
    t: torch.Tensor | float,
    cfg: EnergyDriftConfig,
    create_graph: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    _, strain, _, force = residual_energy_and_force(
        head, positions, atom_types, charges, node_batch_idx, n_graphs,
        create_graph=create_graph, detach_positions=False,
    )
    force = force.clamp(-cfg.force_clip, cfg.force_clip)
    force = center_graph_vectors(force, node_batch_idx, n_graphs)
    if cfg.normalize_force:
        denom = torch.zeros(n_graphs, 1, device=force.device, dtype=force.dtype)
        per_atom_norm = force.norm(dim=-1, keepdim=True)
        denom.scatter_reduce_(0, node_batch_idx[:, None].expand_as(per_atom_norm), per_atom_norm, reduce="amax", include_self=False)
        force = force / denom[node_batch_idx].clamp_min(1.0)
    if cfg.detach_force:
        force = force.detach()
    alpha = alpha_schedule(t, cfg).to(force.device, dtype=force.dtype)
    if alpha.dim() == 0:
        alpha_node = alpha.view(1, 1).expand_as(force[:, :1])
    else:
        alpha_node = alpha[node_batch_idx].view(-1, 1)
    # Energy-coupled drift (SPEC item C): v_BGFM = v_theta + alpha(t,T) *
    # P_SE3[ -beta * grad_r DeltaE_psi ]. beta is sourced from drift_kT_eV
    # via cfg.effective_beta(); it is *separate* from the Boltzmann-bridge
    # kT (which may be tempered/annealed) and from the corrector_kT_eV.
    beta_drift = float(cfg.effective_beta())
    drift = alpha_node * beta_drift * force
    return drift, {
        "native_drift_alpha_mean": alpha.detach().float().mean(),
        "native_drift_beta": float(beta_drift),
        "native_drift_kT_eV": float(1.0 / beta_drift),
        "native_drift_norm_mean": drift.detach().norm(dim=-1).mean(),
        "native_strain_mean": strain.detach().mean(),
    }


def drift_monotonicity_loss(
    head,
    positions: torch.Tensor,
    atom_types: torch.Tensor,
    charges: torch.Tensor,
    node_batch_idx: torch.Tensor,
    n_graphs: int,
    drift: torch.Tensor,
    step_size: float = 1.0e-3,
    margin: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Hinge loss certifying the conservative drift is a descent direction.

    Eq. (eq:drift-monotonicity) in the paper:
        L_mono = mean max(0, Delta E_psi(r + epsilon * d_t, c)
                              - Delta E_psi(r, c) + m).
    The drift d_t is taken with no autograd through positions (use
    .detach()); only Delta E_psi(r + h d_t, c) carries gradient back
    into the strain head and the upstream backbone that produces drift.
    Without this term, the backbone is free to learn a v_theta that
    cancels the conservative drift contribution and pay nothing in the
    norm penalty L_drift.
    """
    if positions.numel() == 0:
        zero = torch.zeros((), device=positions.device, dtype=positions.dtype)
        return zero, {"mono_dE_mean": 0.0, "mono_violation_rate": 0.0}

    # E_psi(r) baseline. residual_energy_and_force returns (E, strain, baseline, force);
    # use the strain component since drift only depends on Delta E_psi.
    _, strain_before, _, _ = residual_energy_and_force(
        head, positions, atom_types, charges, node_batch_idx, n_graphs,
        create_graph=True, detach_positions=False,
    )
    r_step = positions + step_size * drift.detach()
    _, strain_after, _, _ = residual_energy_and_force(
        head, r_step, atom_types, charges, node_batch_idx, n_graphs,
        create_graph=True, detach_positions=False,
    )
    delta = strain_after - strain_before  # (n_graphs,)
    hinge = (delta + margin).clamp_min(0.0)
    loss = hinge.mean()
    diag = {
        "mono_dE_mean": float(delta.detach().mean().item()),
        "mono_violation_rate": float((delta.detach() > 0).float().mean().item()),
    }
    return loss, diag


def patch_energy_coupled_vector_field(model, cfg: EnergyDriftConfig) -> None:
    """Monkey-patch model.vector_field.forward to add energy residual drift."""
    if getattr(model.vector_field, "_bgfm_native_drift_patched", False):
        return
    if not hasattr(model, "_bgfm_native_strain_head"):
        raise RuntimeError("attach model._bgfm_native_strain_head before drift patch")

    vf = model.vector_field
    original_forward = vf.forward

    def forward_with_energy_drift(self_vf, g, t, *args, **kwargs):
        out = original_forward(g, t, *args, **kwargs)
        if (not cfg.enabled) or ("x" not in out) or ("x_t" not in g.ndata):
            return out
        node_batch_idx = kwargs.get("node_batch_idx", None)
        if node_batch_idx is None:
            # FlowMol utilities are available in the training environment.
            from flowmol.data_processing.utils import get_batch_idxs
            node_batch_idx, _ = get_batch_idxs(g)
        n_graphs = int(g.batch_size)
        atom_types, charges = _infer_atom_charge(g)
        positions = g.ndata["x_t"]
        drift, diag = compute_energy_drift(
            model._bgfm_native_strain_head,
            positions,
            atom_types,
            charges,
            node_batch_idx,
            n_graphs,
            t,
            cfg,
            create_graph=self_vf.training,
        )
        out = dict(out)
        out["x"] = out["x"] + drift
        # Diagnostics are attached to the module for the training hook to log.
        model._bgfm_native_last_drift_diag = diag
        return out

    vf.forward = MethodType(forward_with_energy_drift, vf)
    vf._bgfm_native_drift_patched = True
