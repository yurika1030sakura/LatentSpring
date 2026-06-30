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
    beta: float = 40.0  # 1 / 0.025 eV by default
    force_clip: float = 10.0
    normalize_force: bool = True
    detach_force: bool = False


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
    drift = alpha_node * float(cfg.beta) * force
    return drift, {
        "native_drift_alpha_mean": alpha.detach().float().mean(),
        "native_drift_norm_mean": drift.detach().norm(dim=-1).mean(),
        "native_strain_mean": strain.detach().mean(),
    }


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
