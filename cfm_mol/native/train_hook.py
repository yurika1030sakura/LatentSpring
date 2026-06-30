"""Training integration for BGFM-Native.

This hook layers the high-upside BGFM-Native algorithm on top of the
existing FlowMol/FlowMol3 Lightning module.  It can be applied after the
standard cfm_mol constraint patch and after the existing BGFM loss patch.

The hook does three things:
  1. attach a composition-conditioned residual strain head;
  2. optionally patch the vector field so v_x includes conservative drift;
  3. wrap training_step to add native head, bridge, and corrector-in-loop losses.
"""
from __future__ import annotations

from types import MethodType
from typing import Any

import torch

from .strain_head import ResidualStrainEnergyHead, residual_head_loss
from .energy_drift import EnergyDriftConfig, patch_energy_coupled_vector_field
from .corrector_in_loop import corrector_in_loop_loss
from .boltzmann_bridge import local_boltzmann_bridge_loss


def _schedule(step_frac: float, value: float, warmup: float, ramp: float) -> float:
    if step_frac < warmup:
        return 0.0
    if step_frac < warmup + ramp:
        return value * (step_frac - warmup) / max(ramp, 1e-8)
    return value


def _graph_energy_target(g, node_batch_idx, n_graphs: int, dtype: torch.dtype) -> torch.Tensor | None:
    # Several preprocess variants exist.  Handle all common conventions.
    for key in ("energy_1_true", "energies", "energy"):
        if key not in g.ndata:
            continue
        e = g.ndata[key]
        if e.dim() > 1:
            e = e.squeeze(-1)
        if e.numel() == n_graphs:
            return e.to(dtype=dtype)
        if e.numel() == node_batch_idx.numel():
            out = torch.zeros(n_graphs, device=e.device, dtype=e.dtype)
            cnt = torch.zeros(n_graphs, device=e.device, dtype=e.dtype)
            out.index_add_(0, node_batch_idx, e)
            cnt.index_add_(0, node_batch_idx, torch.ones_like(e, dtype=e.dtype))
            return (out / cnt.clamp_min(1.0)).to(dtype=dtype)
    return None


def _atom_charge_from_endpoint(g):
    atom_types = g.ndata["a_1_true"].argmax(dim=-1) if "a_1_true" in g.ndata else g.ndata["a_t"].argmax(dim=-1)
    charges = g.ndata["c_1_true"].argmax(dim=-1) if "c_1_true" in g.ndata else g.ndata["c_t"].argmax(dim=-1)
    return atom_types, charges


def patch_bgfm_native(model, native_cfg: dict[str, Any]) -> None:
    """Install BGFM-Native modules from a config sub-block.

    Expected YAML path:
        mol_fm:
          bgfm:
            native:
              enabled: true
              ...
    """
    if not native_cfg or not bool(native_cfg.get("enabled", False)):
        return
    device = next(model.parameters()).device
    n_atom_types = int(native_cfg.get("n_atom_types", 83))
    head = ResidualStrainEnergyHead(
        n_atom_types=n_atom_types,
        n_charge_classes=int(native_cfg.get("n_charge_classes", 6)),
        hidden_dim=int(native_cfg.get("head_hidden_dim", 192)),
        n_rbf=int(native_cfg.get("head_n_rbf", 32)),
        cutoff=float(native_cfg.get("head_cutoff", 5.0)),
        n_layers=int(native_cfg.get("head_n_layers", 3)),
        baseline_hidden_dim=int(native_cfg.get("baseline_hidden_dim", 128)),
    ).to(device)
    model._bgfm_native_strain_head = head

    drift_cfg = EnergyDriftConfig(
        enabled=bool(native_cfg.get("energy_drift_enabled", True)),
        alpha_max=float(native_cfg.get("drift_alpha_max", 0.2)),
        t_on=float(native_cfg.get("drift_t_on", 0.65)),
        power=float(native_cfg.get("drift_power", 2.0)),
        beta=float(native_cfg.get("beta", 1.0 / float(native_cfg.get("kT", 0.025)))),
        force_clip=float(native_cfg.get("drift_force_clip", 10.0)),
        normalize_force=bool(native_cfg.get("drift_normalize_force", True)),
        detach_force=bool(native_cfg.get("drift_detach_force", False)),
    )
    if bool(native_cfg.get("patch_vector_field", True)):
        patch_energy_coupled_vector_field(model, drift_cfg)

    original_training_step = model.training_step

    def native_training_step(self, g, batch_idx):
        total = original_training_step(g, batch_idx)
        # Approximate training fraction for warmups.
        max_steps = getattr(getattr(self, "trainer", None), "estimated_stepping_batches", None)
        if max_steps is None or max_steps <= 0:
            frac = 1.0
        else:
            frac = min(float(getattr(self, "global_step", 0)) / float(max_steps), 1.0)
        warmup = float(native_cfg.get("warmup_frac", 0.005))
        ramp = float(native_cfg.get("ramp_frac", 0.02))
        from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask
        node_batch_idx, _ = get_batch_idxs(g)
        n_graphs = int(g.batch_size)
        atom_types, charges = _atom_charge_from_endpoint(g)
        r = g.ndata["x_1_true"]
        # Native head calibration.
        l_head_w = _schedule(frac, float(native_cfg.get("lambda_head", 1.0)), warmup, ramp)
        if l_head_w > 0 and "force_1_true" in g.ndata:
            e_tgt = _graph_energy_target(g, node_batch_idx, n_graphs, r.dtype)
            if e_tgt is not None:
                Lh, dh = residual_head_loss(
                    self._bgfm_native_strain_head, r, atom_types, charges,
                    node_batch_idx, n_graphs, e_tgt, g.ndata["force_1_true"],
                    lambda_force=float(native_cfg.get("lambda_head_force", 0.1)),
                )
                if torch.isfinite(Lh):
                    total = total + l_head_w * Lh
                    self.log("train_native_L_head", Lh.detach(), on_step=True, prog_bar=True)
                    for k, v in dh.items():
                        self.log("train_" + k, v, on_step=True)
        # Corrector-in-loop safety loss.
        l_corr_w = _schedule(frac, float(native_cfg.get("lambda_corrector", 0.05)), warmup, ramp)
        if l_corr_w > 0:
            Lc, dc = corrector_in_loop_loss(
                self._bgfm_native_strain_head, r, atom_types, charges,
                node_batch_idx, n_graphs,
                beta=float(native_cfg.get("beta", 1.0 / float(native_cfg.get("kT", 0.025)))),
                j_max=int(native_cfg.get("corrector_train_jmax", 3)),
                eta_init=float(native_cfg.get("corrector_eta_init", 1.0e-3)),
                eta_final=float(native_cfg.get("corrector_eta_final", 1.0e-4)),
                delta_max=float(native_cfg.get("corrector_delta_max", 0.25)),
                stochastic=bool(native_cfg.get("corrector_train_noise", True)),
            )
            if torch.isfinite(Lc):
                total = total + l_corr_w * Lc
                self.log("train_native_L_corrector", Lc.detach(), on_step=True)
                for k, v in dc.items():
                    self.log("train_" + k, v, on_step=True)
        # Local Boltzmann bridge on perturbation loader, if available.
        l_bridge_w = _schedule(frac, float(native_cfg.get("lambda_bridge", 0.0)), warmup, ramp)
        loader = getattr(self, "_bgfm_perturbation_loader", None)
        if l_bridge_w > 0 and loader is not None:
            try:
                from cfm_mol.bgfm_density import log_density_via_flow
                g_pert, energies, parent_id, nbi_pert, uem = loader.next_batch()
                logp = log_density_via_flow(
                    self, g_pert, nbi_pert, uem,
                    n_ode_steps=int(native_cfg.get("bridge_n_ode_steps", 4)),
                    n_hutchinson=int(native_cfg.get("bridge_n_hutchinson", 1)),
                    prior_std=float(native_cfg.get("prior_std", 1.0)),
                    for_training=True,
                )
                Lb, db = local_boltzmann_bridge_loss(
                    logp, energies, parent_id,
                    kT=float(native_cfg.get("kT", 0.025)),
                    energy_clip=float(native_cfg.get("bridge_energy_clip", 200.0)),
                )
                if torch.isfinite(Lb):
                    total = total + l_bridge_w * Lb
                    self.log("train_native_L_bridge", Lb.detach(), on_step=True, prog_bar=True)
                    for k, v in db.items():
                        self.log("train_" + k, v, on_step=True)
            except RuntimeError:
                self.log("train_native_bridge_runtime_skip", 1.0, on_step=True)
        # Drift diagnostics produced by patched vector field.
        diag = getattr(self, "_bgfm_native_last_drift_diag", None)
        if isinstance(diag, dict):
            for k, v in diag.items():
                if torch.is_tensor(v):
                    self.log("train_" + k, v.detach(), on_step=True)
        self.log("train_total_bgfm_native", total.detach(), on_step=True, prog_bar=True)
        return total

    model.training_step = MethodType(native_training_step, model)
    print("[bgfm-native] installed: energy drift=%s lambda_head=%g lambda_bridge=%g" % (
        bool(native_cfg.get("energy_drift_enabled", True)),
        float(native_cfg.get("lambda_head", 1.0)),
        float(native_cfg.get("lambda_bridge", 0.0)),
    ))
