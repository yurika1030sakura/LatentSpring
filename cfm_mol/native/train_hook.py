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
from .energy_drift import (
    EnergyDriftConfig,
    compute_energy_drift,
    drift_monotonicity_loss,
    patch_energy_coupled_vector_field,
)
from .corrector_in_loop import corrector_in_loop_loss
from .boltzmann_bridge import local_boltzmann_bridge_loss


PHYSICAL_KT_EV = 0.02569  # k_B * 298.15 K; reviewers' anchor; do not change
PHYSICAL_T_K = 298.15


def _resolve_temperature(native_cfg: dict) -> dict:
    """Return a normalized temperature sub-config.

    The `temperature:` YAML block under mol_fm.bgfm.native is AUTHORITATIVE.
    Legacy keys (kT, beta) at native_cfg root are still honored when
    `temperature:` is absent, for backward compatibility, but they are
    deprecated and will be removed once all configs migrate.
    """
    t = dict(native_cfg.get("temperature", {}) or {})
    legacy_kT = native_cfg.get("kT", None)
    if "drift_kT_eV" not in t:
        t["drift_kT_eV"] = float(legacy_kT) if legacy_kT is not None else PHYSICAL_KT_EV
    if "bridge_kT_final_eV" not in t:
        t["bridge_kT_final_eV"] = float(legacy_kT) if legacy_kT is not None else PHYSICAL_KT_EV
    t.setdefault("physical_kT_eV", PHYSICAL_KT_EV)
    t.setdefault("physical_T_K", PHYSICAL_T_K)
    t.setdefault("bridge_kT_mode", "annealed")
    t.setdefault("bridge_kT_start_eV", 0.25)
    t.setdefault("force_kT_eV", PHYSICAL_KT_EV)
    t.setdefault("corrector_kT_eV", PHYSICAL_KT_EV)
    t.setdefault("corrector_noise_mode", "deterministic")
    if abs(float(t["physical_kT_eV"]) - PHYSICAL_KT_EV) > 1e-6:
        raise ValueError(
            f"physical_kT_eV must be exactly {PHYSICAL_KT_EV} (room temperature anchor); "
            f"got {t['physical_kT_eV']}"
        )
    return t


def _compute_bridge_kT(step_frac: float, temp_cfg: dict) -> float:
    """Log-linear anneal of bridge kT from start_eV to final_eV across training."""
    import math
    start = float(temp_cfg["bridge_kT_start_eV"])
    final = float(temp_cfg["bridge_kT_final_eV"])
    if str(temp_cfg.get("bridge_kT_mode", "annealed")) == "fixed_tempered":
        return start
    f = max(min(float(step_frac), 1.0), 0.0)
    return math.exp(math.log(start) + f * (math.log(final) - math.log(start)))


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

    temp_cfg = _resolve_temperature(native_cfg)
    drift_kT_eV = float(temp_cfg["drift_kT_eV"])
    drift_cfg = EnergyDriftConfig(
        enabled=bool(native_cfg.get("energy_drift_enabled", True)),
        alpha_max=float(native_cfg.get("drift_alpha_max", 0.2)),
        t_on=float(native_cfg.get("drift_t_on", 0.65)),
        power=float(native_cfg.get("drift_power", 2.0)),
        drift_kT_eV=drift_kT_eV,
        beta=1.0 / drift_kT_eV,
        force_clip=float(native_cfg.get("drift_force_clip", 10.0)),
        normalize_force=bool(native_cfg.get("drift_normalize_force", True)),
        detach_force=bool(native_cfg.get("drift_detach_force", False)),
    )
    model._bgfm_native_temperature_cfg = temp_cfg
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
                corrector_kT_eV=float(temp_cfg["corrector_kT_eV"]),
                noise_mode=str(temp_cfg["corrector_noise_mode"]),
                j_max=int(native_cfg.get("corrector_train_jmax", 3)),
                eta_init=float(native_cfg.get("corrector_eta_init", 1.0e-3)),
                eta_final=float(native_cfg.get("corrector_eta_final", 1.0e-4)),
                delta_max=float(native_cfg.get("corrector_delta_max", 0.25)),
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
                kT_sched = {
                    "mode": str(temp_cfg["bridge_kT_mode"]),
                    "start_eV": float(temp_cfg["bridge_kT_start_eV"]),
                    "final_eV": float(temp_cfg["bridge_kT_final_eV"]),
                }
                # Optional joint discrete-continuous density (paper
                # Eq. eq:joint-density). Adds log p_theta(c) to the
                # per-virtual-mol log-density. For K perturbations of a
                # single parent the discrete component is constant, so
                # it cancels inside both the variance and the per-parent
                # softmax of the bridge; we add it here for symmetry
                # with the variance loss and so that the diagnostic
                # logging picks up the discrete contribution.
                if bool(native_cfg.get("joint_density_enabled", False)):
                    try:
                        from cfm_mol.joint_density import joint_log_prob
                        # The discrete CTMC log-probability of (a, c) at
                        # the data endpoint is constant within each
                        # parent's cloud; we use the per-virtual-mol
                        # log_n_atoms surrogate when no head is wired,
                        # so the addition is well-defined but cancels.
                        n_per = torch.bincount(
                            parent_id, minlength=int(parent_id.max().item()) + 1
                        )[parent_id].to(logp.dtype)
                        discrete_logp = (-torch.log(n_per.clamp_min(1.0)))
                        logp = joint_log_prob(logp, discrete_logp)
                        self.log("train_native_joint_density_active", 1.0,
                                 on_step=True)
                    except Exception:
                        self.log("train_native_joint_density_skip", 1.0,
                                 on_step=True)
                Lb, db = local_boltzmann_bridge_loss(
                    logp, energies, parent_id,
                    kT=float(temp_cfg["bridge_kT_final_eV"]),  # static fallback
                    energy_clip=float(native_cfg.get("bridge_energy_clip", 200.0)),
                    kT_schedule=kT_sched,
                    step_frac=float(frac),
                    symmetric=bool(native_cfg.get("bridge_symmetric", False)),
                )
                self.log(
                    "train_native_bridge_kT_eV_eff",
                    _compute_bridge_kT(frac, temp_cfg),
                    on_step=True,
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

        # Drift-monotonicity hinge (lambda_mono * L_mono).
        # Certifies that the conservative drift component is a descent
        # direction on the learned strain (paper Eq. eq:drift-monotonicity).
        # Sampled at t in [t_on, 1] where alpha(t) is nonzero. Cheap: two
        # extra strain-head evaluations per batch.
        l_mono_w = _schedule(
            frac, float(native_cfg.get("lambda_mono", 0.0)), warmup, ramp
        )
        if l_mono_w > 0 and drift_cfg.enabled:
            try:
                t_lo = float(native_cfg.get("drift_t_on", 0.65))
                t_mono = float(native_cfg.get("mono_t_eval", min(0.95, max(t_lo, 0.85))))
                drift_t, _ = compute_energy_drift(
                    self._bgfm_native_strain_head,
                    r, atom_types, charges, node_batch_idx, n_graphs,
                    t=torch.full(
                        (n_graphs,), t_mono, device=r.device, dtype=r.dtype
                    ),
                    cfg=drift_cfg,
                    create_graph=False,
                )
                Lm, dm = drift_monotonicity_loss(
                    self._bgfm_native_strain_head,
                    r, atom_types, charges, node_batch_idx, n_graphs,
                    drift=drift_t,
                    step_size=float(native_cfg.get("mono_step_size", 1.0e-3)),
                    margin=float(native_cfg.get("mono_margin", 0.0)),
                )
                if torch.isfinite(Lm):
                    total = total + l_mono_w * Lm
                    self.log("train_native_L_mono", Lm.detach(),
                             on_step=True, prog_bar=True)
                    self.log("train_lambda_mono", l_mono_w, on_step=True)
                    for k, v in dm.items():
                        self.log("train_" + k, v, on_step=True)
            except RuntimeError:
                self.log("train_native_mono_runtime_skip", 1.0, on_step=True)

        self.log("train_total_bgfm_native", total.detach(), on_step=True, prog_bar=True)
        return total

    model.training_step = MethodType(native_training_step, model)
    print("[bgfm-native] installed: energy drift=%s lambda_head=%g lambda_bridge=%g" % (
        bool(native_cfg.get("energy_drift_enabled", True)),
        float(native_cfg.get("lambda_head", 1.0)),
        float(native_cfg.get("lambda_bridge", 0.0)),
    ))
    print(
        "[bgfm-native] temperature: physical_kT=%g eV (T=%.2f K), drift_kT=%g eV, "
        "bridge_kT %s [%g -> %g] eV, corrector_kT=%g eV mode=%s"
        % (
            temp_cfg["physical_kT_eV"], temp_cfg["physical_T_K"],
            temp_cfg["drift_kT_eV"],
            temp_cfg["bridge_kT_mode"],
            temp_cfg["bridge_kT_start_eV"], temp_cfg["bridge_kT_final_eV"],
            temp_cfg["corrector_kT_eV"], temp_cfg["corrector_noise_mode"],
        )
    )
