"""BGFM training-step patcher for FlowMol3.

Applies a lightweight monkey-patch to model.training_step that:
  (1) runs the original FM training step (returns fm_total_loss);
  (2) does AUXILIARY forward pass(es) on near-end conditional-path states
      (x_t = (1-alpha_t)x_0 + alpha_t x_1_true, t in t_eval_values);
  (3) derives the implied score via the FM->score closed-form;
  (4) computes force consistency loss vs DFT forces stored on the graph;
  (5) adds (lambda_1 * L_force) to fm_total with warmup/ramp schedule.

Keeps FlowMol3's vector_field unchanged. Re-uses MoleculeDataset's
`force_1_true` node feature (populated only on BGFM-preprocessed datasets).

For Paper 1 v1: force-only BGFM (L_energy deferred to Week 4 grid).
"""
from __future__ import annotations

import torch
import dgl

from cfm_mol.bgfm_loss import (
    score_from_fm_velocity,
    force_loss,
    score_force_cosine,
    force_direction_loss,
)


def _bgfm_schedule(epoch_frac: float, full_lambda: float,
                   warmup_frac: float, ramp_frac: float) -> float:
    if epoch_frac < warmup_frac:
        return 0.0
    if epoch_frac < warmup_frac + ramp_frac:
        return full_lambda * (epoch_frac - warmup_frac) / ramp_frac
    return full_lambda


def _parse_t_eval_values(bgfm_config: dict) -> list[float]:
    raw = bgfm_config.get('t_eval_values', bgfm_config.get('t_eval', 0.95))
    if isinstance(raw, (int, float)):
        values = [float(raw)]
    elif isinstance(raw, (list, tuple)):
        values = [float(v) for v in raw]
    else:
        raise TypeError("bgfm.t_eval_values must be a float or list of floats")
    if not values:
        raise ValueError("bgfm.t_eval_values must contain at least one time")
    for value in values:
        if not (0.0 < value < 1.0):
            raise ValueError(f"BGFM t_eval must be in (0, 1), got {value}")
    return values


def _time_tag(t_eval: float) -> str:
    return str(t_eval).replace('.', '_')


def _build_endpoint_aux_graph(g: dgl.DGLGraph) -> dgl.DGLGraph:
    """Legacy endpoint probe: set x_t = x_1_true, a_t = a_1_true, etc."""
    g_aux = g.clone()
    # Position channel
    g_aux.ndata['x_t'] = g_aux.ndata['x_1_true'].detach().clone().requires_grad_(False)
    # Discrete channels: use the one-hot labels as "current interpolant"
    g_aux.ndata['a_t'] = g_aux.ndata['a_1_true'].detach().clone()
    g_aux.ndata['c_t'] = g_aux.ndata['c_1_true'].detach().clone()
    g_aux.edata['e_t'] = g_aux.edata['e_1_true'].detach().clone()
    return g_aux


def _condition_discrete_on_endpoint(g_aux: dgl.DGLGraph) -> None:
    """Use endpoint labels for the force probe while leaving x_t on its path."""
    if 'a_1_true' in g_aux.ndata:
        g_aux.ndata['a_t'] = g_aux.ndata['a_1_true'].detach().clone()
    if 'c_1_true' in g_aux.ndata:
        g_aux.ndata['c_t'] = g_aux.ndata['c_1_true'].detach().clone()
    if 'e_1_true' in g_aux.edata:
        g_aux.edata['e_t'] = g_aux.edata['e_1_true'].detach().clone()


def _build_path_aux_graph(
    g: dgl.DGLGraph,
    vector_field,
    t: torch.Tensor,
    node_batch_idx: torch.Tensor,
    edge_batch_idx: torch.Tensor,
    upper_edge_mask: torch.Tensor,
) -> dgl.DGLGraph:
    """Clone graph and place the position channel on the true conditional path.

    The FM->score identity is valid for x_t sampled from the conditional path.
    We keep atom/charge/bond channels fixed to the endpoint labels so the force
    target is not paired with randomly masked molecular identities.
    """
    g_aux = g.clone()
    g_aux = vector_field.sample_conditional_path(
        g_aux,
        t,
        node_batch_idx=node_batch_idx,
        edge_batch_idx=edge_batch_idx,
        upper_edge_mask=upper_edge_mask,
    )
    _condition_discrete_on_endpoint(g_aux)
    return g_aux


def _force_targets(forces: torch.Tensor, mode: str) -> torch.Tensor:
    """Return the force target used by the BGFM auxiliary loss."""
    if mode == 'true':
        return forces
    if mode == 'shuffle_atoms':
        if forces.shape[0] <= 1:
            return forces
        return forces[torch.randperm(forces.shape[0], device=forces.device)]
    raise ValueError(f"unknown bgfm.force_target_mode: {mode}")


def patch_flowmol_bgfm(model, bgfm_config: dict) -> None:
    """Install BGFM auxiliary-loss hook onto model.training_step.

    Expected keys in bgfm_config:
        enabled: bool
        lambda_1: float
        lambda_2: float (energy-consistency term, active when > 0)
        kT: float                 (fixed temperature in eV, used if kT_conditioning off)
        kT_conditioning: bool     (Phase B: sample kT per step, condition model on kT)
        kT_min, kT_max: float     (range for sampling if kT_conditioning enabled)
        force_loss_type: mse | cosine | norm_mse
        force_target_mode: true | shuffle_atoms
        probe_mode: path | endpoint
        t_eval: float in (0, 1), or t_eval_values: list[float]
        warmup_frac: float
        ramp_frac: float
    """
    if not bgfm_config.get('enabled', False):
        print("[bgfm] BGFM disabled in config; skipping patch.")
        return

    lambda_1 = float(bgfm_config.get('lambda_1', 0.5))
    lambda_2 = float(bgfm_config.get('lambda_2', 0.0))
    # Energy-consistency (Boltzmann) term controls. Active iff lambda_2 > 0.
    # The density estimate is expensive (FFJORD trajectory), so we expose
    # step count, Hutchinson samples, and a stride to amortize the cost.
    energy_n_ode_steps = int(bgfm_config.get('energy_n_ode_steps', 8))
    energy_n_hutchinson = int(bgfm_config.get('energy_n_hutchinson', 1))
    energy_every_k_steps = int(bgfm_config.get('energy_every_k_steps', 1))
    energy_enabled = lambda_2 > 0.0
    # Optional anchor loss (Plan C). Active iff lambda_3 > 0.
    # Adds (log p + E/kT + log_Z_pred(mol))^2 to the energy step, where
    # log_Z_pred is a small invariant aux network. This stabilizes the absolute
    # density offset; variance-only L_energy constrains relative probabilities
    # but is insensitive to per-parent additive constants.
    lambda_3 = float(bgfm_config.get('lambda_3', 0.0))
    anchor_enabled = energy_enabled and lambda_3 > 0.0
    log_z_hidden = int(bgfm_config.get('log_z_hidden_dim', 16))
    # Temperature conditioning (Path B): if enabled, sample kT per training step
    # from a log-uniform range and condition the model on kT via a residual
    # projection patched onto the vector_field. See cfm_mol/kt_conditioning.py.
    kT_conditioning = bool(bgfm_config.get('kT_conditioning', False))
    kT_min = float(bgfm_config.get('kT_min', 0.025))
    kT_max = float(bgfm_config.get('kT_max', 1.0))
    # Per-mol perturbation source for the within-molecule Boltzmann variance.
    # The cross-batch form (one geometry per molecule) is mathematically
    # broken -- variance picks up the per-mol log Z spread, not the
    # within-mol Boltzmann deviation. See cfm_mol/bgfm_density.py.
    energy_perturb_paths = bgfm_config.get('energy_perturbation_shards', [])
    energy_b_parents = int(bgfm_config.get('energy_b_parents', 4))
    if isinstance(energy_perturb_paths, str):
        energy_perturb_paths = [energy_perturb_paths]
    if energy_enabled and not energy_perturb_paths:
        raise ValueError(
            "BGFM lambda_2 > 0 requires bgfm.energy_perturbation_shards in config "
            "(list of pre-computed perturbation shards from "
            "scripts/precompute_energy_perturbations.py).")
    kT = float(bgfm_config.get('kT', 1.0))
    force_loss_type = str(bgfm_config.get('force_loss_type', 'mse'))
    if force_loss_type not in {'mse', 'cosine', 'norm_mse'}:
        raise ValueError(
            "bgfm.force_loss_type must be one of: mse, cosine, norm_mse; "
            f"got {force_loss_type!r}"
        )
    force_target_mode = str(bgfm_config.get('force_target_mode', 'true'))
    if force_target_mode not in {'true', 'shuffle_atoms'}:
        raise ValueError(
            "bgfm.force_target_mode must be one of: true, shuffle_atoms; "
            f"got {force_target_mode!r}"
        )
    probe_mode = str(bgfm_config.get('probe_mode', 'path'))
    if probe_mode not in {'path', 'endpoint'}:
        raise ValueError(
            "bgfm.probe_mode must be one of: path, endpoint; "
            f"got {probe_mode!r}"
        )
    t_eval_values = _parse_t_eval_values(bgfm_config)
    warmup_frac = float(bgfm_config.get('warmup_frac', 0.1))
    ramp_frac = float(bgfm_config.get('ramp_frac', 0.2))

    # Optional force-corrected FM target (off by default).
    # This is a heuristic data-augmentation ablation: shift the endpoint
    # x_1 by alpha * F(x_1), i.e. one deterministic low-energy step under
    # the OMol25 force. It should **not** be described as a proof that the
    # induced density is Boltzmann, especially when noise is disabled or the
    # force is evaluated only at the original endpoint. Set
    # `force_correction_alpha = 0` to disable (default).
    force_correction_alpha = float(bgfm_config.get('force_correction_alpha', 0.0))
    force_correction_noise_std = float(bgfm_config.get('force_correction_noise_std', 0.0))

    from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask
    original_training_step = model.training_step

    def bgfm_training_step(self, g, batch_idx):  # noqa: D401
        # 0. Force-corrected FM target. Shift x_1_true BEFORE the FM step
        # so the FM target velocity = (x_1 + alpha*F) - x_0 implicitly.
        # COM-removed afterward to preserve translation invariance.
        if (force_correction_alpha > 0 and 'force_1_true' in g.ndata
                and 'x_1_true' in g.ndata):
            x_1 = g.ndata['x_1_true']
            F = g.ndata['force_1_true']
            shift = force_correction_alpha * F
            if force_correction_noise_std > 0:
                shift = shift + torch.randn_like(F) * force_correction_noise_std
            x_1_shifted = x_1 + shift
            # Re-center per molecule (COM-removed). Use node_batch_idx via
            # FlowMol3's helper to scatter-mean per-graph.
            from flowmol.data_processing.utils import get_batch_idxs as _gbi
            _nbi, _ = _gbi(g)
            B = int(_nbi.max().item()) + 1 if _nbi.numel() else 0
            com = torch.zeros(B, 3, device=x_1_shifted.device, dtype=x_1_shifted.dtype)
            com.scatter_add_(0, _nbi.unsqueeze(-1).expand(-1, 3), x_1_shifted)
            counts = torch.zeros(B, device=x_1_shifted.device, dtype=x_1_shifted.dtype)
            counts.scatter_add_(0, _nbi, torch.ones_like(_nbi, dtype=x_1_shifted.dtype))
            com = com / counts.unsqueeze(-1).clamp(min=1)
            x_1_shifted = x_1_shifted - com[_nbi]
            g.ndata['x_1_true'] = x_1_shifted

        # 1. Original FM step (now learning to flow to the shifted target)
        fm_total = original_training_step(g, batch_idx)

        # Skip if BGFM labels not present (e.g., QM9/GEOM pretraining)
        if 'force_1_true' not in g.ndata:
            return fm_total

        # 2. Auxiliary forward(s) near the data endpoint. Multiple late times make
        # the oral-frame claim stronger: the learned trajectory is force-aligned
        # near the data manifold, not only at one endpoint probe.
        device = g.device
        node_batch_idx, edge_batch_idx = get_batch_idxs(g)
        upper_edge_mask = get_upper_edge_mask(g)

        # Sample kT for this training step if T-conditional, else use fixed kT.
        # Single kT per step (broadcast to all molecules in batch) keeps the
        # design simple while still exposing the model to a wide kT range over
        # many training steps.
        if getattr(self, '_bgfm_kT_conditioning', False):
            from cfm_mol.kt_conditioning import sample_kT
            kT_step = sample_kT(1, device,
                                kT_min=self._bgfm_kT_min,
                                kT_max=self._bgfm_kT_max).item()
            self.log('train_kT', kT_step, on_step=True)
        else:
            kT_step = kT

        force_terms = []
        cosine_terms = []
        for t_eval_i in t_eval_values:
            t = torch.full((g.batch_size,), t_eval_i, device=device, dtype=torch.float32)
            if probe_mode == 'path':
                g_aux = _build_path_aux_graph(
                    g,
                    self.vector_field,
                    t,
                    node_batch_idx=node_batch_idx,
                    edge_batch_idx=edge_batch_idx,
                    upper_edge_mask=upper_edge_mask,
                )
            else:
                g_aux = _build_endpoint_aux_graph(g)
            x_t = g_aux.ndata['x_t']
            forces = g_aux.ndata['force_1_true']
            forces_for_loss = _force_targets(forces, force_target_mode)

            # Forward (with gradients on for backward to model parameters).
            # Pass kT to enable T-conditional models; ignored otherwise.
            vf_kwargs = dict(node_batch_idx=node_batch_idx,
                             upper_edge_mask=upper_edge_mask)
            if getattr(self, '_bgfm_kT_conditioning', False):
                vf_kwargs['kT'] = torch.full((g.batch_size,), kT_step,
                                             device=device, dtype=torch.float32)
            vf_output = self.vector_field(g_aux, t, **vf_kwargs)
            v_theta = vf_output['x']  # (N_total, 3)

            # 3. Score from FM velocity
            t_per_atom = t[node_batch_idx]
            s_theta = score_from_fm_velocity(v_theta, x_t, t_per_atom, prior_std=1.0)

            # 4. Force consistency loss + diagnostic cosine
            # DFT forces on real atoms (fake atoms padded to zero in dataset)
            if force_loss_type == 'mse':
                L_force_i = force_loss(s_theta, forces_for_loss, kT=kT_step)
            else:
                L_force_i = force_direction_loss(
                    s_theta, forces_for_loss, kT=kT_step, mode=force_loss_type,
                )
            cos_i = score_force_cosine(
                s_theta.detach(), forces_for_loss.detach(), kT=kT_step,
            )
            force_terms.append(L_force_i)
            cosine_terms.append(cos_i)
            if force_target_mode != 'true':
                cos_true_i = score_force_cosine(
                    s_theta.detach(), forces.detach(), kT=kT_step,
                )
                self.log('train_score_force_cos_true', cos_true_i.detach(), on_step=True, prog_bar=False)
            if len(t_eval_values) > 1:
                tag = _time_tag(t_eval_i)
                self.log(f'train_L_force_t_{tag}', L_force_i.detach(), on_step=True, prog_bar=False)
                self.log(f'train_score_force_cos_t_{tag}', cos_i.detach(), on_step=True, prog_bar=False)

        L_force = torch.stack(force_terms).mean()
        score_force_cos = torch.stack(cosine_terms).mean()

        # 5. Schedule (always computed -- energy block needs it too)
        batches_per_epoch = getattr(self, 'batches_per_epoch',
                                    len(self.trainer.train_dataloader))
        max_epochs = self.trainer.max_epochs or 1
        epoch_frac = (self.current_epoch + batch_idx / batches_per_epoch) / max_epochs
        l1 = _bgfm_schedule(epoch_frac, lambda_1, warmup_frac, ramp_frac)

        # NaN guard: if force loss went non-finite (numerical instability
        # at late t_eval, large-magnitude outliers, etc.), skip force
        # contribution this step rather than poisoning the optimizer.
        # FM step still backprops, training continues.
        if not torch.isfinite(L_force):
            self.log('train_L_force_nan_skip', 1.0, on_step=True)
            L_force = torch.zeros_like(L_force).detach()
            l1 = 0.0

        # 6. Combine force term (l1=0 if NaN-skipped)
        total = fm_total + l1 * L_force
        self.log('train_L_force', L_force.detach(), on_step=True, prog_bar=False)
        self.log('train_score_force_cos', score_force_cos.detach(), on_step=True, prog_bar=True)
        self.log('train_lambda_1', l1, on_step=True)

        # 7. Energy-consistency (Boltzmann) term. This is the term that makes
        # "Boltzmann-Guided" literal: force alone only matches the local
        # gradient (score = F/kT); energy matches the global log-density
        # shape log p = -E/kT + const.
        #
        # Per-molecule perturbation form: we pull a B-parent batch from a
        # pre-computed perturbation shard (K small geometric perturbations
        # of each parent with OMol25 energies), compute log p for the B*K
        # virtual molecules, and take the within-parent variance of
        # (log p + E/kT). Within-parent isolates the Boltzmann deviation;
        # the cross-batch variance form (one geometry per molecule) was
        # off by 1e7 in magnitude because it picked up per-mol log Z spread.
        # Expensive (FFJORD); applied every energy_every_k_steps.
        if energy_enabled and (batch_idx % energy_every_k_steps == 0):
            pert_loader = self._bgfm_perturbation_loader  # lazily initialized below
            (g_pert, energies_pert, parent_id_pert,
             nbi_pert, uem_pert) = pert_loader.next_batch()
            kT_pert = None
            if getattr(self, '_bgfm_kT_conditioning', False):
                kT_pert = torch.full((g_pert.batch_size,), float(kT_step),
                                     device=g_pert.device, dtype=torch.float32)
            if anchor_enabled:
                # Combined variance + anchor in a single FFJORD pass.
                from cfm_mol.bgfm_density import energy_consistency_loss_per_mol_with_anchor
                # Build PARENT-only invariant features (atom type idx + node
                # batch idx + charges), one copy per parent (NOT per virtual
                # molecule -- log Z is a property of the parent).
                parent_atom_idx, parent_nbi, parent_charges = \
                    pert_loader.parent_invariant_features(g_pert, parent_id_pert)
                L_energy, L_anchor, energy_diag = \
                    energy_consistency_loss_per_mol_with_anchor(
                        self, g_pert, nbi_pert, uem_pert,
                        energies=energies_pert, parent_id=parent_id_pert,
                        log_z_predictor=self._bgfm_log_z_predictor,
                        atom_type_idx=parent_atom_idx,
                        parent_node_batch_idx=parent_nbi,
                        atom_charges_raw=parent_charges,
                        kT=kT_step, kT_tensor=kT_pert,
                        n_ode_steps=energy_n_ode_steps,
                        n_hutchinson=energy_n_hutchinson, prior_std=1.0)
            else:
                from cfm_mol.bgfm_density import energy_consistency_loss_per_mol
                L_energy, energy_diag = energy_consistency_loss_per_mol(
                    self, g_pert, nbi_pert, uem_pert,
                    energies=energies_pert, parent_id=parent_id_pert,
                    kT=kT_step, kT_tensor=kT_pert,
                    n_ode_steps=energy_n_ode_steps,
                    n_hutchinson=energy_n_hutchinson, prior_std=1.0)
                L_anchor = None
            # NaN guard on energy term (v9: also skip on magnitude outliers
            # so that a finite-but-huge L_energy doesn't blow up backward
            # through the FFJORD divergence integral under bf16 mixed
            # precision).
            L_ENERGY_OUTLIER_CAP = float(bgfm_config.get(
                'l_energy_outlier_cap', 1.0e5))
            if (not torch.isfinite(L_energy)) or (L_energy.detach().abs() > L_ENERGY_OUTLIER_CAP):
                self.log('train_L_energy_nan_skip', 1.0, on_step=True)
                self.log('train_L_energy_raw', L_energy.detach() if torch.isfinite(L_energy) else 0.0, on_step=True)
                L_energy = torch.zeros_like(L_energy).detach()
                l2 = 0.0
            else:
                l2 = _bgfm_schedule(epoch_frac, lambda_2, warmup_frac, ramp_frac)
            total = total + l2 * L_energy
            if anchor_enabled and L_anchor is not None:
                # NaN guard on anchor (v9: same outlier check)
                L_ANCHOR_OUTLIER_CAP = float(bgfm_config.get(
                    'l_anchor_outlier_cap', 1.0e12))
                if (not torch.isfinite(L_anchor)) or (L_anchor.detach().abs() > L_ANCHOR_OUTLIER_CAP):
                    self.log('train_L_anchor_nan_skip', 1.0, on_step=True)
                    self.log('train_L_anchor_raw', L_anchor.detach() if torch.isfinite(L_anchor) else 0.0, on_step=True)
                    L_anchor = torch.zeros_like(L_anchor).detach()
                    l3 = 0.0
                else:
                    l3 = _bgfm_schedule(epoch_frac, lambda_3, warmup_frac, ramp_frac)
                total = total + l3 * L_anchor
                self.log('train_L_anchor', L_anchor.detach(), on_step=True, prog_bar=True)
                self.log('train_lambda_3', l3, on_step=True)
                self.log('train_anchor_residual_rms',
                         energy_diag.get('anchor_anchor_residual_rms', 0.0),
                         on_step=True)
                self.log('train_log_z_pred_mean',
                         energy_diag.get('anchor_log_z_pred_mean', 0.0),
                         on_step=True)
            self.log('train_L_energy', L_energy.detach(), on_step=True, prog_bar=True)
            self.log('train_lambda_2', l2, on_step=True)
            self.log('train_logp_mean', energy_diag['logp_mean'], on_step=True)
            self.log('train_energy_residual_within_std',
                     energy_diag['residual_within_std'], on_step=True)
            self.log('train_energy_n_groups',
                     energy_diag['n_groups_used'], on_step=True)

        # Module 3: calibrated scalar energy head.
        # Compute L_head only when an energy head has been attached
        # (see patch_flowmol_bgfm). The head consumes the data
        # endpoint (positions, atom_types, charges) and the targets
        # are the precomputed OMol25 force/energy on the same data
        # point. The force term uses autograd through the head,
        # which requires create_graph=True so the head can be
        # back-propagated.
        head = getattr(self, '_bgfm_energy_head', None)
        lambda_4_full = float(bgfm_config.get('lambda_4', 0.0)) if head is not None else 0.0
        lambda_F = float(bgfm_config.get('lambda_F', 0.1)) if head is not None else 0.0
        l4 = _bgfm_schedule(epoch_frac, lambda_4_full, warmup_frac, ramp_frac)
        if head is not None and l4 > 0.0 and 'force_1_true' in g.ndata:
            from cfm_mol.energy_head import energy_and_force
            from cfm_mol.bgfm_loss import energy_head_calibration_loss
            n_graphs = int(g.batch_size)
            # node_batch_idx is computed earlier in the BGFM block, but
            # the FM-only path can reach here without it; recompute.
            nbi_h, _ = get_batch_idxs(g)
            r_data = g.ndata['x_1_true']
            a_data = g.ndata['a_1_true'].argmax(dim=-1)
            c_data = g.ndata['c_1_true'].argmax(dim=-1)

            # OMol25 energy is sometimes stored as a per-node copy and
            # sometimes as a per-graph tensor. Convert robustly to one
            # scalar target per graph; otherwise the head loss silently
            # shape-mismatches and gets skipped.
            E_node = g.ndata.get('energy_1_true', None)
            if E_node is None:
                E_target = torch.zeros(n_graphs, device=g.device, dtype=r_data.dtype)
            else:
                E_node = E_node.to(device=g.device, dtype=r_data.dtype).view(-1)
                if E_node.numel() == n_graphs:
                    E_target = E_node
                elif E_node.numel() == r_data.shape[0]:
                    E_target = torch.zeros(n_graphs, device=g.device, dtype=r_data.dtype)
                    counts = torch.zeros(n_graphs, device=g.device, dtype=r_data.dtype)
                    E_target.scatter_add_(0, nbi_h, E_node)
                    counts.scatter_add_(0, nbi_h, torch.ones_like(E_node))
                    E_target = E_target / counts.clamp_min(1.0)
                else:
                    raise RuntimeError(
                        f"energy_1_true has unsupported length {E_node.numel()} "
                        f"for {n_graphs} graphs and {r_data.shape[0]} atoms")

            F_target = g.ndata['force_1_true']
            try:
                E_pred, F_pred = energy_and_force(
                    head, r_data, a_data, c_data, nbi_h, n_graphs,
                    create_graph=True,
                )
                L_head, head_diag = energy_head_calibration_loss(
                    E_pred, F_pred, E_target, F_target, lambda_F=lambda_F)
                if torch.isfinite(L_head):
                    total = total + l4 * L_head
                    self.log('train_L_head', L_head.detach(), on_step=True, prog_bar=True)
                    self.log('train_lambda_4', l4, on_step=True)
                    self.log('train_head_energy_mae',
                             head_diag['head_energy_mae'], on_step=True)
                    self.log('train_head_force_mse',
                             head_diag['head_force_mse'], on_step=True)
                else:
                    self.log('train_L_head_nan_skip', 1.0, on_step=True)
            except RuntimeError as e:
                # Autograd through the head can fail on rare malformed
                # batches. Log the skip, but do not let the main FM/BGFM
                # step die. In debugging runs, set bgfm.raise_head_errors=True.
                self.log('train_L_head_runtime_skip', 1.0, on_step=True)
                if bool(bgfm_config.get('raise_head_errors', False)):
                    raise

        self.log('train_total_bgfm', total.detach(), on_step=True, prog_bar=True)
        return total

    import types
    model.training_step = types.MethodType(bgfm_training_step, model)

    # v9: gradient-level NaN/inf guard. The forward-side L_energy /
    # L_anchor NaN-skip catches non-finite *loss values*, but at low kT
    # the loss can be finite while its gradient through the FFJORD
    # divergence integral overflows under bf16 mixed precision and
    # produces NaN parameters at the next optimizer step. We hook
    # on_before_optimizer_step to scan parameter gradients and zero
    # them out if any non-finite value is found; the step then becomes
    # a no-op for the BGFM-affected branches and the model survives.
    grad_nan_skip_enabled = bool(bgfm_config.get('grad_nan_skip', True))
    if grad_nan_skip_enabled:
        original_obos = getattr(model, 'on_before_optimizer_step', None)

        def on_before_optimizer_step(self, optimizer):  # noqa: D401
            n_bad = 0
            for p in self.parameters():
                if p.grad is None:
                    continue
                if not torch.isfinite(p.grad).all():
                    p.grad.zero_()
                    n_bad += 1
            if n_bad > 0:
                self.log('train_grad_nan_skip', float(n_bad), on_step=True)
            if original_obos is not None and callable(original_obos):
                return original_obos(optimizer)
            return None

        model.on_before_optimizer_step = types.MethodType(
            on_before_optimizer_step, model)
        print("[bgfm] v9 grad NaN-skip enabled (zero non-finite grads "
              "before optimizer step)")

    # Lazily initialize the perturbation loader on first energy step (avoids
    # I/O if the run only does FM+force). Held as a model attribute so the
    # iterator state persists across steps.
    if energy_enabled:
        from cfm_mol.perturbation_loader import PerturbationLoader
        # Read atom-map sizing from the existing model so the one-hot widths
        # match the FlowMol vector_field's input expectations.
        # Use atom_type_map length + extras (fake/mask) inferred from model.
        n_atom_types = int(getattr(model, 'n_atom_types',
                                   bgfm_config.get('n_atom_types', 83)))
        n_extra = int(bgfm_config.get('n_extra_atom_classes', 0))
        n_bond_types = int(bgfm_config.get('n_bond_types', 4))
        energy_max_atoms_per_parent = bgfm_config.get('energy_max_atoms_per_parent', None)
        loader = PerturbationLoader(
            shard_paths=energy_perturb_paths,
            n_atom_types=n_atom_types,
            n_extra_atom_classes=n_extra,
            n_charge_classes=6,
            n_bond_types=n_bond_types,
            b_parents=energy_b_parents,
            device=next(model.parameters()).device,
            max_atoms_per_parent=(int(energy_max_atoms_per_parent)
                                  if energy_max_atoms_per_parent is not None else None),
        )
        model._bgfm_perturbation_loader = loader
        print(f"[bgfm] perturbation loader: M={loader.M} parents, K={loader.K}, "
              f"shards={[str(p) for p in energy_perturb_paths]}, B_parents={energy_b_parents}")

        if anchor_enabled:
            from cfm_mol.log_z_predictor import LogZPredictor
            log_z_pred = LogZPredictor(
                n_atom_types=n_atom_types, hidden_dim=log_z_hidden,
            ).to(next(model.parameters()).device)
            # Make it part of the lightning module so its parameters get
            # included in the optimizer + checkpointed alongside the main model.
            # Lightning Module + nn.Module sub-attribute auto-registers params.
            model._bgfm_log_z_predictor = log_z_pred
            n_params = sum(p.numel() for p in log_z_pred.parameters())
            print(f"[bgfm] LogZPredictor enabled (lambda_3={lambda_3}, "
                  f"hidden={log_z_hidden}, params={n_params})")


    # kT conditioning is independent of the energy-density term: force-only
    # T-conditional ablations still need the vector field to accept kT.
    if kT_conditioning:
        from cfm_mol.kt_conditioning import patch_kT_conditioning
        n_hidden_scalars = int(getattr(model.vector_field, 'n_hidden_scalars', 256))
        patch_kT_conditioning(model.vector_field, n_hidden_scalars)
        model._bgfm_kT_conditioning = True
        model._bgfm_kT_min = kT_min
        model._bgfm_kT_max = kT_max
        print(f"[bgfm] kT-conditional training: sample kT log-uniform in "
              f"[{kT_min}, {kT_max}] eV per step")

    # Module 3: calibrated scalar energy head.
    # Attached as model._bgfm_energy_head so its parameters are part of
    # the Lightning module and get included in the optimizer and
    # checkpoint. Accessed by bgfm_training_step via the same name.
    energy_head_enabled = bool(bgfm_config.get('energy_head_enabled', False))
    if energy_head_enabled:
        from cfm_mol.energy_head import EnergyHead
        n_atom_types_h = int(bgfm_config.get('n_atom_types', 83))
        hidden_dim_h = int(bgfm_config.get('energy_head_hidden_dim', 128))
        n_layers_h = int(bgfm_config.get('energy_head_n_layers', 3))
        cutoff_h = float(bgfm_config.get('energy_head_cutoff', 5.0))
        n_rbf_h = int(bgfm_config.get('energy_head_n_rbf', 32))
        head = EnergyHead(
            n_atom_types=n_atom_types_h,
            n_charge_classes=6,
            hidden_dim=hidden_dim_h,
            n_rbf=n_rbf_h,
            cutoff=cutoff_h,
            n_layers=n_layers_h,
        ).to(next(model.parameters()).device)
        model._bgfm_energy_head = head
        n_params_h = sum(p.numel() for p in head.parameters())
        print(f"[bgfm] EnergyHead enabled (hidden={hidden_dim_h}, "
              f"n_layers={n_layers_h}, cutoff={cutoff_h}, params={n_params_h})")


    print(f"[bgfm] BGFM hook installed: lambda_1={lambda_1}, lambda_2={lambda_2}, "
          f"kT={kT} eV, force_loss_type={force_loss_type}, "
          f"force_target_mode={force_target_mode}, probe_mode={probe_mode}, "
          f"t_eval_values={t_eval_values}, warmup={warmup_frac}, ramp={ramp_frac}")
