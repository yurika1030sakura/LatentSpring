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

    # NEW METHOD: Force-Corrected FM target (Approach B).
    # Shift the per-graph FM endpoint x_1 by alpha * F(x_1), i.e. one
    # implicit gradient-descent step on the OMol25 potential. The flow then
    # learns to map prior samples to force-corrected (lower-energy) targets
    # rather than raw data. The induced density at t=1 is provably closer
    # to Boltzmann than the raw-data distribution, with no auxiliary losses,
    # no FFJORD integration, no score-from-velocity singularity. Set
    # `force_correction_alpha = 0` to disable (default).
    #
    # Mathematical view: targeting (x_1 + alpha*F) is exactly one Euler
    # step of overdamped Langevin dynamics on E (with zero noise). Adding
    # noise via `force_correction_noise_std > 0` gives the full Langevin
    # step and pushes the target distribution toward the Boltzmann ensemble
    # at temperature kT = alpha / (noise_std^2 / 2).
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
            # NaN guard on energy term
            if not torch.isfinite(L_energy):
                self.log('train_L_energy_nan_skip', 1.0, on_step=True)
                L_energy = torch.zeros_like(L_energy).detach()
                l2 = 0.0
            else:
                l2 = _bgfm_schedule(epoch_frac, lambda_2, warmup_frac, ramp_frac)
            total = total + l2 * L_energy
            if anchor_enabled and L_anchor is not None:
                # NaN guard on anchor
                if not torch.isfinite(L_anchor):
                    self.log('train_L_anchor_nan_skip', 1.0, on_step=True)
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

        self.log('train_total_bgfm', total.detach(), on_step=True, prog_bar=True)
        return total

    import types
    model.training_step = types.MethodType(bgfm_training_step, model)

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
            device="cuda",
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
            ).to("cuda")
            # Make it part of the lightning module so its parameters get
            # included in the optimizer + checkpointed alongside the main model.
            # Lightning Module + nn.Module sub-attribute auto-registers params.
            model._bgfm_log_z_predictor = log_z_pred
            n_params = sum(p.numel() for p in log_z_pred.parameters())
            print(f"[bgfm] LogZPredictor enabled (lambda_3={lambda_3}, "
                  f"hidden={log_z_hidden}, params={n_params})")

        if kT_conditioning:
            from cfm_mol.kt_conditioning import patch_kT_conditioning
            # n_hidden_scalars must match the vector_field's scalar embedding dim.
            # Read it from the model config (vector_field block).
            n_hidden_scalars = int(getattr(model.vector_field, 'n_hidden_scalars', 256))
            patch_kT_conditioning(model.vector_field, n_hidden_scalars)
            model._bgfm_kT_conditioning = True
            model._bgfm_kT_min = kT_min
            model._bgfm_kT_max = kT_max
            print(f"[bgfm] kT-conditional training: sample kT log-uniform in "
                  f"[{kT_min}, {kT_max}] eV per step")

    print(f"[bgfm] BGFM hook installed: lambda_1={lambda_1}, lambda_2={lambda_2}, "
          f"kT={kT} eV, force_loss_type={force_loss_type}, "
          f"force_target_mode={force_target_mode}, probe_mode={probe_mode}, "
          f"t_eval_values={t_eval_values}, warmup={warmup_frac}, ramp={ramp_frac}")
