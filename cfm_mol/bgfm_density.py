"""BGFM energy-consistency: log-density via flow change-of-variables.

This module implements the *second* BGFM loss term (lambda_2 * L_energy).
It estimates a **position-only** FFJORD density for coordinates r,
conditioned on fixed molecular identity/composition c.  The loss enforces
local relative Boltzmann consistency within perturbations of the same parent
molecule; it is not an exact global molecular Boltzmann sampler.

Density of a data point under the learned position flow (conditioned on
fixed atom types/charges, since E(r,c) = E(positions | atoms)):

    log p_theta(x_1) = log p_0(x_0) - integral_0^1 div(v_theta^x)(x_t, t) dt,
        where x_0 = phi_1^{-1}(x_1)  (reverse-time ODE from data to prior).

We accumulate the divergence along the reverse trajectory using the
Hutchinson estimator (cfm_mol.bgfm_loss.divergence_hutchinson). For the
variance-form energy loss, an exact log-density is not required: a
batch-consistent biased estimate suffices, because constant offsets and
consistent bias cancel in Var(log p + E/kT). This lets us use a small
number of ODE steps (default 10) and 1 Hutchinson sample, keeping the
cost ~10x the base FM step rather than ~50x.

References: Chen et al. 2018 (Neural ODE), Grathwohl et al. 2019 (FFJORD).
"""
from __future__ import annotations

import math
from typing import Callable

import torch

from cfm_mol.bgfm_loss import divergence_hutchinson


def gaussian_prior_log_density(
    x0: torch.Tensor,
    n_atoms_per_graph: torch.Tensor,
    prior_std: float = 1.0,
    com_free: bool = True,
) -> torch.Tensor:
    """log p_0(x_0) for a centered isotropic Gaussian prior, per graph.

    For COM-free coordinates the effective dimension is 3*(N-1) (the COM
    is removed). We use that as the normalizer; constant terms cancel in
    the variance-form energy loss but we keep them for interpretability.

    Args:
        x0: (N_total, 3) prior-side coordinates
        n_atoms_per_graph: (B,)
        prior_std: std of the Gaussian prior
        com_free: if True use 3*(N-1) effective dim, else 3*N

    Returns:
        (B,) log-density per graph
    """
    device = x0.device
    cumsum = torch.cat([
        torch.zeros(1, dtype=n_atoms_per_graph.dtype, device=device),
        n_atoms_per_graph.cumsum(0),
    ])
    B = n_atoms_per_graph.shape[0]
    out = torch.zeros(B, device=device, dtype=x0.dtype)
    var = prior_std ** 2
    for b in range(B):
        seg = x0[cumsum[b]:cumsum[b + 1]]
        n = seg.shape[0]
        d = 3 * (n - 1) if com_free else 3 * n
        sq = (seg ** 2).sum()
        out[b] = -0.5 * sq / var - 0.5 * d * math.log(2 * math.pi * var)
    return out


def make_position_velocity_fn(
    model,
    g_template,
    t_scalar: torch.Tensor,
    node_batch_idx: torch.Tensor,
    upper_edge_mask: torch.Tensor,
    kT: torch.Tensor | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a position-only velocity closure v(x) = v_theta^x(x, t | a,c,e fixed).

    The discrete channels (a_t, c_t, e_t) are held at their values already
    set on g_template (data labels). Only x_t varies. Returns a function
    suitable for divergence_hutchinson.

    Args:
        kT: optional (B,) per-graph temperature (eV) for T-conditional models.
            If None, vector_field is called without kT (non-conditional).
    """
    kwargs = dict(node_batch_idx=node_batch_idx, upper_edge_mask=upper_edge_mask)
    if kT is not None:
        kwargs["kT"] = kT
    def v_fn(x: torch.Tensor) -> torch.Tensor:
        g_template.ndata['x_t'] = x
        vf_output = model.vector_field(g_template, t_scalar, **kwargs)
        return vf_output['x']
    return v_fn


def log_density_via_flow(
    model,
    g_aux,
    node_batch_idx: torch.Tensor,
    upper_edge_mask: torch.Tensor,
    n_ode_steps: int = 10,
    n_hutchinson: int = 1,
    prior_std: float = 1.0,
    for_training: bool = False,
    kT: torch.Tensor | None = None,
) -> torch.Tensor:
    """Estimate log p_theta(x_1) for the data points in g_aux.

    g_aux must have ndata['x_1_true'] (data positions) and discrete
    channels set to their data labels (a_t = a_1_true, etc.) -- the same
    g_aux used for the force loss.

    Procedure (reverse-time Euler with divergence accumulation):
      x <- x_1 ; logp_acc <- 0
      for t from 1 down to 0 in n_ode_steps:
          div = Hutchinson divergence of v_theta^x at (x, t)
          x   <- x - dt * v_theta^x(x, t)       # reverse step
          logp_acc <- logp_acc + dt * div        # accumulate integral
      log p_theta(x_1) = log p_0(x) + logp_acc   # x is now x_0

    Note: integral_0^1 div dt is accumulated as sum(dt * div). The sign
    works out so log p_1 = log p_0(x_0) - integral, and integral is
    accumulated with the +dt*div convention while stepping backward
    (dt > 0 magnitude), giving log p_1 = log p_0(x_0) + logp_acc only if
    we define div with the reverse-time sign. We use the standard
    instantaneous change of variables; see unit test for sign validation.

    Returns:
        (B,) estimated log-densities
    """
    device = g_aux.device
    B = g_aux.batch_size
    x = g_aux.ndata['x_1_true'].detach().clone()

    dt = 1.0 / n_ode_steps
    logp_integral = torch.zeros(B, device=device, dtype=x.dtype)
    n_apg = _n_atoms_per_graph(g_aux, node_batch_idx)

    # Reverse-time trajectory: t goes 1 -> 0
    for step in range(n_ode_steps):
        t_val = 1.0 - (step + 0.5) * dt   # midpoint time of this interval
        t_scalar = torch.full((B,), t_val, device=device, dtype=torch.float32)

        x_req = x.detach().requires_grad_(True)
        v_fn = make_position_velocity_fn(
            model, g_aux, t_scalar, node_batch_idx, upper_edge_mask, kT=kT)
        # Divergence at current point (per graph). create_graph only when
        # this density feeds a training loss; for eval we keep it False so
        # no second-order graph is built (avoids OOM over the trajectory).
        div = divergence_hutchinson(
            v_fn, x_req, n_apg,
            n_samples=n_hutchinson, rademacher=True,
            create_graph=for_training)
        # Velocity for the reverse Euler step (no grad needed for the step)
        with torch.no_grad():
            g_aux.ndata['x_t'] = x
            vf_kwargs = dict(node_batch_idx=node_batch_idx,
                             upper_edge_mask=upper_edge_mask)
            if kT is not None:
                vf_kwargs["kT"] = kT
            v = model.vector_field(g_aux, t_scalar, **vf_kwargs)['x']
            x = x - dt * v   # reverse step toward prior
        # Accumulate divergence integral. Detach for eval to free graph.
        logp_integral = logp_integral + dt * (div if for_training else div.detach())

    # x is now x_0 (prior side). log p_0(x_0):
    logp0 = gaussian_prior_log_density(
        x, _n_atoms_per_graph(g_aux, node_batch_idx), prior_std=prior_std)

    # log p_1(x_1) = log p_0(x_0) - integral_0^1 div dt.
    # Reverse accumulation gives logp_integral ~ integral div dt, so:
    return logp0 - logp_integral


def _n_atoms_per_graph(g, node_batch_idx: torch.Tensor) -> torch.Tensor:
    """Atoms per graph from node_batch_idx."""
    B = int(node_batch_idx.max().item()) + 1 if node_batch_idx.numel() else 0
    return torch.bincount(node_batch_idx, minlength=B)


def energy_consistency_loss(
    model,
    g_aux,
    node_batch_idx: torch.Tensor,
    upper_edge_mask: torch.Tensor,
    energies: torch.Tensor,
    kT: float = 1.0,
    n_ode_steps: int = 10,
    n_hutchinson: int = 1,
    prior_std: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """[DEPRECATED] Cross-batch variance of (log p + E/kT).

    WARNING: this form is mathematically broken when the batch contains
    geometries of DIFFERENT molecules. The Boltzmann condition
    log p(x|m) + E_m(x)/kT = -log Z_m is per-molecule -- log Z_m varies
    across molecules (dominated by size/composition), so cross-molecule
    variance picks up the log Z spread, NOT within-molecule Boltzmann
    deviation. Empirically this loss read ~1e9 on a mixed-size batch,
    making the energy term untrainable.

    Use `energy_consistency_loss_per_mol` instead, which takes K-perturbation
    tuples per parent molecule and computes within-molecule variance,
    cleanly isolating the per-molecule log Z constant.

    Kept here only for the unit test that validates the variance machinery
    on a synthetic same-molecule batch.

    Args:
        energies: (B,) per-graph DFT energies (eV)

    Returns:
        (loss, diagnostics_dict)
    """
    log_p = log_density_via_flow(
        model, g_aux, node_batch_idx, upper_edge_mask,
        n_ode_steps=n_ode_steps, n_hutchinson=n_hutchinson, prior_std=prior_std,
        for_training=True)
    residual = log_p + energies / kT
    loss = residual.var()
    diag = {
        "logp_mean": float(log_p.detach().mean().item()),
        "logp_std": float(log_p.detach().std().item()) if log_p.numel() > 1 else 0.0,
        "residual_std": float(residual.detach().std().item()) if residual.numel() > 1 else 0.0,
    }
    return loss, diag


def within_group_variance_loss(
    residual: torch.Tensor,
    parent_id: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Mean over parents of the within-parent variance of `residual`.

    The core math of L_energy_per_mol, factored out so the Boltzmann grouping
    logic can be unit-tested without the FFJORD machinery. NaN residuals are
    dropped; parents with <2 valid samples are excluded.

    Args:
        residual: (M*K,) the quantity (log p + E/kT) per virtual molecule.
        parent_id: (M*K,) integer parent-molecule id, 0..M-1.

    Returns:
        (loss, diagnostics)
    """
    valid = torch.isfinite(residual)
    if int(valid.sum().item()) < 2:
        return residual.sum() * 0.0, {
            "n_valid": int(valid.sum().item()), "n_groups_used": 0,
            "residual_within_std": 0.0,
        }

    res_v = residual[valid]
    pid_v = parent_id[valid]
    device = res_v.device
    M = int(parent_id.max().item()) + 1

    counts = torch.zeros(M, device=device, dtype=res_v.dtype)
    counts.scatter_add_(0, pid_v, torch.ones_like(res_v))
    sums = torch.zeros(M, device=device, dtype=res_v.dtype)
    sums.scatter_add_(0, pid_v, res_v)
    means = sums / counts.clamp(min=1)
    sq_dev = (res_v - means[pid_v]) ** 2
    sq_sums = torch.zeros(M, device=device, dtype=res_v.dtype)
    sq_sums.scatter_add_(0, pid_v, sq_dev)

    # Biased variance (divide by K). Const rescale absorbed into lambda_2.
    used_mask = counts >= 2
    n_used = int(used_mask.sum().item())
    if n_used == 0:
        return residual.sum() * 0.0, {
            "n_valid": int(valid.sum().item()), "n_groups_used": 0,
            "residual_within_std": 0.0,
        }

    group_vars = sq_sums[used_mask] / counts[used_mask]
    loss = group_vars.mean()
    return loss, {
        "n_valid": int(valid.sum().item()),
        "n_groups_used": n_used,
        "residual_within_std": float(group_vars.detach().sqrt().mean().item()),
    }


def energy_consistency_loss_per_mol(
    model,
    g_pert,
    node_batch_idx: torch.Tensor,
    upper_edge_mask: torch.Tensor,
    energies: torch.Tensor,
    parent_id: torch.Tensor,
    kT=1.0,
    n_ode_steps: int = 10,
    n_hutchinson: int = 1,
    prior_std: float = 1.0,
    kT_tensor: torch.Tensor | None = None,
    discrete_log_p: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """L_energy = mean_m Var_k(log p_theta(x_{m,k}) + E(x_{m,k}) / kT).

    Per-molecule Boltzmann variance: for each parent molecule m, compute the
    variance of (log p + E/kT) across its K perturbations, then average
    over parent molecules. This isolates the within-molecule Boltzmann
    deviation from the across-molecule log Z spread (which is dominated by
    size/composition and is NOT what we want to penalize).

    Args:
        g_pert: batched DGL graph with M*K virtual molecules (K perturbations
            of each of M parent molecules). x_t = x_1_true at entry; the
            discrete channels (a_t, c_t, e_t) are set to data labels and
            held fixed across the reverse-time ODE.
        node_batch_idx: (N_total,) virtual-molecule index per node
        upper_edge_mask: edge mask for the batched graph
        energies: (M*K,) OMol25 per-graph energies in eV. NaN entries are
            silently skipped (failed inference at preprocess).
        parent_id: (M*K,) parent molecule id in 0..M-1. K virtual molecules
            sharing the same parent_id are perturbations of the same molecule.
        kT, n_ode_steps, n_hutchinson, prior_std: as in log_density_via_flow.

    Returns:
        (loss, diagnostics_dict)
    """
    log_p = log_density_via_flow(
        model, g_pert, node_batch_idx, upper_edge_mask,
        n_ode_steps=n_ode_steps, n_hutchinson=n_hutchinson, prior_std=prior_std,
        for_training=True, kT=kT_tensor)
    # Module 3.5 (joint density): if the caller provides a per-graph
    # discrete log-probability log p_theta(c), combine it with the
    # coordinate-conditional FFJORD integral so that the variance is
    # computed on the joint log-density log p_theta(x) = log p_theta(r|c)
    # + log p_theta(c). When discrete_log_p is None, this falls back to
    # the position-only formulation.
    if discrete_log_p is not None:
        log_p_joint = log_p + discrete_log_p
    else:
        log_p_joint = log_p
    # kT can be a scalar (fixed-T training) or a (B,) tensor per virtual mol
    # (T-conditional training). The variance loss is computed per-parent so we
    # need E/kT to broadcast correctly.
    if kT_tensor is not None:
        # kT_tensor has shape (M*K,) — per virtual molecule
        residual = log_p_joint + energies / kT_tensor
    else:
        residual = log_p_joint + energies / float(kT)
    loss, diag = within_group_variance_loss(residual, parent_id)
    diag["logp_mean"] = float(log_p.detach().mean().item())
    if discrete_log_p is not None:
        diag["logp_joint_mean"] = float(log_p_joint.detach().mean().item())
        diag["logp_discrete_mean"] = float(discrete_log_p.detach().mean().item())
    # Expose the per-virtual-mol joint log-density for downstream consumers
    # (the Boltzmann bridge loss reuses it without a second FFJORD pass).
    # Stored under a leading-underscore key so it is filtered out of
    # scalar metric logs.
    diag["_log_p_for_bridge"] = log_p_joint
    return loss, diag


def energy_anchor_loss(
    log_p: torch.Tensor,
    energies: torch.Tensor,
    log_Z_pred: torch.Tensor,
    parent_id: torch.Tensor,
    kT=1.0,
) -> tuple[torch.Tensor, dict]:
    """L_anchor = mean_m ((log p_theta(x_{m,k}) + E(x_{m,k}) / kT + log Z_pred(m))^2).

    The mathematical complement to L_energy_per_mol: where the variance form
    only fixes relative probabilities up to a per-molecule additive constant,
    the anchor form stabilizes that offset via a small invariant composition
    head.  The head is not supervised by true thermodynamic log Z and should
    not be interpreted as a guaranteed partition-function estimator.

    Args:
        log_p:       (M*K,) log-density at each virtual molecule (from
                     log_density_via_flow). Requires grad if training.
        energies:    (M*K,) OMol25 per-virtual-mol energies in eV.
        log_Z_pred:  (M,) per-parent log Z prediction from the LogZPredictor
                     aux network. INVARIANT in positions by construction.
        parent_id:   (M*K,) parent-mol id (0..M-1).
        kT:          eV; matches the L_energy convention.

    The (M*K,)-shaped residual is squared and averaged over valid entries.
    NaN energies are skipped.
    """
    # Clamp log_p to a bounded range before squaring. FFJORD log-density
    # can drift to extreme values during training (e.g. -1e5 if a perturbed
    # geometry is far from the model's support); squaring then producing
    # unbounded loss + gradient. The clamp range is wide enough that a
    # well-trained model never hits it, but it prevents NaN cascades.
    #
    # NOTE: at low kT (room T = 0.025 eV), E/kT magnitudes are ~40x larger
    # than at kT=1 eV. log_p must compensate with similar magnitude to keep
    # residual bounded. Bumping cap accordingly.
    LOG_P_CAP = 1e6
    log_p_clamped = log_p.clamp(min=-LOG_P_CAP, max=LOG_P_CAP)
    # kT can be float or per-virtual-mol tensor (T-conditional training)
    residual = log_p_clamped + energies / kT + log_Z_pred[parent_id]
    valid = torch.isfinite(residual)
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return residual.sum() * 0.0, {
            "n_valid": 0, "anchor_residual_rms": 0.0,
            "log_z_pred_mean": float(log_Z_pred.detach().mean().item()),
        }
    sq = residual[valid].pow(2)
    loss = sq.mean()
    return loss, {
        "n_valid": n_valid,
        "anchor_residual_rms": float(sq.detach().sqrt().mean().item()),
        "log_z_pred_mean": float(log_Z_pred.detach().mean().item()),
        "log_z_pred_std": float(log_Z_pred.detach().std().item())
                          if log_Z_pred.numel() > 1 else 0.0,
    }


def energy_consistency_loss_per_mol_with_anchor(
    model,
    g_pert,
    node_batch_idx: torch.Tensor,
    upper_edge_mask: torch.Tensor,
    energies: torch.Tensor,
    parent_id: torch.Tensor,
    log_z_predictor,
    atom_type_idx: torch.Tensor,
    parent_node_batch_idx: torch.Tensor,
    atom_charges_raw: torch.Tensor | None = None,
    kT=1.0,
    n_ode_steps: int = 10,
    n_hutchinson: int = 1,
    prior_std: float = 1.0,
    kT_tensor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Variance + anchor energy loss in one call (single FFJORD pass).

    Computes log_p once, then both:
      L_var    = within-parent variance of (log_p + E/kT)
      L_anchor = mean((log_p + E/kT + log_Z_pred)^2)

    The caller scales each with its own coefficient (lambda_2 and lambda_3)
    so the trade-off can be tuned without re-running the expensive FFJORD.

    Args:
        log_z_predictor:        LogZPredictor module
        atom_type_idx:          (N_parent,) atom type indices for the M
                                PARENT molecules (one copy, NOT M*K)
        parent_node_batch_idx:  (N_parent,) parent-graph index per atom
        atom_charges_raw:       (N_parent,) optional raw atom charges (the
                                preprocess puts total mol charge on atom 0)

    Returns:
        (L_var, L_anchor, diag) -- two losses + merged diagnostics.
    """
    log_p = log_density_via_flow(
        model, g_pert, node_batch_idx, upper_edge_mask,
        n_ode_steps=n_ode_steps, n_hutchinson=n_hutchinson, prior_std=prior_std,
        for_training=True, kT=kT_tensor)
    # Variable kT support: kT_tensor (M*K,) per virtual mol if T-conditional;
    # else scalar kT.
    kT_div = kT_tensor if kT_tensor is not None else float(kT)
    residual = log_p + energies / kT_div
    L_var, diag = within_group_variance_loss(residual, parent_id)
    diag["logp_mean"] = float(log_p.detach().mean().item())

    log_Z_pred = log_z_predictor(
        atom_type_idx=atom_type_idx,
        node_batch_idx=parent_node_batch_idx,
        atom_charges_raw=atom_charges_raw,
    )
    L_anchor, anchor_diag = energy_anchor_loss(
        log_p, energies, log_Z_pred, parent_id, kT=kT_div)
    # Prefix anchor diagnostics so logger keys don't collide
    for k, v in anchor_diag.items():
        diag[f"anchor_{k}"] = v
    # Expose the per-virtual-mol coordinate log-density for the Boltzmann
    # bridge loss (reuses this FFJORD pass). Leading-underscore key signals
    # that downstream scalar logging should ignore it.
    diag["_log_p_for_bridge"] = log_p
    return L_var, L_anchor, diag
