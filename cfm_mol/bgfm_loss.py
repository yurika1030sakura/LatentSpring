"""Boltzmann-Guided Flow Matching (BGFM) loss components.

Formal derivations in notes/bgfm_method.md (Sections 3-7).
Theorem statements in notes/appendix_A_v4.tex.

BGFM does **not** by itself prove exact sampling from the global
Boltzmann distribution over all molecular identities. The implemented
training signal is a local, conditional-coordinate regularizer: for a
fixed molecular identity/composition ``c``, it encourages the model's
position density to satisfy local Boltzmann relative probabilities under
an external neural-potential energy ``E_NP(r, c)``.

The main loss is

    L_total = L_FM + lambda_1 * L_force + lambda_2 * L_energy
              + lambda_3 * L_anchor + lambda_4 * L_head

    L_FM     : standard flow-matching velocity loss
    L_force  : late-time FM-implied coordinate score matches F_NP/kT
    L_energy : within-parent Var(log p_pos(r|c) + E_NP(r,c)/kT)
    L_anchor : optional composition-conditioned offset stabilizer
    L_head   : optional calibrated scalar energy-head loss

These terms are complementary approximations. They should be described
as Boltzmann *regularization* or local conditional Boltzmann alignment,
not as an exact guarantee that generated molecules follow a natural
Boltzmann distribution.

This module exposes three core functions:

  score_from_fm_velocity(v_theta, x_t, t, prior_std)
      Closed-form: given FM velocity at time t, compute implied score of
      the time-t marginal density. Derivation: Section 3.1 of bgfm_method.md.

  divergence_exact_atomwise(v_fn, x)
      Exact divergence of v w.r.t. atom positions, O(N * 3) backward
      passes. Faster than Hutchinson for moderate N because of the
      per-atom block-diagonal structure of GVP-Transformer.

  bgfm_loss(batch, model, lambda_1, lambda_2, kT, t_score, n_steps_trace)
      Full 3-term loss. Consumes {forces, energies} precomputed in
      preprocess_omol25.py.

Unit tests (tests/test_bgfm_loss.py) verify:
  - score_from_fm_velocity on 1D Gaussian toy
  - divergence_exact_atomwise against finite differences
  - bgfm_loss has no NaN on small synthetic batch
  - lambda_* = 0 reduces exactly to vanilla FM loss
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Score derivation from flow-matching velocity
# ---------------------------------------------------------------------------

def score_from_fm_velocity(
    v_theta: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    prior_std: float = 1.0,
) -> torch.Tensor:
    """Closed-form score of the flow-matching marginal density at time t.

    For linear interpolant x_t = (1 - t) * x_0 + t * x_1 with Gaussian
    prior x_0 ~ N(0, prior_std^2 * I), the marginal density p_t satisfies

        nabla_x log p_t(x) = - [x - t * E[x_1 | x_t = x]] / [(1 - t) * prior_std^2]

    We recover E[x_1 | x_t = x] from the marginal velocity:

        v*(x_t, t) = E[x_1 - x_0 | x_t = x]
        => E[x_1 | x_t = x] = x_t + (1 - t) * v*(x_t, t)       (holds on support)

    Substituting x_1_pred = x_t + (1 - t) * v_theta gives

        s_theta(x_t, t) = [t * v_theta(x_t,t) - x_t] / [(1-t) * prior_std^2].

    At t = 1 this expression is singular, so BGFM evaluates it at late
    but finite t_eval values such as 0.70, 0.80, and 0.90.

    Args:
        v_theta: model velocity, shape (B, *), matches x_t
        x_t: current state at time t, shape (B, *)
        t: time, shape (B,) or scalar. Must satisfy 0 <= t < 1.
        prior_std: std of the Gaussian base distribution (default 1.0)

    Returns:
        s: implied score, same shape as x_t.
    """
    # Guard against t = 1 singularity
    one_minus_t = (1.0 - t).clamp(min=1e-3)
    if t.dim() < x_t.dim():
        # broadcast t to x_t shape
        t = t.view(-1, *[1] * (x_t.dim() - 1))
        one_minus_t = one_minus_t.view(-1, *[1] * (x_t.dim() - 1))

    # Standard formula: s(x, t) = [t * v - x] / [(1 - t) * prior_std^2]
    # Formula for the Gaussian-prior linear interpolant
    # with N(0, sigma^2 I) prior: s_t=(t*v-x_t)/((1-t)*sigma^2).
    score = (t * v_theta - x_t) / (one_minus_t * (prior_std ** 2))
    # Defensive clamp: at late t the (1-t) denominator amplifies score
    # values, and downstream MSE on raw score can produce unbounded loss
    # (cascading into bf16/fp16 overflow). Clamp per-atom L2 norm to a
    # generous bound (1000 corresponds to a 30 sigma deviation per coord,
    # which a well-trained model never reaches in normal operation).
    SCORE_NORM_CAP = 1000.0
    norm = score.norm(dim=-1, keepdim=True).clamp(min=1.0)
    scale = (SCORE_NORM_CAP / norm).clamp(max=1.0)
    score = score * scale
    return score


# ---------------------------------------------------------------------------
# 2. Divergence of the velocity field
# ---------------------------------------------------------------------------

def divergence_exact_atomwise(
    v_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    n_atoms_per_graph: torch.Tensor,
) -> torch.Tensor:
    """Exact divergence via per-coordinate backward passes. Reference
    implementation for unit tests on small molecules. 3N backward passes;
    too slow for training (use divergence_hutchinson instead).

    tr(J) = sum_{i, c} partial v[i, c] / partial x[i, c]

    For each (i, c), differentiate v[i, c] w.r.t. x and extract [i, c].

    Args:
        v_fn: maps positions (N_total, 3) -> velocity (N_total, 3).
        x: positions (N_total, 3). requires_grad=True.
        n_atoms_per_graph: (B,) atoms per graph; divergence summed per-graph.

    Returns:
        div_v: (B,) tensor.
    """
    if not x.requires_grad:
        x = x.detach().requires_grad_(True)
    v = v_fn(x)
    assert v.shape == x.shape

    N = x.shape[0]
    per_atom_div = torch.zeros(N, device=x.device, dtype=x.dtype)
    for i in range(N):
        for c in range(3):
            grad_ic = torch.autograd.grad(
                outputs=v[i, c],
                inputs=x,
                retain_graph=True,
                create_graph=x.requires_grad and (i < N - 1 or c < 2),
            )[0]
            per_atom_div[i] = per_atom_div[i] + grad_ic[i, c]

    # Sum per-atom -> per-graph
    cumsum = torch.cat([
        torch.zeros(1, dtype=n_atoms_per_graph.dtype, device=n_atoms_per_graph.device),
        n_atoms_per_graph.cumsum(0),
    ])
    B = n_atoms_per_graph.shape[0]
    return torch.stack([per_atom_div[cumsum[b]:cumsum[b + 1]].sum() for b in range(B)])


def divergence_hutchinson(
    v_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    n_atoms_per_graph: torch.Tensor,
    n_samples: int = 1,
    rademacher: bool = True,
    create_graph: bool | None = None,
) -> torch.Tensor:
    """Hutchinson trace estimator for the divergence of v_fn at x.

    Identity: tr(J) = E_{xi ~ p}[xi^T J xi] for any distribution p with
    E[xi] = 0 and E[xi xi^T] = I. With Rademacher xi in {-1, +1}, the
    estimator variance is a factor of ~2 smaller than Gaussian.

    Computed via a single backward pass per xi:
        xi^T J xi = (d/dx) [v(x) . xi] . xi

    Args:
        v_fn: positions (N_total, 3) -> velocity (N_total, 3)
        x: positions, requires_grad externally
        n_atoms_per_graph: (B,) for per-graph reduction
        n_samples: number of Hutchinson samples (averaged)
        rademacher: use {+/-1} vs N(0, 1)

    Returns:
        div_v: (B,) per-graph divergence
    """
    if not x.requires_grad:
        x = x.detach().requires_grad_(True)
    v = v_fn(x)
    assert v.shape == x.shape

    # create_graph controls whether the divergence is itself differentiable
    # (needed only when the divergence feeds a TRAINING loss that backprops
    # to model params). For pure evaluation (log-density value only) keep it
    # False to avoid building a second-order graph -> huge memory.
    if create_graph is None:
        create_graph = bool(x.requires_grad)

    cumsum = torch.cat([
        torch.zeros(1, dtype=n_atoms_per_graph.dtype, device=n_atoms_per_graph.device),
        n_atoms_per_graph.cumsum(0),
    ])
    B = n_atoms_per_graph.shape[0]
    acc = torch.zeros(B, device=x.device, dtype=x.dtype)

    for k in range(n_samples):
        if rademacher:
            xi = (torch.randint(0, 2, x.shape, device=x.device, dtype=x.dtype) * 2 - 1)
        else:
            xi = torch.randn_like(x)
        # When create_graph=True (training), the OUTER loss.backward() will
        # backprop through this divergence again (second-order), which needs
        # v's forward graph retained. So retain whenever create_graph, or
        # between Hutchinson samples.
        retain = create_graph or (k < n_samples - 1)
        grad_xi = torch.autograd.grad(
            outputs=(v * xi).sum(),
            inputs=x,
            retain_graph=retain,
            create_graph=create_graph,
            allow_unused=True,
        )[0]
        if grad_xi is None:
            # v has no dependence on x (e.g. a constant field) -> div = 0.
            continue
        # (grad_xi * xi) per atom gives xi^T J xi per 3-coord block
        per_atom = (grad_xi * xi).sum(dim=-1)  # (N_total,)
        for b in range(B):
            acc[b] = acc[b] + per_atom[cumsum[b]:cumsum[b + 1]].sum()
    return acc / float(n_samples)


# ---------------------------------------------------------------------------
# 3. Force consistency loss
# ---------------------------------------------------------------------------

def force_loss(
    s_theta: torch.Tensor,
    forces_data: torch.Tensor,
    kT: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """L_force = ||s_theta(x, t_eval) - F_data(x) / kT||^2

    In the current BGFM training hook, s_theta is usually evaluated at a
    late conditional-path point x_t, while forces_data is the precomputed
    endpoint force F(x_1). This is a late-time approximation F(x_t)≈F(x_1),
    used to avoid online OMol25 calls inside every training step.

    The physical sign convention: if p_theta = Boltzmann(E, kT), then
        nabla_x log p_theta(x) = -nabla_x E(x) / kT = F(x) / kT
    so the target for s_theta is F_data / kT (positive).

    Args:
        s_theta: model's implied score, shape (N_total, 3)
        forces_data: DFT forces, shape (N_total, 3), eV/A
        kT: temperature * Boltzmann const, eV (room temp ~0.025 eV)
        reduction: 'mean' or 'sum'

    Returns:
        scalar loss
    """
    target = forces_data / kT
    err = (s_theta - target).pow(2).sum(dim=-1)  # (N_total,)
    # Per-atom L2 error cap. OMol25 forces have rare outliers (transition
    # states, large radicals) where |F|/kT can reach 100-1000; squared MSE
    # on those single atoms can overwhelm the batch gradient. Cap per atom
    # at a generous bound; equivalent to switching from MSE to L1 above
    # this threshold.
    PER_ATOM_ERR_CAP = 1e4
    err = err.clamp(max=PER_ATOM_ERR_CAP)
    if reduction == "mean":
        return err.mean()
    elif reduction == "sum":
        return err.sum()
    else:
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction}")


def score_force_cosine(
    s_theta: torch.Tensor,
    forces_data: torch.Tensor,
    kT: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean cosine alignment between model score and Boltzmann force target.

    This is a diagnostic, not a training loss. If BGFM is doing what it says,
    the direction of the implied score should align with F/kT even before the
    force MSE is fully optimized.
    """
    target = forces_data / kT
    target_norm = target.norm(dim=-1)
    score_norm = s_theta.norm(dim=-1)
    mask = target_norm > eps
    if not mask.any():
        return torch.zeros((), device=s_theta.device, dtype=s_theta.dtype)
    cos = (s_theta * target).sum(dim=-1) / (score_norm * target_norm + eps)
    return cos[mask].mean()


def force_direction_loss(
    s_theta: torch.Tensor,
    forces_data: torch.Tensor,
    kT: float = 1.0,
    eps: float = 1e-8,
    mode: str = "cosine",
) -> torch.Tensor:
    """Direction-only force consistency loss.

    The default force MSE is a strong magnitude target. Near ``t -> 1`` the
    FM-implied score includes a ``1 / (1 - t)`` factor, so magnitude errors
    can dominate and over-steer samples. This loss keeps the local Boltzmann
    direction signal while reducing sensitivity to force scale.

    Args:
        s_theta: model's implied score, shape (N_total, 3)
        forces_data: DFT forces, shape (N_total, 3), eV/A
        kT: temperature scale, eV
        eps: zero-force guard
        mode: ``cosine`` for 1 - cos, or ``norm_mse`` for MSE between
            normalized vectors.

    Returns:
        scalar loss over atoms with nonzero force targets.
    """
    target = forces_data / kT
    target_norm = target.norm(dim=-1, keepdim=True)
    score_norm = s_theta.norm(dim=-1, keepdim=True)
    mask = target_norm.squeeze(-1) > eps
    if not mask.any():
        return torch.zeros((), device=s_theta.device, dtype=s_theta.dtype)

    target_unit = target[mask] / target_norm[mask].clamp_min(eps)
    score_unit = s_theta[mask] / score_norm[mask].clamp_min(eps)
    if mode == "cosine":
        return 1.0 - (score_unit * target_unit).sum(dim=-1).mean()
    if mode == "norm_mse":
        return F.mse_loss(score_unit, target_unit)
    raise ValueError(f"unknown force direction loss mode: {mode}")


# ---------------------------------------------------------------------------
# 4. Energy consistency loss
# ---------------------------------------------------------------------------

def energy_loss_variance(
    log_p_theta: torch.Tensor,
    energies_data: torch.Tensor,
    kT: float = 1.0,
) -> torch.Tensor:
    """L_energy = Var_batch[log p_theta(x) + E(x) / kT]

    If p_theta is Boltzmann, then log p_theta(x) + E(x)/kT = -log Z is
    a constant. The variance across a batch of training molecules
    measures how close log p_theta matches the Boltzmann form, up to an
    unknown partition function (absorbed into the batch mean).

    This is PREFERRED over |log p + E/kT - const|^2 because we don't
    need to learn or tune the normalization constant.

    Args:
        log_p_theta: (B,) tensor of model log-densities on training data
        energies_data: (B,) tensor of OMol25 energies (eV)
        kT: temperature * Boltzmann const, eV

    Returns:
        scalar variance
    """
    residual = log_p_theta + energies_data / kT
    return residual.var()


# ---------------------------------------------------------------------------
# 5. Total BGFM loss
# ---------------------------------------------------------------------------

def bgfm_total_loss(
    fm_loss_val: torch.Tensor,
    force_loss_val: Optional[torch.Tensor],
    energy_loss_val: Optional[torch.Tensor],
    lambda_1: float = 0.5,
    lambda_2: float = 0.1,
    epoch_frac: float = 1.0,
    warmup_frac: float = 0.1,
    ramp_frac: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    """Total BGFM loss with warmup/ramp schedule.

    Schedule for lambda_1, lambda_2:
        0 <= epoch_frac < warmup_frac:                lambda = 0 (FM only)
        warmup_frac <= epoch_frac < warmup + ramp:    linear ramp 0 -> lambda
        else:                                          full lambda

    Args:
        fm_loss_val: scalar FM loss
        force_loss_val: scalar force loss, or None
        energy_loss_val: scalar energy loss, or None
        lambda_1: force-loss weight (full)
        lambda_2: energy-loss weight (full)
        epoch_frac: fraction of training elapsed, in [0, 1]
        warmup_frac: fraction of training with lambda = 0
        ramp_frac: fraction of training to linearly ramp up

    Returns:
        (total, components_dict) where components_dict logs individual
        weighted terms for wandb.
    """
    def _scale(full_lambda: float) -> float:
        if epoch_frac < warmup_frac:
            return 0.0
        elif epoch_frac < warmup_frac + ramp_frac:
            return full_lambda * (epoch_frac - warmup_frac) / ramp_frac
        else:
            return full_lambda

    l1 = _scale(lambda_1)
    l2 = _scale(lambda_2)

    total = fm_loss_val.clone()
    comps = {"L_FM": fm_loss_val.detach().item(), "lambda_1": l1, "lambda_2": l2}

    if force_loss_val is not None and l1 > 0:
        total = total + l1 * force_loss_val
        comps["L_force"] = force_loss_val.detach().item()
        comps["L_force_weighted"] = l1 * force_loss_val.detach().item()

    if energy_loss_val is not None and l2 > 0:
        total = total + l2 * energy_loss_val
        comps["L_energy"] = energy_loss_val.detach().item()
        comps["L_energy_weighted"] = l2 * energy_loss_val.detach().item()

    comps["L_total"] = total.detach().item()
    return total, comps


# ---------------------------------------------------------------------------
# 6. End-to-end BGFM step (convenience wrapper)
# ---------------------------------------------------------------------------

def compute_bgfm_step(
    model,  # FlowMol LightningModule
    batch,  # DGL batched graph with forces, energies
    lambda_1: float,
    lambda_2: float,
    kT: float = 1.0,
    t_eval: float = 0.95,
    use_exact_divergence: bool = True,
    epoch_frac: float = 1.0,
    warmup_frac: float = 0.1,
    ramp_frac: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    """Compute the BGFM total loss for a single training step.

    High-level outline:
        1. Standard FM forward: v_theta = model(x_t, t) for sampled t
           -> L_FM (position MSE + discrete CE, as before)
        2. Evaluate score at data endpoint: s_theta(x_1, t_eval)
           -> L_force against precomputed forces in batch
        3. Compute log p_theta(x_1) via trajectory divergence integral
           -> L_energy (variance form) against precomputed energies
        4. Combine with warmup/ramp schedule

    Args:
        model: FlowMol instance (for model.vector_field.forward)
        batch: DGL graph with .ndata['forces'] and .ndata['energies']
        lambda_1, lambda_2: force/energy loss weights
        kT: temperature scale (eV)
        t_eval: time at which to evaluate score (close to 1)
        use_exact_divergence: if True, use atom-wise exact; else Hutchinson
        epoch_frac: for warmup schedule

    Returns:
        (loss, log_dict) where log_dict contains each loss component
        for wandb / tensorboard.

    This legacy convenience wrapper is intentionally not the active
    training entry point. The production integration is the monkey-patched
    Lightning training step installed by ``cfm_mol.bgfm_train_hook``.
    Keeping this function as a hard failure prevents reviewers and scripts
    from accidentally assuming an unimplemented path is used.
    """
    raise NotImplementedError(
        "compute_bgfm_step is a deprecated contract only. Use "
        "cfm_mol.bgfm_train_hook.patch_flowmol_bgfm, which installs the "
        "actual BGFM Lightning training_step."
    )


def energy_head_calibration_loss(
    E_pred: torch.Tensor,
    F_pred: torch.Tensor,
    E_target: torch.Tensor,
    F_target: torch.Tensor,
    lambda_F: float = 0.1,
    energy_reduction: str = "mean",
) -> tuple[torch.Tensor, dict]:
    r"""Calibration loss for the BGFM scalar energy head.

    .. math::
        \mathcal{L}_{\rm head}
        = \mathbb{E}\bigl|\hat E_\psi(r,c) - E_{\rm NP}(r,c)\bigr|
        + \lambda_F\,\mathbb{E}\bigl\|-\nabla_r \hat E_\psi(r,c) - F_{\rm NP}(r,c)\bigr\|^2.

    The energy term uses L1 (absolute) regression to stay robust under
    OMol25's heavy-tailed energy distribution. The force term uses
    MSE so that gradient information is matched in expectation.

    Args:
        E_pred: (B,) predicted energy per graph from the head.
        F_pred: (N_total, 3) predicted forces from autograd through
            the head.
        E_target: (B,) target neural-potential energy per graph.
        F_target: (N_total, 3) target neural-potential forces.
        lambda_F: weight on the force-matching term.
        energy_reduction: 'mean' or 'sum'.

    Returns:
        (loss, diag) where diag carries per-term magnitudes for
        logging.
    """
    e_diff = (E_pred - E_target).abs()
    if energy_reduction == "mean":
        e_loss = e_diff.mean()
    elif energy_reduction == "sum":
        e_loss = e_diff.sum()
    else:
        raise ValueError(energy_reduction)
    f_diff = (F_pred - F_target).pow(2).sum(dim=-1)
    f_loss = f_diff.mean()
    total = e_loss + lambda_F * f_loss
    diag = {
        "head_energy_mae": float(e_loss.detach().item()),
        "head_force_mse": float(f_loss.detach().item()),
    }
    return total, diag
