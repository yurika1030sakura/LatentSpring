"""Tangent projection and retraction on the steric fibre $\\mathcal{M}_{\\mathrm{ster}}(a)$.

The steric fibre at discrete state $a$ is
    $\\mathcal{M}_{\\rm ster}(a) = \\{ r \\in \\mathbb{R}^{3N} : \\|r_i - r_j\\| > d_{\\min}(a_i, a_j),\\ i \\ne j \\}$,
the complement of a finite union of open balls.

For constrained flow matching on this fibre we need:
  1. **tangent_project(v, r, a)**: project a proposed velocity v onto the tangent
     cone of $\\mathcal{M}_{\\rm ster}(a)$ at r. For each pair (i,j) that is on or
     near the steric sphere, clip the outward-normal component of the relative
     velocity so the pair cannot approach.
  2. **retract(r, a)**: retract a point that may have drifted outside the fibre
     back onto it. Iterative pairwise sphere push-apart, same geometry as
     before but now serving as the ODE's post-step projection, not as a
     reflection of stochastic noise.
  3. **sample_interpolant(r0, r1, t, a)**: produce r_t = (1-t) r0 + t r1
     retracted onto the fibre -- the training-time conditional interpolant.

These three are the operational kernel of product-manifold constrained
flow matching (methods_derivation.tex Appendix A).
"""
from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Pairwise machinery
# ---------------------------------------------------------------------------

def _pair_distances_and_dmins(
    r: torch.Tensor,
    a: torch.LongTensor,
    d_min_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pairwise distances and d_min(a_i, a_j) lookup.

    Returns
    -------
    dists : (B, N, N)
    d_min_pair : (B, N, N)
    """
    B, N, _ = r.shape
    diffs = r.unsqueeze(-2) - r.unsqueeze(-3)          # (B, N, N, 3)
    dists = diffs.norm(dim=-1).clamp_min(1e-12)
    ai = a.unsqueeze(-1).expand(B, N, N)
    aj = a.unsqueeze(-2).expand(B, N, N)
    d_min_pair = d_min_table[ai, aj]
    return dists, d_min_pair


# ---------------------------------------------------------------------------
# Tangent projection
# ---------------------------------------------------------------------------

def tangent_project(
    v: torch.Tensor,
    r: torch.Tensor,
    a: torch.LongTensor,
    d_min_table: torch.Tensor,
    margin: float = 0.05,
) -> torch.Tensor:
    """Project velocity v onto the tangent cone of the steric fibre at r.

    For every pair (i, j) with distance d_{ij} within `margin` of d_min(a_i, a_j)
    (i.e. the pair is on or near the steric sphere), the outward-normal
    component of the relative velocity v_i - v_j along the pair axis is
    clipped to zero if it is negative (i.e. would push them closer).

    Pairs already far from the boundary contribute nothing.

    Math: at the steric boundary {||r_i - r_j|| = d_min}, the tangent cone on
    the relative-coordinate (r_i - r_j) requires <d(r_i - r_j)/dt, axis> >= 0
    where axis = (r_i - r_j) / ||r_i - r_j||. We split the correction equally
    between v_i and v_j to preserve centre-of-mass velocity.

    This is a **cone** projection (each active constraint clipped), not an
    orthogonal Euclidean projection. For velocity fields near the boundary it
    is equivalent to first-order.

    Args
    ----
    v : (B, N, 3) proposed velocity.
    r : (B, N, 3) current position (assumed inside or on fibre).
    a : (B, N) atom-type indices.
    d_min_table : (A, A).
    margin : activate the constraint when d_{ij} < d_min + margin.

    Returns
    -------
    v_proj : (B, N, 3) tangent-projected velocity.
    """
    B, N, _ = r.shape
    v_proj = v.clone()

    dists, d_min_pair = _pair_distances_and_dmins(r, a, d_min_table)
    active = (dists < d_min_pair + margin) & ~torch.eye(
        N, dtype=torch.bool, device=r.device
    ).unsqueeze(0)                                      # (B, N, N)

    # Unit vector from j to i.
    diffs = r.unsqueeze(-2) - r.unsqueeze(-3)          # (B, N, N, 3)
    axis = diffs / dists.unsqueeze(-1)                 # (B, N, N, 3)

    # Relative velocity along axis: positive = separating (good),
    # negative = approaching (bad, clip).
    vi = v_proj.unsqueeze(-2).expand(B, N, N, 3)
    vj = v_proj.unsqueeze(-3).expand(B, N, N, 3)
    rel_vel = vi - vj                                  # (B, N, N, 3)
    v_along = (rel_vel * axis).sum(dim=-1)             # (B, N, N)
    bad = active & (v_along < 0)

    # Per-pair correction: subtract half of v_along along axis from i, add to j.
    # Serial over bad pairs to keep math exact for overlapping pairs.
    # TODO[Week 2]: vectorise scatter for larger molecules.
    for b_idx in range(B):
        pair_idx = torch.nonzero(bad[b_idx], as_tuple=False)   # (K, 2)
        for ij in pair_idx:
            i, j = int(ij[0].item()), int(ij[1].item())
            if i >= j:
                continue
            correction = 0.5 * v_along[b_idx, i, j] * axis[b_idx, i, j]
            v_proj[b_idx, i] = v_proj[b_idx, i] - correction
            v_proj[b_idx, j] = v_proj[b_idx, j] + correction
    return v_proj


# ---------------------------------------------------------------------------
# Retraction (post-Euler-step correction)
# ---------------------------------------------------------------------------

def retract(
    r: torch.Tensor,
    a: torch.LongTensor,
    d_min_table: torch.Tensor,
    margin: float = 0.01,
    max_iters: int = 20,
) -> tuple[torch.Tensor, torch.BoolTensor]:
    """Push pairs with d_{ij} < d_min back to d_min + margin.

    Iterates pairwise push-apart of the most-violating pair per molecule
    until all pairs satisfy d_{ij} >= d_min (or max_iters hit).

    For a violating pair (i, j), both atoms are moved along the pair axis by
    half of the deficit (d_target - d_{ij}) / 2. This preserves the pair
    midpoint and is minimal displacement among feasible corrections for that
    single pair.

    Gluing-lemma constant (methods_derivation.tex Lemma 4.2): per pair the
    displacement is bounded by (d_min - d_{ij})_+ / 2 <= d_min. Under the
    local-transition assumption, C <= 2 * max_{a,a'} d_min(a, a').

    Args
    ----
    r : (B, N, 3)
    a : (B, N)
    d_min_table : (A, A)
    margin : feasibility buffer
    max_iters : maximum push rounds

    Returns
    -------
    r_new : (B, N, 3) on (or near) the fibre
    converged : (B,) True where all pairs feasible
    """
    B, N, _ = r.shape
    r_cur = r.clone()
    converged = torch.zeros(B, dtype=torch.bool, device=r.device)

    for _ in range(max_iters):
        dists, d_min_pair = _pair_distances_and_dmins(r_cur, a, d_min_table)
        slack = dists - d_min_pair                    # neg = violating
        eye = torch.eye(N, dtype=torch.bool, device=r.device).unsqueeze(0)
        violating = (slack < 0) & ~eye

        done = ~violating.reshape(B, -1).any(dim=-1)
        converged = converged | done
        if done.all():
            break

        slack_masked = torch.where(violating, slack, torch.full_like(slack, float("inf")))
        flat_idx = slack_masked.reshape(B, -1).argmin(dim=-1)
        i_idx, j_idx = flat_idx // N, flat_idx % N

        for b_idx in range(B):
            if done[b_idx]:
                continue
            i, j = int(i_idx[b_idx]), int(j_idx[b_idx])
            ri, rj = r_cur[b_idx, i], r_cur[b_idx, j]
            d = (ri - rj).norm().clamp_min(1e-12)
            d_target = d_min_pair[b_idx, i, j] + margin
            axis = (ri - rj) / d
            delta = 0.5 * (d_target - d)
            r_cur[b_idx, i] = ri + delta * axis
            r_cur[b_idx, j] = rj - delta * axis
    return r_cur, converged


# ---------------------------------------------------------------------------
# Conditional interpolant (training time)
# ---------------------------------------------------------------------------

def sample_interpolant(
    r0: torch.Tensor,
    r1: torch.Tensor,
    t: torch.Tensor,
    a: torch.LongTensor,
    d_min_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Straight-line CFM interpolant, retracted onto the fibre.

    r_t = (1 - t) r0 + t r1, then retract onto M_ster(a).

    Both r0 and r1 are assumed inside the fibre. The straight-line segment
    between them is always inside a convex fibre, but ours (complement of a
    union of balls) is non-convex. For most (r0, r1) pairs the segment stays
    inside; when it cuts through an obstacle ball the retraction corrects it.

    The associated conditional velocity is r1 - r0 (used for the CFM regression
    target). The retraction's derivative is identity when the segment stays
    inside and a projection at boundary points; in the high-data limit
    (Lipman 2023 Thm 1) the network learns the correct marginal velocity.

    Args
    ----
    r0 : (B, N, 3) source (e.g. Gaussian, retracted once)
    r1 : (B, N, 3) data (on fibre)
    t : (B,) in [0, 1]
    a : (B, N)
    d_min_table : (A, A)

    Returns
    -------
    r_t : (B, N, 3) interpolant on the fibre
    u : (B, N, 3) conditional velocity r1 - r0
    """
    if t.ndim == 1:
        t_ = t.view(-1, 1, 1)
    else:
        t_ = t
    r_t_raw = (1 - t_) * r0 + t_ * r1
    r_t, _ = retract(r_t_raw, a, d_min_table)
    u = r1 - r0
    return r_t, u


# ---------------------------------------------------------------------------
# Euler step on the fibre (inference time)
# ---------------------------------------------------------------------------

def euler_step_on_fibre(
    r: torch.Tensor,
    v: torch.Tensor,
    dt: float,
    a: torch.LongTensor,
    d_min_table: torch.Tensor,
) -> torch.Tensor:
    """One Euler step of the constrained flow ODE.

    Composes:
      1. tangent-project v  -->  v_proj
      2. Euler update        -->  r + dt * v_proj
      3. retract onto fibre  -->  r_new

    This is the operational inference step. Call T times for sampling.
    """
    v_proj = tangent_project(v, r, a, d_min_table)
    r_raw = r + dt * v_proj
    r_new, _ = retract(r_raw, a, d_min_table)
    return r_new


# ---------------------------------------------------------------------------
# Prior sample: Gaussian, retracted onto fibre
# ---------------------------------------------------------------------------

def sample_prior(
    n_atoms: int,
    batch_size: int,
    a: torch.LongTensor,
    d_min_table: torch.Tensor,
    scale: float = 3.0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Sample r_0 from the source distribution: Gaussian in R^{3N} retracted
    onto the fibre.

    `scale` controls the Gaussian std; should be large enough that typical
    samples need only a few retraction iterations.
    """
    r = torch.randn(batch_size, n_atoms, 3, device=device) * scale
    r, _ = retract(r, a, d_min_table)
    return r
