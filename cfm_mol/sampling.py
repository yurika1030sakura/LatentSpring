"""Constrained flow-ODE sampler with discrete flow + gluing.

Full inference loop for product-manifold constrained flow matching
(methods_derivation.tex Section 3 + Appendix A).

Per Euler step:
  1. Continuous Euler on the current fibre: v_theta(r,t,a,b)
     -> tangent-project -> step -> retract (cfm_mol.fibre.euler_step_on_fibre).
  2. Discrete Euler step on (a, b) via a simple CTMC with rate matrices emitted
     by the network (Gat et al. 2024). Minimal implementation: apply a mask
     over discrete channels and accept per-token proposals with probability
     Delta t * rate.
  3. If the discrete state changed, project onto the valence + connectivity
     manifold (cfm_mol.projection.project_valence, project_connectivity),
     then retract r onto the new steric fibre (gluing, Lemma 4.2).

The callable `v_theta` is expected to accept (r_t, t, a, b) and return a
tuple (v_coord, rate_a, rate_b), where:
  - v_coord : (B, N, 3) velocity on the coordinate channel.
  - rate_a  : (B, N, A) per-atom rate over atom-type classes (or None to hold a fixed).
  - rate_b  : (B, N, N, B) per-edge rate over bond-order classes (or None to hold b fixed).

For the Week 1 toy setup we use a reduced version that holds (a, b) fixed
and only evolves the coordinate channel -- the `sample_coords_only` function
below is what cfm_mol.train_toy uses to demonstrate end-to-end inference.
"""
from __future__ import annotations

import torch

from cfm_mol.domain import (
    DEFAULT_VALENCE_SETS,
    connectivity_ok,
    steric_all_ok,
    valence_ok,
)
from cfm_mol.fibre import euler_step_on_fibre, sample_prior
from cfm_mol.projection import (
    gluing_retract,
    project_connectivity,
    project_valence,
)


# ---------------------------------------------------------------------------
# Minimal sampler: coord-only (fixed atom types / bonds)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_coords_only(
    v_net: torch.nn.Module,
    a: torch.LongTensor,
    b: torch.LongTensor,
    d_min_table: torch.Tensor,
    n_steps: int = 50,
    prior_scale: float = 3.0,
) -> tuple[torch.Tensor, float]:
    """Sample r by integrating the constrained flow ODE, with (a, b) held fixed.

    Used by cfm_mol.train_toy for the Week 1 smoke test.

    Args
    ----
    v_net : callable (r_t, t, a, b) -> (B, N, 3)
    a : (B, N) atom types.
    b : (B, N, N) bond-order indices.
    d_min_table : (A, A)
    n_steps : Euler steps (T = 1.0 / n_steps each).
    prior_scale : std of Gaussian prior before retraction.

    Returns
    -------
    r : (B, N, 3) final samples.
    validity : fraction of samples passing the steric check.
    """
    B, N = a.shape
    device = a.device

    r = sample_prior(
        n_atoms=N, batch_size=B, a=a, d_min_table=d_min_table,
        scale=prior_scale, device=device,
    )
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((B,), k * dt, device=device)
        v = v_net(r, t, a, b)
        r = euler_step_on_fibre(r, v, dt, a, d_min_table)

    validity = steric_all_ok(r, a, d_min_table).float().mean().item()
    return r, validity


# ---------------------------------------------------------------------------
# Full sampler: continuous + discrete flow + gluing
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_full(
    v_net: torch.nn.Module,
    batch_size: int,
    n_atoms: int,
    n_atom_types: int,
    n_bond_classes: int,
    d_min_table: torch.Tensor,
    valence_table: dict[int, tuple[int, ...]] | None = None,
    n_steps: int = 100,
    prior_scale: float = 3.0,
    device: str | torch.device = "cpu",
) -> dict:
    """Full constrained-flow inference: coord ODE + discrete flow + gluing.

    v_net is expected to emit (v_coord, rate_a, rate_b) at each step.

    Args
    ----
    v_net : callable (r, t, a, b) -> (v_coord, rate_a, rate_b).
            rate_a : (B, N, n_atom_types) rate matrix row for each atom.
            rate_b : (B, N, N, n_bond_classes) rate matrix row for each edge.
    batch_size, n_atoms : shapes.
    n_atom_types, n_bond_classes : discrete dimensions.
    d_min_table : (n_atom_types, n_atom_types)
    valence_table : see domain.DEFAULT_VALENCE_SETS (defaults to that).

    Returns
    -------
    dict with keys 'r', 'a', 'b', 'validity_steric', 'validity_valence',
    'validity_connectivity', 'validity_all'.
    """
    if valence_table is None:
        # Index-aligned: build from DEFAULT_VALENCE_SETS keyed on the symbol
        # order expected by the caller. For Week 2 we use MiDi ordering.
        midi_symbols = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
        valence_table = {
            i: DEFAULT_VALENCE_SETS[sym]
            for i, sym in enumerate(midi_symbols[:n_atom_types])
            if sym in DEFAULT_VALENCE_SETS
        }

    B, N = batch_size, n_atoms

    # Initialise: random atom types, all zero bonds, Gaussian-then-retracted r.
    a = torch.randint(0, n_atom_types, (B, N), device=device)
    b = torch.zeros(B, N, N, dtype=torch.long, device=device)
    r = sample_prior(
        n_atoms=N, batch_size=B, a=a, d_min_table=d_min_table,
        scale=prior_scale, device=device,
    )

    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((B,), k * dt, device=device)

        # Network emits velocity and discrete rates.
        v_coord, rate_a, rate_b = v_net(r, t, a, b)

        # 1. continuous Euler on fibre
        r = euler_step_on_fibre(r, v_coord, dt, a, d_min_table)

        # 2. discrete flow: per-atom/per-edge class transitions with Delta t * rate
        a_new, b_new = a.clone(), b.clone()
        if rate_a is not None:
            # Softmax rate interpreted as target distribution; take a
            # Categorical sample with probability dt of changing.
            probs = torch.softmax(rate_a, dim=-1)               # (B, N, A)
            change = torch.rand(B, N, device=device) < dt
            sampled = torch.distributions.Categorical(probs=probs).sample()
            a_new = torch.where(change, sampled, a)
        if rate_b is not None:
            probs = torch.softmax(rate_b, dim=-1)               # (B, N, N, Bnd)
            change = torch.rand(B, N, N, device=device) < dt
            sampled = torch.distributions.Categorical(probs=probs).sample()
            b_new = torch.where(change, sampled, b)
            # Enforce symmetry + zero diagonal.
            b_new = torch.max(b_new, b_new.transpose(-1, -2))
            eye = torch.eye(N, dtype=torch.bool, device=device)
            b_new[:, eye] = 0

        # 3. discrete projection + gluing if changed
        changed = (a_new != a).any(dim=-1) | (b_new != b).any(dim=(-1, -2))
        if changed.any():
            b_new = project_valence(a_new, b_new, valence_table)
            b_new = project_connectivity(b_new, r=r)
            r_retracted, _ = gluing_retract(r, a_new, d_min_table)
            r = torch.where(changed.view(-1, 1, 1), r_retracted, r)

        a, b = a_new, b_new

    # Validity report
    v_ster = steric_all_ok(r, a, d_min_table)
    v_val = valence_ok(a, b, valence_table).all(dim=-1)
    n_atoms_tensor = torch.full((B,), N, dtype=torch.long, device=device)
    v_conn = connectivity_ok(b, n_atoms_tensor)
    v_all = v_ster & v_val & v_conn

    return {
        "r": r, "a": a, "b": b,
        "validity_steric": v_ster.float().mean().item(),
        "validity_valence": v_val.float().mean().item(),
        "validity_connectivity": v_conn.float().mean().item(),
        "validity_all": v_all.float().mean().item(),
    }
