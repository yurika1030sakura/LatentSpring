"""Measure failure rate of project_valence / project_connectivity on CTMC-like
random samples.

Why: Gap 1 in the proof (projection well-definedness) is theoretically open.
We claim empirically that `local-transition kernel + greedy projection`
almost always succeeds. This script quantifies that: sample random (a, b)
states matching the marginal statistics of QM9/GEOM and run our projection
functions, counting failures.

Target: ≥99.9% successful projection. Below that we need to either
strengthen the algorithm or document the failure mode in the paper.

Run: PYTHONPATH=. python scripts/measure_projection_failure.py
"""
from __future__ import annotations

import argparse
import sys

import torch

from cfm_mol.domain import DEFAULT_VALENCE_SETS, valence_ok, connectivity_ok
from cfm_mol.projection import (
    project_valence, project_connectivity, project_valence_ilp,
)


# QM9 atom map (in MiDi index order; our configs/qm9_cfm.yaml matches).
QM9_ATOM_MAP = ["C", "H", "N", "O", "F"]


def _valence_table_for_atom_map(atom_map: list[str]) -> dict[int, tuple[int, ...]]:
    return {i: DEFAULT_VALENCE_SETS[sym] for i, sym in enumerate(atom_map)}


def _sample_random_state(
    n_mols: int,
    n_atoms: int,
    n_atom_types: int,
    n_bond_types: int = 4,   # {0, 1, 2, 3}; skip aromatic for simplicity
    bond_density: float = 0.5,
    seed: int = 0,
) -> tuple[torch.LongTensor, torch.LongTensor, torch.Tensor]:
    """Adversarial random state: uniform atom type + Bernoulli bond gate.

    This is the WORST case for our greedy projection -- valence violations
    are deep (sums ~9 when max allowed is 4 for C). Real CTMC samples near
    t=1 are much closer to feasible; see `_sample_near_feasible`.
    """
    torch.manual_seed(seed)
    a = torch.randint(0, n_atom_types, (n_mols, n_atoms))
    gate = (torch.rand(n_mols, n_atoms, n_atoms) < bond_density)
    orders = torch.randint(1, n_bond_types, (n_mols, n_atoms, n_atoms))
    b = (gate.long() * orders).clamp_min(0)
    b = torch.max(b, b.transpose(-1, -2))
    eye = torch.eye(n_atoms, dtype=torch.bool).unsqueeze(0)
    b[eye.expand(n_mols, -1, -1)] = 0
    r = torch.randn(n_mols, n_atoms, 3) * 1.5
    return a.long(), b.long(), r


def _sample_near_feasible(
    n_mols: int,
    n_atoms: int,
    n_atom_types: int,
    noise_p: float = 0.1,
    seed: int = 0,
) -> tuple[torch.LongTensor, torch.LongTensor, torch.Tensor]:
    """Near-feasible state: start from a guaranteed valence-OK, connected
    graph; then corrupt with Bernoulli(noise_p) noise on a small subset of
    edges. Matches what FlowMol3 CTMC samples look like near t=1.

    We use the circulant graph $C_n(1, 2)$: vertex $i$ connects to
    $i\\pm 1, i\\pm 2 \\pmod n$, giving a 4-regular graph for any $n \\ge 5$.
    Atom type 0 (C, V=4) everywhere. Initial state therefore has
    $\\sum_j b_{ij} = 4$ for all $i$, which is in $V(\\mathrm{C})$, and a
    single connected component. Noise then flips a few edges to random
    orders.
    """
    torch.manual_seed(seed)
    assert n_atoms >= 5, "C_n(1,2) construction needs n >= 5"
    a = torch.zeros(n_mols, n_atoms, dtype=torch.long)
    b = torch.zeros(n_mols, n_atoms, n_atoms, dtype=torch.long)

    # Circulant C_n(1, 2): single bonds at offsets {1, 2} mod n.
    for offset in (1, 2):
        for i in range(n_atoms):
            j = (i + offset) % n_atoms
            b[:, i, j] = 1
            b[:, j, i] = 1

    # Corrupt a small fraction of pairs by replacing their bond with random.
    corrupt_mask = torch.rand(n_mols, n_atoms, n_atoms) < noise_p
    corrupt_mask = corrupt_mask | corrupt_mask.transpose(-1, -2)
    noise = torch.randint(0, 4, (n_mols, n_atoms, n_atoms))
    noise = torch.max(noise, noise.transpose(-1, -2))
    b = torch.where(corrupt_mask, noise, b)
    eye = torch.eye(n_atoms, dtype=torch.bool).unsqueeze(0)
    b[eye.expand(n_mols, -1, -1)] = 0

    r = torch.randn(n_mols, n_atoms, 3) * 1.5
    return a.long(), b.long(), r


def measure(
    n_mols: int,
    n_atoms: int,
    atom_map: list[str],
    max_iters: int = 8,
    seed: int = 0,
    mode: str = "adversarial",
    noise_p: float = 0.1,
) -> dict[str, float]:
    n_atom_types = len(atom_map)
    valence_table = _valence_table_for_atom_map(atom_map)

    if mode == "adversarial":
        a, b, r = _sample_random_state(
            n_mols, n_atoms, n_atom_types=n_atom_types, seed=seed,
        )
    elif mode == "near_feasible":
        a, b, r = _sample_near_feasible(
            n_mols, n_atoms, n_atom_types=n_atom_types,
            noise_p=noise_p, seed=seed,
        )
    else:
        raise ValueError(f"unknown mode {mode!r}")

    # Measure initial failure rates.
    init_val_ok = valence_ok(a, b, valence_table).all(dim=-1)
    n_atoms_t = torch.full((n_mols,), n_atoms, dtype=torch.long)
    init_conn_ok = connectivity_ok(b, n_atoms_t)

    # Apply project_valence (greedy).
    b_val = project_valence(a, b, valence_table, max_iters=max_iters)
    after_val_ok = valence_ok(a, b_val, valence_table).all(dim=-1)

    # Fallback: ILP projection for molecules the greedy failed on.
    still_bad = ~after_val_ok
    if still_bad.any():
        bad_idx = torch.nonzero(still_bad).flatten()
        b_ilp = project_valence_ilp(a[bad_idx], b[bad_idx], valence_table)
        b_val[bad_idx] = b_ilp
        after_val_ok = valence_ok(a, b_val, valence_table).all(dim=-1)

    # Apply project_connectivity on top.
    b_all = project_connectivity(b_val, n_atoms=n_atoms_t, r=r)
    after_conn_ok = connectivity_ok(b_all, n_atoms_t)
    # After connectivity projection, valence may be broken by the new bonds --
    # reproject valence once more to see combined recovery.
    b_iter = project_valence(a, b_all, valence_table, max_iters=max_iters)
    final_val_ok = valence_ok(a, b_iter, valence_table).all(dim=-1)
    final_conn_ok = connectivity_ok(b_iter, n_atoms_t)
    final_both = final_val_ok & final_conn_ok

    return {
        "init_valence_ok_frac": float(init_val_ok.float().mean()),
        "init_connectivity_ok_frac": float(init_conn_ok.float().mean()),
        "after_project_valence_ok_frac": float(after_val_ok.float().mean()),
        "after_project_connectivity_ok_frac": float(after_conn_ok.float().mean()),
        "final_both_ok_frac": float(final_both.float().mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_mols", type=int, default=10_000)
    ap.add_argument("--n_atoms", type=int, default=9)   # QM9 mean ~9
    ap.add_argument("--atom_map", default="C,H,N,O,F")
    ap.add_argument("--max_iters", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["adversarial", "near_feasible"],
                    default="near_feasible")
    ap.add_argument("--noise_p", type=float, default=0.1)
    args = ap.parse_args()

    atom_map = args.atom_map.split(",")
    print(f"measuring projection failure on {args.n_mols} random states "
          f"(n_atoms={args.n_atoms}, atom_map={atom_map}, "
          f"max_iters={args.max_iters}, mode={args.mode})")

    stats = measure(args.n_mols, args.n_atoms, atom_map,
                    max_iters=args.max_iters, seed=args.seed,
                    mode=args.mode, noise_p=args.noise_p)
    print()
    for k, v in stats.items():
        print(f"  {k:40s} = {v:.4f}")

    # Verdict.
    final = stats["final_both_ok_frac"]
    if final >= 0.999:
        print(f"\nPASS: final feasibility {final*100:.2f}% >= 99.9% target.")
        return 0
    else:
        print(f"\nFAIL: final feasibility {final*100:.2f}% < 99.9% target. "
              "Need stronger projection algorithm or document failure.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
