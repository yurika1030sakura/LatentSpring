"""Precompute OMol25 stats needed by FlowMol3's model/datamodule:

  - train_data_marginal_dists.pt : (p_a, p_c, p_e, p_c_given_a)
  - train_data_n_atoms_histogram.pt : 1-D tensor, counts at each n

These are global stats -- a small sample (default 100k) is enough to estimate
marginal distributions accurately. Uses covalent-radius bonds to match the
OMol25Dataset on-the-fly logic.

Also writes a val split (default 10k random indices from the same shards)
as a small .pt file to use as the validation set during training.

Usage:
  python scripts/prepare_omol25_stats.py \\
      --src /n/netscratch/ryl_lab/Lab/omol25/train_4M \\
      --out_dir /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/data/omol25_4m_stats \\
      --sample 100000
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from cfm_mol.data.omol25 import DEFAULT_ATOM_MAP_OMOL25, _SYM_TO_Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True,
                    help="Directory or list (comma-sep) of LMDB shards.")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=100_000,
                    help="Number of random molecules to sample for stats.")
    ap.add_argument("--val_size", type=int, default=10_000)
    ap.add_argument("--max_atoms", type=int, default=120)
    ap.add_argument("--bond_tolerance", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--atom_map", type=str, default=None,
                    help="Comma-sep overriding atom map (default: 83 elems).")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    atom_map = args.atom_map.split(",") if args.atom_map else list(DEFAULT_ATOM_MAP_OMOL25)
    n_atom_types = len(atom_map)
    z_to_idx = {_SYM_TO_Z[s]: i for i, s in enumerate(atom_map) if s in _SYM_TO_Z}
    print(f"atom_map has {n_atom_types} elements")

    from fairchem.core.datasets import AseDBDataset
    from ase.neighborlist import natural_cutoffs, neighbor_list

    src = args.src if "," not in args.src else args.src.split(",")
    db = AseDBDataset({"src": src if not isinstance(src, list) else src})
    N = len(db)
    print(f"dataset has {N} molecules")

    sample_size = min(args.sample, N)
    sample_idx = np.random.choice(N, size=sample_size, replace=False)
    print(f"sampling {sample_size} molecules for marginal-dist estimate ...")

    p_a = np.zeros(n_atom_types, dtype=np.int64)
    p_c = np.zeros(6, dtype=np.int64)  # charges in [-2, 3] shifted to [0, 5]
    p_e = np.zeros(4, dtype=np.int64)  # bond types: none, single, double, triple
    p_c_given_a = np.zeros((n_atom_types, 6), dtype=np.int64)
    n_atoms_counter = np.zeros(args.max_atoms + 1, dtype=np.int64)
    ok, skipped, element_miss, element_counts = 0, 0, 0, {}

    for i, db_idx in enumerate(sample_idx):
        if i % 5000 == 0 and i > 0:
            print(f"  processed {i}/{sample_size} (ok={ok}, skip={skipped}, elem_miss={element_miss})")
        try:
            atoms = db.get_atoms(int(db_idx))
        except Exception:
            skipped += 1
            continue
        n = len(atoms)
        if n < 2 or n > args.max_atoms:
            skipped += 1
            continue
        z = atoms.get_atomic_numbers()
        type_idx = np.array([z_to_idx.get(int(zi), -1) for zi in z], dtype=np.int64)
        if (type_idx < 0).any():
            for zi in z:
                element_counts[int(zi)] = element_counts.get(int(zi), 0) + 1
            element_miss += 1
            continue

        # atoms / charges
        np.add.at(p_a, type_idx, 1)
        # total charge spread to atom 0
        charge_total = int(atoms.info.get("charge", 0))
        per_atom_c = np.zeros(n, dtype=np.int64)
        if charge_total != 0:
            per_atom_c[0] = charge_total
        per_atom_c = np.clip(per_atom_c, -2, 3) + 2
        np.add.at(p_c, per_atom_c, 1)
        # p(c|a)
        for a_idx, c_idx in zip(type_idx, per_atom_c):
            p_c_given_a[int(a_idx), int(c_idx)] += 1

        # bonds
        cutoffs = natural_cutoffs(atoms, mult=args.bond_tolerance)
        bi, bj = neighbor_list("ij", atoms, cutoffs)
        mask = bi < bj
        bi = bi[mask]
        n_bonds = bi.shape[0]

        n_all_pairs = n * (n - 1) // 2
        p_e[0] += max(0, n_all_pairs - n_bonds)  # no-bond pairs
        p_e[1] += n_bonds                          # all covalent = single

        n_atoms_counter[n] += 1
        ok += 1

    if ok == 0:
        raise RuntimeError("No molecules survived. Check src path / atom_map.")

    # Normalize.
    p_a_t = torch.tensor(p_a, dtype=torch.float32)
    p_a_t = p_a_t / p_a_t.sum()
    p_c_t = torch.tensor(p_c, dtype=torch.float32)
    p_c_t = p_c_t / p_c_t.sum()
    p_e_t = torch.tensor(p_e, dtype=torch.float32)
    p_e_t = p_e_t / p_e_t.sum()
    p_c_given_a_t = torch.tensor(p_c_given_a, dtype=torch.float32)
    p_c_given_a_t = p_c_given_a_t / p_c_given_a_t.sum(dim=1, keepdim=True).clamp(min=1.0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save((p_a_t, p_c_t, p_e_t, p_c_given_a_t),
               args.out_dir / "train_data_marginal_dists.pt")
    torch.save(torch.tensor(n_atoms_counter, dtype=torch.long),
               args.out_dir / "train_data_n_atoms_histogram.pt")

    print(f"\nkept {ok}/{sample_size} ({skipped} skipped, {element_miss} w/ unknown element)")
    print(f"p_a top 10 elements:")
    top = torch.topk(p_a_t, k=min(10, n_atom_types))
    for p, idx in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {atom_map[idx]:>3s}: {p:.4f}")
    print(f"p_e distribution: {p_e_t.tolist()}")
    print(f"n_atoms mean: {(n_atoms_counter * np.arange(args.max_atoms + 1)).sum() / max(1, n_atoms_counter.sum()):.1f}")
    if element_counts:
        print(f"top missing elements (Z -> count):")
        for z, c in sorted(element_counts.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  Z={z}: {c}")
    print(f"wrote stats to {args.out_dir}")

    # Save val index list (subset of the sampled molecules, deterministic).
    if args.val_size > 0:
        val_idx = sample_idx[:min(args.val_size, len(sample_idx))]
        torch.save(torch.tensor(val_idx, dtype=torch.long),
                   args.out_dir / "val_indices.pt")
        print(f"wrote val_indices.pt ({len(val_idx)} indices)")


if __name__ == "__main__":
    main()
