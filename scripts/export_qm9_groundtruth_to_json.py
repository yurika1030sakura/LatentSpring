"""Export QM9 ground-truth molecules (from processed data) to the JSON
format consumed by `compute_omol25_energy.py`.

This lets us sanity-check whether FlowMol3-sampled OMol25 energies match
ground-truth QM9 OMol25 energies. If they match, our "refinement"
improvement is a DFT-level artifact (B3LYP->ωB97M-V swap). If FlowMol3
samples are significantly higher, our refinement is genuine.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch


ATOM_MAP_QM9 = ["C", "H", "N", "O", "F"]
_SYMBOL_TO_Z = {"C": 6, "H": 1, "N": 7, "O": 8, "F": 9}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_data", type=Path, required=True)
    ap.add_argument("--n_samples", type=int, default=300)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"loading {args.processed_data} ...")
    data = torch.load(args.processed_data)
    positions = data["positions"]              # (N_total, 3)
    atom_types = data["atom_types"]            # (N_total, A) one-hot or similar
    node_idx = data["node_idx_array"]          # (n_mols, 2)
    n_mols = node_idx.shape[0]
    print(f"total mols: {n_mols}")

    if atom_types.dim() == 2:
        # One-hot or simplex -> argmax (cast to float first for bool tensors)
        atom_types_idx = atom_types.float().argmax(dim=-1)
    else:
        atom_types_idx = atom_types.long()

    # Random sample
    random.seed(args.seed)
    sel = random.sample(range(n_mols), min(args.n_samples, n_mols))

    samples: list[dict] = []
    for mol_idx in sel:
        s, e = int(node_idx[mol_idx, 0]), int(node_idx[mol_idx, 1])
        pos = positions[s:e].tolist()           # (n, 3)
        types = atom_types_idx[s:e].tolist()    # (n,)
        # Only take atoms with type in the known ATOM_MAP (<=4 for QM9)
        if any(t >= len(ATOM_MAP_QM9) for t in types):
            continue
        numbers = [_SYMBOL_TO_Z[ATOM_MAP_QM9[int(t)]] for t in types]
        samples.append({
            "atomic_numbers": numbers,
            "positions": pos,
            "charge": 0,
            "spin": 1,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(samples, f)
    print(f"wrote {len(samples)} GT samples to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
