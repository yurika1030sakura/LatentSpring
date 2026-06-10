"""Annotate GEOM-Drugs processed tensors with OMol25 NN forces + energies.

Cell B of Paper 1 ablation matrix: enables BGFM training on GEOM data with
the same physics labels (force, energy via OMol25 NN) used in our main
OMol25 experiments. This isolates the *method* contribution (BGFM 3-term
loss) from the *data* contribution (OMol25 vs GEOM training corpus).

Run in `envs/omol25` (torch 2.8 + fairchem-core 2.19), GPU recommended.

Output schema: same as preprocess_omol25.py output --
  - existing GEOM fields (positions, atom_types, atom_charges, bond_*, *_idx_array)
  - NEW: forces (N_total, 3) float32, OMol25 NN-predicted forces (eV/A)
  - NEW: energies (M,) float32, OMol25 NN-predicted energies (eV)

Usage:
  conda activate envs/omol25
  python scripts/annotate_geom_with_omol25.py \\
      --src_dir /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/baselines/flowmol3/data/geom_5_kekulized \\
      --out_dir /n/netscratch/ryl_lab/Lab/yulili_cfm_mol/geom_bgfm_processed \\
      --device cuda
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch


# GEOM-Drugs atom map (10 organic elements)
GEOM_ATOM_MAP = ["C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
SYM_TO_Z = {"C": 6, "H": 1, "N": 7, "O": 8, "F": 9, "P": 15,
            "S": 16, "Cl": 17, "Br": 35, "I": 53}


def _load_omol25(ckpt: str, device: str = "cuda"):
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    pred = load_predict_unit(ckpt, device=device)
    return FAIRChemCalculator(pred)


def annotate_split(src_path: Path, out_path: Path, calc, atom_map: list[str]):
    """Add forces + energies to a single GEOM split tensor file."""
    import ase

    print(f"[load] {src_path}", flush=True)
    data = torch.load(src_path)
    positions = data["positions"]
    atom_types = data["atom_types"]
    node_idx_array = data["node_idx_array"]
    M = node_idx_array.shape[0]
    N_atoms = positions.shape[0]
    print(f"  {M} molecules, {N_atoms} atoms total", flush=True)

    # Decode atom types: support both 2D one-hot and 1D int8 storage formats.
    if atom_types.dim() == 2:
        at_idx = atom_types.long().argmax(dim=-1)
    else:
        at_idx = atom_types.long()
    z_per_type = torch.tensor([SYM_TO_Z[s] for s in atom_map], dtype=torch.long)
    z_per_atom = z_per_type[at_idx]   # (N_atoms,)

    forces_out = torch.zeros((N_atoms, 3), dtype=torch.float32)
    energies_out = torch.zeros((M,), dtype=torch.float32)
    n_failed = 0

    t0 = time.time()
    last_log = t0
    for mi in range(M):
        if time.time() - last_log > 30:
            elapsed = time.time() - t0
            rate = (mi + 1) / elapsed
            eta = (M - mi - 1) / max(rate, 1e-6)
            print(f"  {mi+1:>7d}/{M} fail={n_failed} rate={rate:.1f} mol/s "
                  f"eta={eta/60:.1f} min", flush=True)
            last_log = time.time()
        try:
            start = int(node_idx_array[mi, 0])
            end = int(node_idx_array[mi, 1])
            n = end - start
            pos = positions[start:end].numpy().astype(float)
            z = z_per_atom[start:end].numpy().astype(int)

            atoms = ase.Atoms(numbers=z, positions=pos)
            atoms.info["charge"] = 0
            atoms.info["spin"] = 1
            atoms.calc = calc
            E = float(atoms.get_potential_energy())
            F = atoms.get_forces().astype(np.float32)
            if F.shape != (n, 3):
                raise ValueError(f"force shape mismatch: {F.shape}")
            energies_out[mi] = E
            forces_out[start:end] = torch.from_numpy(F)
        except Exception as exc:
            n_failed += 1
            if n_failed < 5:
                print(f"  [warn] mol {mi} failed: {type(exc).__name__}: {exc}",
                      flush=True)
            # Leave forces/energy at zero for failed molecules.
            continue

    print(f"\n[done] {M-n_failed}/{M} succeeded ({n_failed} failed) in "
          f"{(time.time()-t0)/60:.1f} min", flush=True)

    # Write output (same dict + new fields).
    data["forces"] = forces_out
    data["energies"] = energies_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    print(f"[save] {out_path} ({out_path.stat().st_size / 1e9:.1f} GB)",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", type=Path, required=True,
                    help="Directory with train_data_processed.pt etc.")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=str,
                    default="/n/netscratch/ryl_lab/Lab/hf_cache/models--facebook--OMol25/"
                            "snapshots/039b7070e59d1537e56c93a3a455263d062ed9c8/checkpoints/"
                            "esen_sm_conserving_all.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    print(f"loading OMol25 from {args.ckpt} on {args.device} ...", flush=True)
    calc = _load_omol25(args.ckpt, args.device)
    print("OMol25 loaded", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Copy / replicate non-tensor files (marginal_dists, n_atoms_hist, valencies).
    import shutil
    for fname in ("train_data_marginal_dists.pt",
                  "train_data_n_atoms_histogram.pt"):
        src = args.src_dir / fname
        if src.exists():
            shutil.copy2(src, args.out_dir / fname)
            print(f"copied {fname}", flush=True)
    for valency_glob in args.src_dir.glob("train_data_valencies_*.json"):
        shutil.copy2(valency_glob, args.out_dir / valency_glob.name)
        print(f"copied {valency_glob.name}", flush=True)

    for split in args.splits:
        src_path = args.src_dir / f"{split}_data_processed.pt"
        out_path = args.out_dir / f"{split}_data_processed.pt"
        if not src_path.exists():
            print(f"[skip] {split}: source not found at {src_path}", flush=True)
            continue
        if out_path.exists():
            print(f"[skip] {split}: output already exists at {out_path}", flush=True)
            continue
        annotate_split(src_path, out_path, calc, GEOM_ATOM_MAP)


if __name__ == "__main__":
    main()
