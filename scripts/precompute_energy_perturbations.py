"""Pre-compute K-perturbation tuples + OMol25 energies per molecule.

Why this script exists:
  L_energy = Var(log p_θ + E/kT) is a *per-molecule* Boltzmann condition: for
  geometries x_1, ..., x_K of the SAME molecule, log p + E/kT should be the
  same constant (-log Z_m). The variance of (log p + E/kT) across DIFFERENT
  molecules picks up the per-molecule log Z spread, which is dominated by
  size effects and has nothing to do with whether each molecule's density
  is Boltzmann.

  Our processed dataset has one geometry per molecule, so within-molecule
  variance literally cannot be computed at train time. This script generates
  K small perturbations per molecule and runs OMol25 to get their DFT-quality
  energies, producing a "perturbation shard" the training loss can consume.

Output: <out>/perturbation_<split>_<n>.pt
  dict with keys:
    'positions'      : (N_total_K, 3)   K*M molecules' atom coords, flat
    'atom_types'     : (N_total_K,)     int8 atom type indices
    'atom_charges'   : (N_total_K,)     int8
    'node_idx_array' : (M*K, 2)         (start, end) per virtual molecule
    'energies'       : (M*K,)           OMol25 energy in eV per virtual mol
    'group_id'       : (M*K,)           int64 parent molecule id (0..M-1).
                                        K consecutive entries share group_id.
  Also storage convention: virtual molecules are stored in K-contiguous
  blocks (block 0 = mol 0 pert 0..K-1, block 1 = mol 1 pert 0..K-1, ...).
  So the dataset loader can fetch K perturbations of one molecule by reading
  K consecutive entries.

Usage (envs/omol25):
  conda activate envs/omol25
  python scripts/precompute_energy_perturbations.py \\
      --src /n/netscratch/ryl_lab/Lab/yulili_cfm_mol/omol25_4m_processed/val_data_processed.pt \\
      --out /n/netscratch/ryl_lab/Lab/yulili_cfm_mol/omol25_4m_processed/perturbation_val.pt \\
      --atom_map_config configs/omol25_4m_bgfm.yaml \\
      --n_molecules 10000 --sigmas 0.05,0.10,0.20,0.40
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def _load_calc(ckpt: str, device: str = "cuda"):
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    return FAIRChemCalculator(load_predict_unit(ckpt, device=device))


CKPT_DEFAULT = (
    "/n/netscratch/ryl_lab/Lab/hf_cache/models--facebook--OMol25/"
    "snapshots/039b7070e59d1537e56c93a3a455263d062ed9c8/checkpoints/"
    "esen_sm_conserving_all.pt"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True,
                    help="Processed shard: val_data_processed.pt or train_data_processed.pt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--atom_map_config", type=Path, required=True,
                    help="Config YAML containing dataset.atom_map (idx -> symbol)")
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_molecules", type=int, default=10000,
                    help="Number of base molecules to perturb")
    ap.add_argument("--sigmas", default="0.05,0.10,0.20,0.40",
                    help="Comma-sep perturbation std (Angstrom); K=number of sigmas")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start_index", type=int, default=0,
                    help="Skip the first N base molecules (for sharding/resume)")
    ap.add_argument("--charge_atom0_default", type=int, default=0,
                    help="Read by atom-0 convention; usually 0 (neutral)")
    args = ap.parse_args()

    import ase

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"[pert] loading processed shard: {args.src}", flush=True)
    src = torch.load(args.src, map_location="cpu", weights_only=False)
    positions = src["positions"].numpy()             # (N_total, 3) float32
    atom_types = src["atom_types"].numpy().astype(np.int64)   # (N_total,)
    atom_charges = src["atom_charges"].numpy().astype(np.int64)
    nia = src["node_idx_array"].numpy()              # (M, 2)
    M_total = nia.shape[0]
    print(f"[pert] base molecules in shard: M_total={M_total}", flush=True)

    # Load atom_map (idx -> element symbol).
    with open(args.atom_map_config) as f:
        cfg = yaml.safe_load(f)
    atom_map = cfg["dataset"]["atom_map"]
    sym_to_Z = {
        sym: idx + 1 for idx, sym in enumerate([
            "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
            "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
            "Ga","Ge","As","Se","Br","Kr",
            "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
            "In","Sn","Sb","Te","I","Xe",
            "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy",
            "Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt",
            "Au","Hg","Tl","Pb","Bi",
        ])
    }
    type_to_Z = np.array([sym_to_Z[atom_map[i]] for i in range(len(atom_map))], dtype=np.int64)

    sigmas = [float(x) for x in args.sigmas.split(",")]
    K = len(sigmas)
    M_take = min(args.n_molecules, M_total - args.start_index)
    print(f"[pert] K={K} sigmas={sigmas}  taking M={M_take} base molecules "
          f"(start_index={args.start_index}) -> M*K={M_take*K} virtual molecules", flush=True)

    calc = _load_calc(args.ckpt, args.device)
    print("[pert] OMol25 loaded", flush=True)

    # Output buffers (sized incrementally for safety)
    pos_buf, at_buf, ac_buf = [], [], []
    nia_buf = []                # (M_take*K, 2)
    energies_buf = []           # (M_take*K,)
    group_id_buf = []           # (M_take*K,)
    running_N = 0
    skipped = 0
    t0 = time.time()

    for mi in range(M_take):
        bi = args.start_index + mi
        s, e = int(nia[bi, 0]), int(nia[bi, 1])
        base_pos = positions[s:e].astype(np.float64)      # (n, 3)
        base_at = atom_types[s:e]                          # (n,)
        base_ac = atom_charges[s:e]                        # (n,)
        n = base_pos.shape[0]
        if n < 2:
            skipped += 1
            continue
        Z = type_to_Z[base_at]
        # Atomic charge / spin: same convention as preprocess (atom 0 carries total charge).
        total_charge = int(base_ac[0]) if base_ac.size > 0 else args.charge_atom0_default

        for ki, sig in enumerate(sigmas):
            noise = rng.normal(scale=sig, size=base_pos.shape).astype(np.float64)
            pos_k = base_pos + noise
            pos_k = pos_k - pos_k.mean(axis=0, keepdims=True)
            try:
                atoms = ase.Atoms(numbers=Z, positions=pos_k)
                atoms.info["charge"] = total_charge
                atoms.info["spin"] = 1
                atoms.calc = calc
                E = float(atoms.get_potential_energy())
            except Exception as exc:
                print(f"[pert] mol={bi} sig={sig}: {type(exc).__name__}: {str(exc)[:80]}", flush=True)
                # Use a sentinel; we'll skip these in the loss later.
                E = float("nan")

            pos_buf.append(pos_k.astype(np.float32))
            at_buf.append(base_at.astype(np.int8))
            ac_buf.append(base_ac.astype(np.int8))
            nia_buf.append((running_N, running_N + n))
            energies_buf.append(E)
            group_id_buf.append(mi)
            running_N += n

        if (mi + 1) % 50 == 0:
            dt = time.time() - t0
            rate = (mi + 1) / max(dt, 1e-6)
            eta = (M_take - mi - 1) / max(rate, 1e-6)
            print(f"[pert] {mi+1:>6d}/{M_take}  ({dt/60:.1f} min, {rate:.2f} mol/s, "
                  f"eta {eta/60:.1f} min)  skipped={skipped}", flush=True)

    out = {
        "positions": torch.from_numpy(np.concatenate(pos_buf, axis=0)),
        "atom_types": torch.from_numpy(np.concatenate(at_buf, axis=0)),
        "atom_charges": torch.from_numpy(np.concatenate(ac_buf, axis=0)),
        "node_idx_array": torch.tensor(nia_buf, dtype=torch.int64),
        "energies": torch.tensor(energies_buf, dtype=torch.float32),
        "group_id": torch.tensor(group_id_buf, dtype=torch.int64),
        "K": K,
        "sigmas": sigmas,
    }
    torch.save(out, args.out)
    n_real = int((~torch.isnan(out["energies"])).sum().item())
    print(f"[pert] wrote {args.out} ({len(energies_buf)} virtual mols, "
          f"{n_real} with valid energy)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
