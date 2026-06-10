"""Level 3 reference: OMol25 + Langevin MD for Boltzmann ensemble ground truth.

For 3-5 small test molecules (15-40 atoms), run Langevin dynamics with the
OMol25 universal neural potential as the energy function, at temperature T.
Save N decorrelated geometries per molecule -- these are the ground-truth
Boltzmann samples that BGFM K-step generation will be compared against.

Why this is the right reference:
  Boltzmann distribution is defined operationally by the long-time limit of
  MD at temperature T under potential V. With the same NP that defines our
  L_energy target, MD samples are the unimpeachable ground truth. Any
  efficient sampler that claims "Boltzmann at temperature T" must match
  these statistics.

Why same NP as BGFM (not DFT):
  We want to isolate "did BGFM learn the conditional density p(x|m) for the
  potential it was trained against." Using DFT for reference would conflate
  "did BGFM learn what we asked it to" with "is the NP a good approximation
  of DFT" -- a separate question.

Output: <out>/level3_md_<mol_id>.npz with keys
  positions   : (N_samples, n_atoms, 3) float32
  energies    : (N_samples,) float32 eV
  step_indices: (N_samples,) int   -- which MD step each sample came from
  atomic_numbers: (n_atoms,)
  charge, spin, T_kelvin, dt_fs, friction, n_md_steps_total

Usage (envs/omol25):
  conda activate envs/omol25
  python scripts/level3_md_reference.py \\
      --eval_data /n/netscratch/.../val_data_processed.pt \\
      --atom_map_config configs/omol25_4m_bgfm_energy.yaml \\
      --mol_indices 0,7,42,100,2000 \\
      --n_md_steps 100000 --save_every 1000 \\
      --temperature 300 --dt_fs 0.5 --friction 0.01 \\
      --out_dir runs/eval/level3_md_reference
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


CKPT_DEFAULT = (
    "/n/netscratch/ryl_lab/Lab/hf_cache/models--facebook--OMol25/"
    "snapshots/039b7070e59d1537e56c93a3a455263d062ed9c8/checkpoints/"
    "esen_sm_conserving_all.pt"
)


def _load_calc(ckpt: str, device: str = "cuda"):
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    return FAIRChemCalculator(load_predict_unit(ckpt, device=device))


def _build_atoms(positions: np.ndarray, atom_type_idx: np.ndarray,
                 atom_charges: np.ndarray, atom_map: list[str],
                 calc) -> "ase.Atoms":
    import ase
    sym_to_Z = {sym: idx + 1 for idx, sym in enumerate([
        "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
        "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
        "Ga","Ge","As","Se","Br","Kr",
        "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
        "In","Sn","Sb","Te","I","Xe",
        "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy",
        "Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt",
        "Au","Hg","Tl","Pb","Bi",
    ])}
    Z = np.array([sym_to_Z[atom_map[int(i)]] for i in atom_type_idx])
    atoms = ase.Atoms(numbers=Z, positions=positions.astype(np.float64))
    atoms.info["charge"] = int(atom_charges[0]) if atom_charges.size > 0 else 0
    atoms.info["spin"] = 1
    atoms.calc = calc
    return atoms


def _run_langevin(atoms, T_kelvin: float, dt_fs: float, friction: float,
                  n_md_steps: int, save_every: int,
                  log_every: int = 5000) -> tuple[list[np.ndarray], list[float], list[int]]:
    """Run Langevin MD, save geometry + energy every save_every steps.

    Energy is the potential energy E (not E + KE). For Boltzmann analysis we
    want p(x) marginalized over momentum, which depends only on V(x).
    """
    import ase
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    MaxwellBoltzmannDistribution(atoms, temperature_K=T_kelvin)
    dyn = Langevin(
        atoms,
        timestep=dt_fs * units.fs,
        temperature_K=T_kelvin,
        friction=friction / units.fs,    # friction has units of 1/time
    )

    positions_buf = []
    energies_buf = []
    step_buf = []

    t0 = time.time()
    # Initial sample (step 0)
    positions_buf.append(atoms.get_positions().astype(np.float32))
    energies_buf.append(float(atoms.get_potential_energy()))
    step_buf.append(0)

    for step in range(1, n_md_steps + 1):
        dyn.run(1)
        if step % save_every == 0:
            positions_buf.append(atoms.get_positions().astype(np.float32))
            energies_buf.append(float(atoms.get_potential_energy()))
            step_buf.append(step)
        if step % log_every == 0:
            dt = time.time() - t0
            print(f"    step {step:>7d}/{n_md_steps}  E={energies_buf[-1]:.3f} eV  "
                  f"({dt/60:.1f} min, {step/dt:.0f} steps/s, "
                  f"eta {(n_md_steps-step)/(step/dt)/60:.1f} min)", flush=True)

    return positions_buf, energies_buf, step_buf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_data", type=Path, required=True,
                    help="val_data_processed.pt (source of test molecules)")
    ap.add_argument("--atom_map_config", type=Path, required=True)
    ap.add_argument("--mol_indices", required=True,
                    help="Comma-sep indices into val shard (e.g., 0,7,42)")
    ap.add_argument("--n_md_steps", type=int, default=100000,
                    help="Total Langevin steps. 100k @ 0.5 fs = 50 ps.")
    ap.add_argument("--save_every", type=int, default=1000,
                    help="Save every N steps (decorrelation interval).")
    ap.add_argument("--temperature", type=float, default=300.0,
                    help="Target temperature in K")
    ap.add_argument("--dt_fs", type=float, default=0.5,
                    help="MD timestep in femtoseconds")
    ap.add_argument("--friction", type=float, default=0.01,
                    help="Langevin friction in 1/fs (typical 0.01)")
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_atoms_filter", type=int, default=60,
                    help="Skip molecules with more atoms than this (MD too slow)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    print(f"[lvl3-md] loading val shard: {args.eval_data}", flush=True)
    src = torch.load(args.eval_data, map_location="cpu", weights_only=False)
    positions = src["positions"].numpy()
    atom_types = src["atom_types"].numpy().astype(np.int64)
    atom_charges = src["atom_charges"].numpy().astype(np.int64)
    nia = src["node_idx_array"].numpy()
    M_total = nia.shape[0]
    print(f"[lvl3-md] val shard has M={M_total} molecules", flush=True)

    with open(args.atom_map_config) as f:
        cfg = yaml.safe_load(f)
    atom_map = cfg["dataset"]["atom_map"]

    indices = [int(x) for x in args.mol_indices.split(",")]
    print(f"[lvl3-md] target mol indices: {indices}", flush=True)

    calc = _load_calc(args.ckpt, args.device)
    print("[lvl3-md] OMol25 loaded", flush=True)

    for mi in indices:
        if mi >= M_total:
            print(f"[lvl3-md] mol_idx {mi} >= M={M_total}, skip", flush=True)
            continue
        s, e = int(nia[mi, 0]), int(nia[mi, 1])
        n_atoms = e - s
        if n_atoms > args.max_atoms_filter:
            print(f"[lvl3-md] mol_idx {mi} has {n_atoms} atoms > "
                  f"{args.max_atoms_filter}, skip (MD too slow)", flush=True)
            continue

        pos0 = positions[s:e].astype(np.float64)
        at_idx = atom_types[s:e]
        ac = atom_charges[s:e]
        atoms = _build_atoms(pos0, at_idx, ac, atom_map, calc)

        out_path = args.out_dir / f"level3_md_mol{mi:05d}.npz"
        if out_path.exists():
            print(f"[lvl3-md] mol_idx {mi} already done at {out_path}, skip", flush=True)
            continue

        print(f"\n[lvl3-md] === mol_idx {mi}: {n_atoms} atoms ===", flush=True)
        try:
            pos_list, energies, step_list = _run_langevin(
                atoms, T_kelvin=args.temperature, dt_fs=args.dt_fs,
                friction=args.friction, n_md_steps=args.n_md_steps,
                save_every=args.save_every,
            )
        except Exception as exc:
            print(f"[lvl3-md] mol_idx {mi} failed: {type(exc).__name__}: {exc}",
                  flush=True)
            continue

        sym_to_Z = {sym: idx + 1 for idx, sym in enumerate([
            "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
            "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
            "Ga","Ge","As","Se","Br","Kr",
            "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
            "In","Sn","Sb","Te","I","Xe",
            "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy",
            "Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt",
            "Au","Hg","Tl","Pb","Bi",
        ])}
        Z = np.array([sym_to_Z[atom_map[int(i)]] for i in at_idx], dtype=np.int64)
        np.savez(
            out_path,
            positions=np.array(pos_list, dtype=np.float32),
            energies=np.array(energies, dtype=np.float32),
            step_indices=np.array(step_list, dtype=np.int64),
            atomic_numbers=Z,
            charge=int(ac[0]) if ac.size > 0 else 0,
            spin=1,
            T_kelvin=args.temperature,
            dt_fs=args.dt_fs,
            friction=args.friction,
            n_md_steps_total=args.n_md_steps,
        )
        print(f"[lvl3-md] wrote {out_path} ({len(pos_list)} samples)", flush=True)

    print("\n[lvl3-md] all molecules done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
