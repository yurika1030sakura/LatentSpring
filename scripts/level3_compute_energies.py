"""Level 3 Stage-2: compute OMol25 energy for each BGFM-sampled geometry.

The Level-3 BGFM sampler (level3_bgfm_sample.py) outputs npz files with
positions filled and energies = NaN. This script (omol25 env) fills in the
energies via OMol25, mirroring the Stage-1/Stage-2 split of the Boltzmann
correlation eval pipeline.

Usage (envs/omol25):
  conda activate envs/omol25
  python scripts/level3_compute_energies.py --in_dir runs/eval/level3_bgfm
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


CKPT_DEFAULT = (
    "/n/netscratch/ryl_lab/Lab/hf_cache/models--facebook--OMol25/"
    "snapshots/039b7070e59d1537e56c93a3a455263d062ed9c8/checkpoints/"
    "esen_sm_conserving_all.pt"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=Path, required=True,
                    help="Directory of level3_bgfm_*.npz from level3_bgfm_sample.py")
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import ase
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    calc = FAIRChemCalculator(load_predict_unit(args.ckpt, device=args.device))
    print("[lvl3-e] OMol25 loaded", flush=True)

    files = sorted(args.in_dir.glob("level3_bgfm_*.npz"))
    print(f"[lvl3-e] {len(files)} npz files to process", flush=True)

    for f in files:
        d = dict(np.load(f, allow_pickle=False))
        if not np.any(np.isnan(d["energies"])):
            print(f"[lvl3-e] {f.name}: all energies filled, skip", flush=True)
            continue
        pos = d["positions"]    # (N, n_atoms, 3)
        Z = d["atomic_numbers"]
        charge = int(d["charge"]) if "charge" in d else 0
        spin = int(d["spin"]) if "spin" in d else 1
        N = pos.shape[0]
        energies = d["energies"].copy()
        t0 = time.time()
        for i in range(N):
            if not np.isnan(energies[i]):
                continue
            try:
                atoms = ase.Atoms(numbers=Z, positions=pos[i].astype(np.float64))
                atoms.info["charge"] = charge
                atoms.info["spin"] = spin
                atoms.calc = calc
                energies[i] = float(atoms.get_potential_energy())
            except Exception as exc:
                print(f"  {f.name} sample {i}: {type(exc).__name__}: "
                      f"{str(exc)[:80]}", flush=True)
                energies[i] = np.nan
            if (i + 1) % 25 == 0:
                dt = time.time() - t0
                print(f"  {f.name}  {i+1}/{N}  ({dt:.0f}s)", flush=True)
        d["energies"] = energies.astype(np.float32)
        np.savez(f, **d)
        ok = int((~np.isnan(energies)).sum())
        print(f"[lvl3-e] {f.name}: {ok}/{N} energies filled, "
              f"E_mean={np.nanmean(energies):.2f} eV E_std={np.nanstd(energies):.2f} eV",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
