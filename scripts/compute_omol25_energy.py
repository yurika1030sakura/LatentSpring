"""Compute OMol25 energy for a set of molecules (post-hoc evaluation).

Input: path to a directory of per-sample .xyz files (or a JSON with list of
{atomic_numbers, positions, charge, spin} dicts).
Output: JSON/CSV with energy per sample + summary stats (mean, std,
divergence vs reference).

Run from the dedicated OMol25 env:
    conda activate envs/omol25
    python scripts/compute_omol25_energy.py \\
        --samples_json runs/eval/v4_samples.json \\
        --out runs/eval/v4_energies.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


CKPT_DEFAULT = (
    "/n/netscratch/ryl_lab/Lab/hf_cache/models--facebook--OMol25/"
    "snapshots/039b7070e59d1537e56c93a3a455263d062ed9c8/checkpoints/"
    "esen_sm_conserving_all.pt"
)


def _load_calc(ckpt: str, device: str = "cpu"):
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    pred = load_predict_unit(ckpt, device=device)
    return FAIRChemCalculator(pred)


def _evaluate_samples(
    samples: list[dict],
    calc,
    device: str = "cpu",
    relax_steps: int = 0,
    relax_lr: float = 0.05,
) -> list[dict]:
    """Evaluate OMol25 energy for each sample. If relax_steps > 0, additionally
    run post-hoc gradient descent (OMol25 force-guided) for that many steps,
    simulating a "drift refinement" that our paper's physics-informed
    sampler would do inline. Records energy before AND after refinement.
    """
    import ase
    import numpy as np
    import time

    out: list[dict] = []
    for i, s in enumerate(samples):
        row: dict = {
            "idx": i,
            "n_atoms": len(s.get("atomic_numbers", [])),
            "charge": int(s.get("charge", 0)),
            "spin": int(s.get("spin", 1)),
            "relax_steps": relax_steps,
        }
        try:
            atoms = ase.Atoms(
                numbers=s["atomic_numbers"],
                positions=np.array(s["positions"], dtype=float),
            )
            atoms.info["charge"] = row["charge"]
            atoms.info["spin"] = row["spin"]
            atoms.calc = calc
            t0 = time.time()
            e_raw = atoms.get_potential_energy()
            f_raw = atoms.get_forces()
            row["energy_eV_raw"] = float(e_raw)
            row["max_force_raw_eVpA"] = float(np.abs(f_raw).max())
            row["mean_force_raw_eVpA"] = float(np.sqrt((f_raw**2).sum(-1)).mean())
            row["compute_time_ms"] = round((time.time() - t0) * 1000, 1)

            if relax_steps > 0:
                # Use ASE's BFGS optimizer — adaptive step size, guaranteed
                # to monotonically decrease energy (modulo numerical noise).
                # Simulates an inline "drift refinement" with line-search.
                from ase.optimize import BFGS
                import io, contextlib
                logbuf = io.StringIO()
                with contextlib.redirect_stdout(logbuf):
                    opt = BFGS(atoms, logfile=logbuf)
                    opt.run(fmax=0.01, steps=relax_steps)
                e_relaxed = atoms.get_potential_energy()
                f_relaxed = atoms.get_forces()
                row["energy_eV_relaxed"] = float(e_relaxed)
                row["delta_eV"] = float(e_raw - e_relaxed)    # positive = relaxation lowered E
                row["max_force_final_eVpA"] = float(np.abs(f_relaxed).max())
                row["n_relax_steps_used"] = int(opt.nsteps)
                row["converged_to_1e-2"] = bool(np.abs(f_relaxed).max() < 0.01)
            row["ok"] = True
        except Exception as exc:
            row["ok"] = False
            row["error"] = f"{type(exc).__name__}: {str(exc)[:140]}"
        out.append(row)
        if (i + 1) % 10 == 0:
            print(f"  processed {i+1}/{len(samples)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_json", type=Path, required=True,
                    help="JSON: list of dicts with atomic_numbers, positions, charge, spin")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", type=str, default=CKPT_DEFAULT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--relax_steps", type=int, default=0,
                    help="If >0, run post-hoc OMol25 gradient descent on "
                         "each sample. Records energy before + after.")
    ap.add_argument("--relax_lr", type=float, default=0.05)
    args = ap.parse_args()

    print(f"loading samples from {args.samples_json} ...")
    with open(args.samples_json) as f:
        samples = json.load(f)
    print(f"  {len(samples)} samples")

    print(f"loading OMol25 model from {args.ckpt} ...")
    calc = _load_calc(args.ckpt, device=args.device)
    print("  OMol25 loaded")

    print(f"computing energies (relax_steps={args.relax_steps}) ...")
    results = _evaluate_samples(
        samples, calc, device=args.device,
        relax_steps=args.relax_steps, relax_lr=args.relax_lr,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in results for k in r}))
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {args.out}")

    # Summary
    import statistics as stats
    raw = [r["energy_eV_raw"] for r in results if r.get("ok")]
    if raw:
        print()
        print(f"Summary: {len(raw)}/{len(results)} samples valid")
        print(f"  raw energy    : mean {stats.mean(raw):.3f} eV  "
              f"(stdev {stats.stdev(raw) if len(raw)>1 else 0:.3f})")
    relaxed = [r.get("energy_eV_relaxed") for r in results if r.get("ok") and r.get("energy_eV_relaxed") is not None]
    deltas = [r.get("delta_eV") for r in results if r.get("ok") and r.get("delta_eV") is not None]
    if relaxed:
        print(f"  relaxed energy: mean {stats.mean(relaxed):.3f} eV  "
              f"(stdev {stats.stdev(relaxed) if len(relaxed)>1 else 0:.3f})")
    if deltas:
        print(f"  delta (raw - relaxed): mean {stats.mean(deltas):.3f} eV  "
              f"(median {stats.median(deltas):.3f})")
        print(f"    => OMol25 post-hoc drift lowered energy by "
              f"{stats.mean(deltas):.2f} eV on average (larger = more initial strain)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
