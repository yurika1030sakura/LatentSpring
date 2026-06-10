"""Stage 2 of uniform 5-cell ablation evaluation (in `envs/omol25`).

Reads the JSON dropped by Stage 1, runs OMol25 NN on each sample to get
energy + forces, then runs ASE BFGS relaxation to record relax energy
drop and step count. Outputs Stage 2 CSV + merged final eval CSV.

Usage:
  conda activate envs/omol25
  python scripts/eval_cell_omol25_stage2.py \\
      --cell_id 4d \\
      --samples_json runs/eval/cell_4d/4d_samples.json \\
      --stage1_csv runs/eval/cell_4d/4d_stage1.csv \\
      --out_csv runs/eval/cell_4d/4d_eval.csv \\
      --relax_steps 50
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
import time
from pathlib import Path

import numpy as np


CKPT_DEFAULT = (
    "/n/netscratch/ryl_lab/Lab/hf_cache/models--facebook--OMol25/"
    "snapshots/039b7070e59d1537e56c93a3a455263d062ed9c8/checkpoints/"
    "esen_sm_conserving_all.pt"
)


def _load_calc(ckpt: str, device: str = "cuda"):
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    pred = load_predict_unit(ckpt, device=device)
    return FAIRChemCalculator(pred)


def _eval_sample(s: dict, calc, relax_steps: int = 50, fmax: float = 1e-2) -> dict:
    import ase
    from ase.optimize import BFGS

    row = {
        "n_atoms": len(s.get("atomic_numbers", [])),
        "charge": int(s.get("charge", 0)),
        "spin": int(s.get("spin", 1)),
        "relax_steps_max": relax_steps,
    }
    try:
        atoms = ase.Atoms(numbers=s["atomic_numbers"],
                          positions=np.array(s["positions"], dtype=float))
        atoms.info["charge"] = row["charge"]
        atoms.info["spin"] = row["spin"]
        atoms.calc = calc
        t0 = time.time()
        e_raw = float(atoms.get_potential_energy())
        f_raw = atoms.get_forces()
        row["omol25_energy_eV"] = e_raw
        row["max_force_eVpA"] = float(np.abs(f_raw).max())
        row["mean_force_eVpA"] = float(np.sqrt((f_raw ** 2).sum(-1)).mean())
        row["raw_eval_ms"] = round((time.time() - t0) * 1000, 1)

        # BFGS relaxation
        if relax_steps > 0:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                opt = BFGS(atoms, logfile=buf)
                opt.run(fmax=fmax, steps=relax_steps)
            e_relaxed = float(atoms.get_potential_energy())
            f_relaxed = atoms.get_forces()
            row["omol25_energy_relaxed_eV"] = e_relaxed
            row["relax_energy_drop_eV"] = e_raw - e_relaxed
            row["max_force_relaxed_eVpA"] = float(np.abs(f_relaxed).max())
            row["relax_steps_used"] = int(opt.nsteps)
            row["relax_converged"] = bool(np.abs(f_relaxed).max() < fmax)
        row["ok"] = True
    except Exception as exc:
        row["ok"] = False
        row["error"] = f"{type(exc).__name__}: {str(exc)[:140]}"
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell_id", required=True)
    ap.add_argument("--samples_json", type=Path, required=True)
    ap.add_argument("--stage1_csv", type=Path, default=None,
                    help="Optional: stage1 CSV to merge connectedness metrics.")
    ap.add_argument("--out_csv", type=Path, required=True,
                    help="Final merged per-sample CSV.")
    ap.add_argument("--ckpt", type=str, default=CKPT_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--relax_steps", type=int, default=50)
    ap.add_argument("--fmax", type=float, default=1e-2)
    args = ap.parse_args()

    print(f"[stage2] cell_id={args.cell_id}", flush=True)
    print(f"[stage2] loading samples from {args.samples_json}", flush=True)
    with open(args.samples_json) as f:
        samples = json.load(f)
    print(f"  {len(samples)} samples", flush=True)

    print(f"[stage2] loading OMol25 ({args.device}) ...", flush=True)
    calc = _load_calc(args.ckpt, args.device)
    print("[stage2] OMol25 loaded", flush=True)

    rows = []
    for i, s in enumerate(samples):
        row = _eval_sample(s, calc, relax_steps=args.relax_steps, fmax=args.fmax)
        row["cell_id"] = args.cell_id
        row["mol_idx"] = i
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"[stage2] {i+1}/{len(samples)}", flush=True)

    # Merge with stage1 metrics if available
    if args.stage1_csv and args.stage1_csv.exists():
        with open(args.stage1_csv) as f:
            stage1 = {int(r["mol_idx"]): r for r in csv.DictReader(f)}
        for row in rows:
            s1 = stage1.get(int(row["mol_idx"]), {})
            for k, v in s1.items():
                if k not in row and k not in ("mol_idx",):
                    row[k] = v

    # Write merged CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        all_keys = sorted({k for r in rows for k in r})
        # Ensure cell_id and mol_idx come first
        ordered = ["cell_id", "mol_idx"] + [k for k in all_keys
                                             if k not in ("cell_id", "mol_idx")]
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ordered)
            w.writeheader()
            w.writerows(rows)
        print(f"[stage2] wrote {args.out_csv} ({len(rows)} rows)", flush=True)

    # Summary
    ok = [r for r in rows if r.get("ok")]
    if ok:
        import statistics as stats
        def _mean(key):
            xs = [r[key] for r in ok if key in r]
            return stats.mean(xs) if xs else None

        print(f"\n[stage2 summary]")
        print(f"  n_ok                       : {len(ok)}/{len(rows)}")
        print(f"  mean omol25_energy_eV      : {_mean('omol25_energy_eV')}")
        print(f"  mean max|F| eV/A           : {_mean('max_force_eVpA')}")
        print(f"  mean relaxed |F| eV/A      : {_mean('max_force_relaxed_eVpA')}")
        print(f"  mean relax energy drop eV  : {_mean('relax_energy_drop_eV')}")
        print(f"  mean relax steps used      : {_mean('relax_steps_used')}")
        n_conv = sum(1 for r in ok if r.get('relax_converged'))
        print(f"  fraction converged (fmax)  : {n_conv}/{len(ok)} = {n_conv/max(1,len(ok)):.3f}")
        n_conn = sum(1 for r in ok if str(r.get('connected', '')).lower() == 'true')
        if n_conn > 0:
            print(f"  fraction connected         : {n_conn}/{len(ok)} = {n_conn/max(1,len(ok)):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
