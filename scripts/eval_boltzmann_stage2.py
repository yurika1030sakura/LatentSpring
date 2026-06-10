"""Boltzmann-correlation eval, Stage 2 (envs/omol25): energies + correlation.

Reads Stage-1 JSON (per-molecule perturbations with log_p_theta), computes
OMol25 energy for each perturbed geometry, then measures whether
log p_theta tracks -E/kT within each molecule group. This is the direct
Boltzmann-consistency test:

    Boltzmann  =>  log p_theta = -E/kT + const  (per molecule).

Reports:
  - per-group Pearson r and slope of (log p_theta vs -E/kT)
  - aggregate mean/median r, fraction of groups with r > 0.5
  - global pooled R^2 (after per-group mean-centering, removing the size
    confound) -- this is the headline "Boltzmann alignment" number

Usage:
  conda activate envs/omol25
  python scripts/eval_boltzmann_stage2.py \\
      --samples_json runs/eval/boltzmann/<tag>/boltzmann_samples.json \\
      --out_csv runs/eval/boltzmann/<tag>/boltzmann_correlation.csv \\
      --kT 1.0
"""
from __future__ import annotations

import argparse
import csv
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


def _load_calc(ckpt, device="cuda"):
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    return FAIRChemCalculator(load_predict_unit(ckpt, device=device))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_json", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--kT", type=float, default=1.0)
    args = ap.parse_args()

    import ase

    with open(args.samples_json) as f:
        records = json.load(f)
    print(f"[boltz2] {len(records)} records", flush=True)

    calc = _load_calc(args.ckpt, args.device)
    print("[boltz2] OMol25 loaded", flush=True)

    # Compute energy for each geometry.
    t0 = time.time()
    for i, r in enumerate(records):
        try:
            atoms = ase.Atoms(numbers=r["atomic_numbers"],
                              positions=np.array(r["positions"], dtype=float))
            atoms.info["charge"] = int(r.get("charge", 0))
            atoms.info["spin"] = int(r.get("spin", 1))
            atoms.calc = calc
            r["energy_eV"] = float(atoms.get_potential_energy())
            r["ok"] = True
        except Exception as exc:
            r["ok"] = False
            r["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        if (i + 1) % 100 == 0:
            print(f"[boltz2] {i+1}/{len(records)}  ({(time.time()-t0)/60:.1f} min)", flush=True)

    # Group by molecule and correlate within each group.
    groups: dict[int, list[dict]] = {}
    for r in records:
        if r.get("ok"):
            groups.setdefault(r["group_id"], []).append(r)

    per_group_rows = []
    pooled_x = []  # -E/kT, per-group mean-centered
    pooled_y = []  # log p_theta, per-group mean-centered
    for gid, recs in groups.items():
        if len(recs) < 3:
            continue
        logp = np.array([r["log_p_theta"] for r in recs])
        negE_kT = -np.array([r["energy_eV"] for r in recs]) / args.kT
        # Pearson r within group
        if logp.std() < 1e-9 or negE_kT.std() < 1e-9:
            continue
        r_pearson = float(np.corrcoef(logp, negE_kT)[0, 1])
        slope = float(np.polyfit(negE_kT, logp, 1)[0])
        per_group_rows.append({
            "group_id": gid, "n_pert": len(recs),
            "pearson_r": r_pearson, "slope": slope,
            "logp_std": float(logp.std()),
            "negE_kT_std": float(negE_kT.std()),
        })
        # Mean-center for pooled R^2 (removes per-molecule size offset)
        pooled_x.extend((negE_kT - negE_kT.mean()).tolist())
        pooled_y.extend((logp - logp.mean()).tolist())

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group_id", "n_pert", "pearson_r",
                                          "slope", "logp_std", "negE_kT_std"])
        w.writeheader()
        w.writerows(per_group_rows)

    # Aggregate
    rs = np.array([row["pearson_r"] for row in per_group_rows])
    slopes = np.array([row["slope"] for row in per_group_rows])
    px, py = np.array(pooled_x), np.array(pooled_y)
    pooled_r2 = float(np.corrcoef(px, py)[0, 1] ** 2) if px.size > 2 else float("nan")

    print(f"\n[boltz2 SUMMARY]  (Boltzmann consistency test)")
    print(f"  groups analyzed        : {len(per_group_rows)}")
    print(f"  mean per-group Pearson r: {rs.mean():.3f}  (median {np.median(rs):.3f})")
    print(f"  frac groups r>0.5      : {(rs > 0.5).mean():.3f}")
    print(f"  frac groups r>0.8      : {(rs > 0.8).mean():.3f}")
    print(f"  mean slope (ideal=1)   : {slopes.mean():.3f}")
    print(f"  POOLED R^2 (centered)  : {pooled_r2:.3f}   <-- headline Boltzmann-alignment")
    print(f"\n  Interpretation:")
    print(f"    r ~ 1, slope ~ 1, pooled R^2 high  =>  learned density IS Boltzmann.")
    print(f"    r ~ 0                              =>  density unrelated to energy (not Boltzmann).")
    print(f"\n[boltz2] wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
