"""Level 2 money-plot: DFT relax-steps comparison BGFM vs FM-only.

For each generated geometry, run BFGS optimization with OMol25 as the energy
function until max |force| < target. Report the number of BFGS steps needed.
A more Boltzmann-aligned generator (BGFM) starts closer to local minima of
V_OMol25, so its samples should relax FASTER (fewer steps) than FM-only.

Why this matters for ICLR:
  Boltzmann correlation R^2 alone is a metric, easy to dismiss. "Saved DFT
  steps to convergence" is a downstream practical-deployment number:
  generators in the wild are almost always followed by relaxation, and any
  start-step savings translates directly into compute savings for
  downstream users. This is the kind of usefulness story ICLR orals select
  for.

Procedure:
  - Sample N geometries each from BGFM and a same-architecture FM baseline
  - Run ASE BFGS with OMol25 calculator on each, max_steps cap
  - Track: n_steps_to_converged, final_max_force, converged?
  - Report mean / median steps, fraction converged in N steps

Output: <out_dir>/level2_summary.csv  +  per-model relax traces in npz

Usage (envs/omol25):
  conda activate envs/omol25
  python scripts/level2_relax_comparison.py \\
      --bgfm_samples runs/eval/level2/bgfm/samples.npz \\
      --fm_samples   runs/eval/level2/fm/samples.npz \\
      --force_tol 0.05 --max_steps 200 \\
      --out_dir runs/eval/level2/relax
"""
from __future__ import annotations

import argparse
import csv
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
    return FAIRChemCalculator(load_predict_unit(ckpt, device=device))


def _load_samples(path: Path) -> list[dict]:
    """Either a directory of per-mol npz files or a single npz with arrays."""
    if path.is_dir():
        out = []
        for f in sorted(path.glob("*.npz")):
            d = dict(np.load(f, allow_pickle=False))
            n = d["positions"].shape[0]
            Z = d["atomic_numbers"]
            charge = int(d.get("charge", 0))
            spin = int(d.get("spin", 1))
            for i in range(n):
                out.append({
                    "positions": d["positions"][i],
                    "atomic_numbers": Z,
                    "charge": charge, "spin": spin,
                    "source_file": f.name, "sample_idx": i,
                })
        return out
    d = dict(np.load(path, allow_pickle=False))
    out = []
    Z = d["atomic_numbers"]
    n = d["positions"].shape[0]
    for i in range(n):
        out.append({
            "positions": d["positions"][i],
            "atomic_numbers": Z,
            "charge": int(d.get("charge", 0)),
            "spin": int(d.get("spin", 1)),
            "source_file": path.name, "sample_idx": i,
        })
    return out


def _relax_one(sample: dict, calc, force_tol: float, max_steps: int) -> dict:
    """Run BFGS to F_max < force_tol or max_steps."""
    import ase
    from ase.optimize import BFGS
    atoms = ase.Atoms(
        numbers=sample["atomic_numbers"],
        positions=sample["positions"].astype(np.float64),
    )
    atoms.info["charge"] = sample["charge"]
    atoms.info["spin"] = sample["spin"]
    atoms.calc = calc
    try:
        E0 = float(atoms.get_potential_energy())
        F0 = atoms.get_forces()
        F0_max = float(np.linalg.norm(F0, axis=1).max())
    except Exception as exc:
        return {"error": f"E0 fail: {type(exc).__name__}: {str(exc)[:80]}"}

    dyn = BFGS(atoms, logfile=None)
    try:
        dyn.run(fmax=force_tol, steps=max_steps)
    except Exception as exc:
        return {"E0_eV": E0, "F0_max_eVA": F0_max,
                "error": f"BFGS fail: {type(exc).__name__}: {str(exc)[:80]}"}

    E1 = float(atoms.get_potential_energy())
    F1 = atoms.get_forces()
    F1_max = float(np.linalg.norm(F1, axis=1).max())
    n_steps = int(dyn.nsteps)
    converged = bool(F1_max < force_tol)
    return {
        "E0_eV": E0, "F0_max_eVA": F0_max,
        "E1_eV": E1, "F1_max_eVA": F1_max,
        "n_steps": n_steps, "converged": converged,
        "dE_eV": E0 - E1,
    }


def _summarize(rows: list[dict], label: str) -> dict:
    ok = [r for r in rows if "error" not in r]
    n = len(ok)
    if n == 0:
        return {"label": label, "n_total": len(rows), "n_ok": 0}
    steps = np.array([r["n_steps"] for r in ok])
    f0 = np.array([r["F0_max_eVA"] for r in ok])
    f1 = np.array([r["F1_max_eVA"] for r in ok])
    de = np.array([r["dE_eV"] for r in ok])
    conv = np.array([r["converged"] for r in ok])
    return {
        "label": label,
        "n_total": len(rows), "n_ok": n,
        "fraction_converged": float(conv.mean()),
        "mean_n_steps_ok": float(steps.mean()),
        "median_n_steps_ok": float(np.median(steps)),
        "mean_n_steps_conv": float(steps[conv].mean()) if conv.any() else float("nan"),
        "median_n_steps_conv": float(np.median(steps[conv])) if conv.any() else float("nan"),
        "mean_F0_max": float(f0.mean()),
        "mean_F1_max": float(f1.mean()),
        "mean_dE_eV": float(de.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bgfm_samples", type=Path, required=True,
                    help="Path or directory of BGFM-sampled geometries (npz)")
    ap.add_argument("--fm_samples", type=Path, required=True,
                    help="Path or directory of FM-baseline-sampled geometries (npz)")
    ap.add_argument("--force_tol", type=float, default=0.05,
                    help="BFGS converge threshold on max |force| in eV/A")
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--max_samples_per_model", type=int, default=200)
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[lvl2] loading BGFM samples from {args.bgfm_samples}", flush=True)
    bgfm = _load_samples(args.bgfm_samples)[:args.max_samples_per_model]
    print(f"[lvl2] loading FM   samples from {args.fm_samples}", flush=True)
    fm = _load_samples(args.fm_samples)[:args.max_samples_per_model]
    print(f"[lvl2] N: BGFM={len(bgfm)}  FM={len(fm)}", flush=True)

    calc = _load_calc(args.ckpt, args.device)
    print("[lvl2] OMol25 loaded", flush=True)

    rows = []
    for label, samples in [("BGFM", bgfm), ("FM-baseline", fm)]:
        t0 = time.time()
        for i, s in enumerate(samples):
            r = _relax_one(s, calc, args.force_tol, args.max_steps)
            r["model"] = label
            r["sample_idx_global"] = i
            r["source_file"] = s["source_file"]
            r["source_sample_idx"] = s["sample_idx"]
            rows.append(r)
            if (i + 1) % 20 == 0:
                dt = time.time() - t0
                eta = (len(samples) - i - 1) * dt / (i + 1)
                print(f"  {label}  {i+1}/{len(samples)}  ({dt:.0f}s, "
                      f"eta {eta:.0f}s)", flush=True)

    # Summary
    bg_summary = _summarize([r for r in rows if r["model"] == "BGFM"], "BGFM")
    fm_summary = _summarize([r for r in rows if r["model"] == "FM-baseline"],
                             "FM-baseline")

    with open(args.out_dir / "level2_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bg_summary))
        w.writeheader(); w.writerow(bg_summary); w.writerow(fm_summary)

    # Per-sample CSV (for figures)
    detail_keys = ["model", "sample_idx_global", "source_file", "source_sample_idx",
                   "n_steps", "converged", "E0_eV", "E1_eV", "F0_max_eVA",
                   "F1_max_eVA", "dE_eV", "error"]
    with open(args.out_dir / "level2_per_sample.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=detail_keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in detail_keys})

    # Headline
    print(f"\n[lvl2 SUMMARY]   (force_tol={args.force_tol} eV/A, max_steps={args.max_steps})")
    for s in (bg_summary, fm_summary):
        if s.get("n_ok", 0) == 0:
            print(f"  {s['label']:>11}: NO OK SAMPLES")
            continue
        print(f"  {s['label']:>11}: "
              f"mean_steps={s['mean_n_steps_ok']:.1f}  "
              f"median_steps={s['median_n_steps_ok']:.0f}  "
              f"conv_frac={s['fraction_converged']:.2f}  "
              f"mean_F0={s['mean_F0_max']:.3f}  mean_F1={s['mean_F1_max']:.3f}  "
              f"mean_dE={s['mean_dE_eV']:.2f} eV")
    if bg_summary.get("mean_n_steps_ok") and fm_summary.get("mean_n_steps_ok"):
        rel = (fm_summary["mean_n_steps_ok"] - bg_summary["mean_n_steps_ok"]) \
              / fm_summary["mean_n_steps_ok"]
        print(f"\n  HEADLINE  BGFM saves {rel*100:.1f}% of FM-baseline relax steps")
        print(f"            (target >30% for Level-2 success, >50% for oral pitch)")
    print(f"\n  wrote {args.out_dir}/level2_summary.csv + level2_per_sample.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
