"""Create one uniform physics-first evaluation row for Paper 1 ablations.

Inputs are optional but typically:
  - validity CSV from scripts/evaluate_validity.py
  - OMol25 energy/relax CSV from scripts/compute_omol25_energy.py

Output schema is stable across Cells 2/B/4a/4d, so the ablation table can
be appended safely.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, pstdev


FIELDNAMES = [
    "cell_id",
    "data",
    "method",
    "checkpoint",
    "config",
    "eval_data",
    "n_samples",
    "frac_valid_mols",
    "frac_connected",
    "avg_num_components",
    "avg_frag_frac",
    "omol25_ok_frac",
    "omol25_energy_mean",
    "omol25_energy_std",
    "max_force_mean",
    "max_force_p95",
    "max_force_source",
    "relax_energy_drop_mean",
    "relax_steps_mean",
    "relax_converged_frac",
    "validity_csv",
    "omol25_csv",
]


def _read_first_row(path: Path | None) -> dict:
    if path is None:
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def _read_rows(path: Path | None) -> list[dict]:
    if path is None:
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict, *keys: str):
    for key in keys:
        val = row.get(key)
        if val not in (None, ""):
            try:
                return float(val)
            except ValueError:
                continue
    return ""


def _mean(vals: list[float]):
    return mean(vals) if vals else ""


def _std(vals: list[float]):
    return pstdev(vals) if len(vals) > 1 else (0.0 if vals else "")


def _p95(vals: list[float]):
    if not vals:
        return ""
    vals = sorted(vals)
    idx = min(len(vals) - 1, math.ceil(0.95 * len(vals)) - 1)
    return vals[idx]


def _series(rows: list[dict], *keys: str) -> tuple[list[float], str]:
    for key in keys:
        vals = []
        for row in rows:
            if str(row.get("ok", "True")).lower() not in ("true", "1", "yes"):
                continue
            raw = row.get(key)
            if raw in (None, ""):
                continue
            try:
                vals.append(float(raw))
            except ValueError:
                continue
        if vals:
            return vals, key
    return [], ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell_id", required=True,
                    help="Example: cell_2, cell_B, cell_4a, cell_4d")
    ap.add_argument("--data", default="")
    ap.add_argument("--method", default="")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--config", default="")
    ap.add_argument("--eval_data", default="")
    ap.add_argument("--validity_csv", type=Path, default=None)
    ap.add_argument("--omol25_csv", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    validity = _read_first_row(args.validity_csv)
    energy_rows = _read_rows(args.omol25_csv)
    ok_rows = [r for r in energy_rows if str(r.get("ok", "True")).lower() in ("true", "1", "yes")]

    energies, _ = _series(ok_rows, "energy_eV_raw")
    max_forces, force_source = _series(
        ok_rows,
        "max_force_raw_eVpA",
        "max_force_raw",
        "max_force_final_eVpA",
        "max_force_final",
    )
    deltas, _ = _series(ok_rows, "delta_eV")
    steps, _ = _series(ok_rows, "n_relax_steps_used", "relax_steps")
    converged = [
        str(r.get("converged_to_1e-2", "")).lower() in ("true", "1", "yes")
        for r in ok_rows
        if r.get("converged_to_1e-2", "") != ""
    ]

    n_samples = (
        int(float(validity["n_samples"])) if validity.get("n_samples") not in (None, "")
        else len(energy_rows)
    )
    row = {
        "cell_id": args.cell_id,
        "data": args.data,
        "method": args.method,
        "checkpoint": args.checkpoint or validity.get("checkpoint", ""),
        "config": args.config,
        "eval_data": args.eval_data or validity.get("eval_data", ""),
        "n_samples": n_samples,
        "frac_valid_mols": _float(validity, "frac_valid_mols", "validity_frac_valid_mols"),
        "frac_connected": _float(validity, "frac_connected", "validity_frac_connected"),
        "avg_num_components": _float(validity, "avg_num_components", "validity_avg_num_components"),
        "avg_frag_frac": _float(validity, "avg_frag_frac", "validity_avg_frag_frac"),
        "omol25_ok_frac": (len(ok_rows) / len(energy_rows)) if energy_rows else "",
        "omol25_energy_mean": _mean(energies),
        "omol25_energy_std": _std(energies),
        "max_force_mean": _mean(max_forces),
        "max_force_p95": _p95(max_forces),
        "max_force_source": force_source,
        "relax_energy_drop_mean": _mean(deltas),
        "relax_steps_mean": _mean(steps),
        "relax_converged_frac": (sum(converged) / len(converged)) if converged else "",
        "validity_csv": str(args.validity_csv or ""),
        "omol25_csv": str(args.omol25_csv or ""),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.append or not args.out.exists()
    mode = "a" if args.append else "w"
    with open(args.out, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[summary] wrote {args.out}", flush=True)
    for key in (
        "cell_id", "n_samples", "frac_valid_mols", "frac_connected",
        "avg_num_components", "max_force_mean", "relax_energy_drop_mean",
        "relax_steps_mean",
    ):
        print(f"[summary] {key}: {row[key]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
