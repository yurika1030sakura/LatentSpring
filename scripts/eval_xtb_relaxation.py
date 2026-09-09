"""Independent physical-quality evaluation via GFN2-xTB relaxation (Q2).

This is the audit's headline external evaluator: GFN2-xTB is not used
in any BGFM training loss, so a quality lift here cannot be attributed
to training-evaluation overlap.

For each generated molecule we run a GFN2-xTB geometry optimization
using the standalone ``xtb`` binary, then record:

  - delta_E_kcalmol     E_before - E_after (positive = strain released)
  - max_force_evA       max |F| component before relaxation (eV/A)
  - rmsd_A              RMSD between initial and relaxed geometries
  - steps               xTB optimizer iterations used
  - converged           bool, did xtb reach its gradient threshold
  - failure_reason      None or a short string

We then aggregate the population-level numbers the paper Table 2 wants:
median / mean / p90 ΔE, median max force, median steps, failure rate.

Hard requirements
-----------------

The ``xtb`` binary must be on PATH (install via
``conda install -c conda-forge xtb``). RDKit is used only for symbol
lookup; it is the same dependency the other eval scripts use.

Usage
-----

    python scripts/eval_xtb_relaxation.py \
        --samples runs/eval/geomdrugs/samples.json \
        --out_csv runs/eval/geomdrugs/xtb_relax.csv \
        --max_steps 200 --workers 4

Output is one CSV row per sample plus a summary row keyed
``__summary__``.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional


HARTREE_TO_KCAL = 627.5094740631
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = 27.211386245988 / 0.529177210903
# E_BEFORE / E_AFTER appear in xtb stdout as "TOTAL ENERGY    -157.123456 Eh"
_E_LINE = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh")
# Gradient norm line.
_GRAD_LINE = re.compile(r"GRADIENT NORM\s+(-?\d+\.\d+)\s+Eh/α")
_CONVERGED = re.compile(r"GEOMETRY OPTIMIZATION CONVERGED")
_CYCLES = re.compile(r"finished SCC after\s+(\d+)\s+iterations")


def _atomic_numbers_to_symbols(atomic_numbers: list[int]) -> list[str]:
    try:
        from rdkit.Chem import GetPeriodicTable
        pt = GetPeriodicTable()
        return [pt.GetElementSymbol(int(z)) for z in atomic_numbers]
    except Exception:
        Z_TO_SYM = {1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
                    14: "Si", 15: "P", 16: "S",
                    17: "Cl", 35: "Br", 53: "I"}
        return [Z_TO_SYM.get(int(z), "X") for z in atomic_numbers]


def _write_xyz(path: Path, atomic_numbers: list[int], positions: list[list[float]],
               total_charge: int) -> None:
    symbols = _atomic_numbers_to_symbols(atomic_numbers)
    with open(path, "w") as f:
        f.write(f"{len(symbols)}\n")
        f.write(f"charge={total_charge}\n")
        for sym, (x, y, z) in zip(symbols, positions):
            f.write(f"{sym} {x:.6f} {y:.6f} {z:.6f}\n")


def _read_xyz_positions(path: Path) -> Optional[list[list[float]]]:
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return None
    if len(lines) < 2:
        return None
    try:
        n = int(lines[0].strip())
    except ValueError:
        return None
    coords = []
    for ln in lines[2:2 + n]:
        parts = ln.split()
        if len(parts) < 4:
            return None
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return coords if len(coords) == n else None


def _rmsd(p_before: list[list[float]], p_after: list[list[float]]) -> float:
    n = min(len(p_before), len(p_after))
    if n == 0:
        return float("nan")
    total = 0.0
    for i in range(n):
        dx = p_before[i][0] - p_after[i][0]
        dy = p_before[i][1] - p_after[i][1]
        dz = p_before[i][2] - p_after[i][2]
        total += dx * dx + dy * dy + dz * dz
    return math.sqrt(total / n)


def _xtb_binary() -> Optional[str]:
    return shutil.which("xtb")


def _xtb_single_point_energy(work_dir: Path, xyz_path: Path,
                             charge: int = 0) -> tuple[Optional[float], Optional[float]]:
    """Return (E_Hartree, max_grad_eV_per_A) for a single-point evaluation."""
    xtb = _xtb_binary()
    if xtb is None:
        return None, None
    cmd = [xtb, str(xyz_path), "--grad", "--gfn", "2",
           "--chrg", str(int(charge)), "--silent"]
    try:
        out = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True,
                             timeout=180)
    except (subprocess.TimeoutExpired, OSError):
        return None, None
    E = None
    for m in _E_LINE.finditer(out.stdout):
        E = float(m.group(1))
    grad_path = work_dir / "gradient"
    max_grad_eVA = None
    if grad_path.exists():
        try:
            grad_text = grad_path.read_text().splitlines()
            grads = []
            for line in grad_text:
                cols = line.replace("D", "E").split()
                if len(cols) == 3:
                    try:
                        gx, gy, gz = float(cols[0]), float(cols[1]), float(cols[2])
                        grads.append(max(abs(gx), abs(gy), abs(gz)))
                    except ValueError:
                        continue
            if grads:
                max_grad_au = max(grads)
                max_grad_eVA = max_grad_au * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
        except Exception:
            pass
    return E, max_grad_eVA


def _xtb_relax(work_dir: Path, xyz_path: Path, charge: int,
               max_steps: int) -> dict:
    """Drive ``xtb --opt`` and parse the result."""
    xtb = _xtb_binary()
    if xtb is None:
        return {"failure_reason": "xtb_binary_not_found"}
    cmd = [xtb, str(xyz_path), "--opt", "tight", "--cycles", str(int(max_steps)),
           "--gfn", "2", "--chrg", str(int(charge)), "--silent"]
    try:
        out = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True,
                             timeout=600)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"failure_reason": f"xtb_runtime_error:{type(exc).__name__}"}
    converged = bool(_CONVERGED.search(out.stdout))
    steps = None
    last_cycles = list(_CYCLES.finditer(out.stdout))
    if last_cycles:
        steps = int(last_cycles[-1].group(1))
    energies = [float(m.group(1)) for m in _E_LINE.finditer(out.stdout)]
    E_after = energies[-1] if energies else None
    optimized = work_dir / "xtbopt.xyz"
    relaxed_positions = _read_xyz_positions(optimized)
    if out.returncode != 0:
        return {"failure_reason": f"xtb_exit_{out.returncode}",
                "converged": converged, "steps": steps}
    if E_after is None:
        return {"failure_reason": "no_energy_in_xtb_output",
                "converged": converged, "steps": steps}
    return {
        "E_after_hartree": E_after,
        "converged": converged,
        "steps": steps,
        "relaxed_positions": relaxed_positions,
        "failure_reason": None,
    }


def evaluate_sample(idx: int, sample: dict, max_steps: int) -> dict:
    atomic_numbers = sample.get("atomic_numbers") or []
    positions = sample.get("positions") or []
    charge = int(sample.get("charge", 0))
    base = {"index": idx, "n_atoms": len(atomic_numbers), "charge": charge}
    if not atomic_numbers or len(atomic_numbers) != len(positions):
        return {**base, "failure_reason": "missing_or_mismatched_positions"}
    with tempfile.TemporaryDirectory(prefix="bgfm_xtb_") as td:
        wd = Path(td)
        xyz_path = wd / "input.xyz"
        _write_xyz(xyz_path, atomic_numbers, positions, total_charge=charge)
        E_before, max_force = _xtb_single_point_energy(wd, xyz_path, charge=charge)
        if E_before is None:
            return {**base, "failure_reason": "single_point_failed"}
        relax = _xtb_relax(wd, xyz_path, charge=charge, max_steps=max_steps)
        if relax.get("failure_reason"):
            return {**base, "E_before_hartree": E_before,
                    "max_force_evA": max_force, **relax}
        E_after = relax["E_after_hartree"]
        delta_E_kcal = (E_before - E_after) * HARTREE_TO_KCAL
        rmsd_A = _rmsd(positions, relax["relaxed_positions"]) if relax.get("relaxed_positions") else float("nan")
        return {
            **base,
            "E_before_hartree": E_before,
            "E_after_hartree": E_after,
            "delta_E_kcalmol": delta_E_kcal,
            # per-atom strain: ΔE scales ~linearly with atom count, so the per-atom
            # form is the size-fair cross-model quantity (models generate different
            # size distributions).
            "delta_E_kcal_per_atom": delta_E_kcal / max(1, len(atomic_numbers)),
            "max_force_evA": max_force,
            "rmsd_A": rmsd_A,
            "steps": relax.get("steps"),
            "converged": relax.get("converged"),
            "failure_reason": None,
        }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize(records: list[dict]) -> dict:
    ok = [r for r in records if r.get("failure_reason") is None]
    converged_ok = [r for r in ok if r.get("converged") is True]
    n = len(records)
    n_ok = len(ok)
    dE = [r["delta_E_kcalmol"] for r in ok if r.get("delta_E_kcalmol") is not None]
    dE_pa = [r["delta_E_kcal_per_atom"] for r in ok if r.get("delta_E_kcal_per_atom") is not None]
    forces = [r["max_force_evA"] for r in ok if r.get("max_force_evA") is not None]
    rmsds = [r["rmsd_A"] for r in ok if r.get("rmsd_A") == r.get("rmsd_A")]
    steps = [r["steps"] for r in ok if r.get("steps") is not None]
    return {
        "index": "__summary__",
        "n_total": n,
        "n_ok": n_ok,
        "n_converged": len(converged_ok),
        "n_unconverged_with_energy": n_ok - len(converged_ok),
        "converged_fraction_all": len(converged_ok) / n if n else float("nan"),
        "converged_delta_E_per_atom_median": _percentile(
            [r["delta_E_kcal_per_atom"] for r in converged_ok
             if r.get("delta_E_kcal_per_atom") is not None], 50.0),
        # NOTE: ΔE stats below are over the n_dE successfully-relaxed molecules only;
        # read them WITH failure_rate (a model with many xtb failures has a small,
        # self-selected ΔE population). n_dE is reported so the denominator is explicit.
        "n_dE": len(dE),
        "failure_rate": 1.0 - n_ok / n if n else float("nan"),
        "delta_E_kcal_median": _percentile(dE, 50.0),
        "delta_E_kcal_mean": (sum(dE) / len(dE)) if dE else float("nan"),
        "delta_E_kcal_p90": _percentile(dE, 90.0),
        # size-fair per-atom strain (primary cross-model number)
        "delta_E_per_atom_median": _percentile(dE_pa, 50.0),
        "delta_E_per_atom_mean": (sum(dE_pa) / len(dE_pa)) if dE_pa else float("nan"),
        "delta_E_per_atom_p90": _percentile(dE_pa, 90.0),
        "max_force_evA_median": _percentile(forces, 50.0),
        "rmsd_A_median": _percentile(rmsds, 50.0),
        "steps_median": _percentile([float(s) for s in steps], 50.0),
        "convergence_rate": sum(1 for r in ok if r.get("converged")) / n if n else float("nan"),
    }


def _load_samples(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "samples" in payload:
        return payload["samples"]
    raise ValueError(f"unsupported samples JSON shape in {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--max_steps", type=int, default=200,
                    help="Cap on xtb --cycles per molecule.")
    ap.add_argument("--workers", type=int, default=4,
                    help="Number of parallel xtb processes.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap on the number of samples to process.")
    args = ap.parse_args()

    if _xtb_binary() is None:
        print("[xtb] WARNING: 'xtb' binary not on PATH. Install with "
              "`conda install -c conda-forge xtb` in the omol25 env, or set PATH.",
              file=sys.stderr, flush=True)

    samples = _load_samples(args.samples)
    if args.limit is not None:
        samples = samples[:int(args.limit)]
    if not samples:
        print("[xtb] no samples to evaluate", flush=True)
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            csv.writer(f).writerow(["index"])
        return 0

    records: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
        futures = [ex.submit(evaluate_sample, i, s, int(args.max_steps))
                   for i, s in enumerate(samples)]
        for fut in concurrent.futures.as_completed(futures):
            records.append(fut.result())
    records.sort(key=lambda r: r.get("index", 0))

    summary = summarize(records)
    all_rows = records + [summary]
    keys: list[str] = []
    for row in all_rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in keys})
    print(f"[xtb] wrote {args.out_csv}: {summary}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
