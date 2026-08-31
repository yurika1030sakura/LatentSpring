#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dump_boltz_records.py -- write the PER-PERTURBATION table that fig2 needs.

WHY THIS EXISTS
  scripts/eval_boltzmann_independent.py already computes, for every generated
  perturbation, both  log q_theta(x)  (from the Stage-1 FFJORD pass) and
  E_GFN2-xTB(x).  It then *aggregates* to per-parent Pearson r and throws the
  individual pairs away -- only `boltz_independent.csv` (one row per parent) is
  persisted.  fig2_boltzmann_scatter wants the individual pairs so that the
  three example-parent scatter panels are real rather than illustrative.

  This script re-derives them from the Stage-1 JSON (which does keep every
  geometry and every log q_theta) by re-running the GFN2-xTB single points, and
  writes  <run_dir>/boltz_records.csv  with columns

      group_id,pert_id,n_atoms,charge,log_p_theta,E_hartree,E_eV,negE_kT

  make_experiment_figures.py picks that file up automatically and drops the
  red PLACEHOLDER stamp from fig2.

  It does NOT touch anything the eval pipeline wrote, so it is safe to run
  while other jobs are in flight.

WHERE TO RUN
  Anywhere `xtb` is on PATH.  The Boltzmann evals run it from the flowmol env
  on a compute node:
      source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
      "$FLOWMOL_PY" paper/figures/dump_boltz_records.py \
          --run_dir /n/holylabs/woo_lab/Lab/yulili/bgfm/runs/eval_ours/wide_a3_energy_only_s1

  Cost: one GFN2-xTB single point per record (1080 records for an n=120,
  n_perturb=8 run), i.e. a few CPU-minutes for small organics.  Use
  --limit_groups to do just the handful of parents fig2 actually plots.

CONSISTENCY
  kT and the Hartree->eV constant are taken from the same places the eval uses
  (kT = 1.0 eV; r and r^2 are kT-invariant, only the axis scale changes).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]          # .../bgfm
sys.path.insert(0, str(REPO / "scripts"))

HARTREE_TO_EV = 27.211386245988


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, required=True,
                    help="a runs/eval_ours/<tag> directory containing "
                         "boltz1/boltzmann_samples.json")
    ap.add_argument("--samples_json", type=Path, default=None,
                    help="override the Stage-1 JSON path")
    ap.add_argument("--out_csv", type=Path, default=None,
                    help="default: <run_dir>/boltz_records.csv")
    ap.add_argument("--kT_eV", type=float, default=1.0)
    ap.add_argument("--limit_groups", type=int, default=0,
                    help="only the first N distinct group_ids (0 = all)")
    ap.add_argument("--only_groups", default="",
                    help="comma list of group_ids to process (overrides "
                         "--limit_groups); use this to make just the parents "
                         "that fig2 plots real")
    a = ap.parse_args(argv)

    from eval_xtb_relaxation import _write_xyz, _xtb_binary          # noqa: E402
    from eval_boltzmann_independent import _xtb_single_point_energy  # noqa: E402

    if _xtb_binary() is None:
        print("ERROR: `xtb` is not on PATH -- nothing to do.  Source "
              "scripts/fasrc/env.sh (or activate envs/omol25) first.",
              file=sys.stderr)
        return 2

    sj = a.samples_json or (a.run_dir / "boltz1" / "boltzmann_samples.json")
    if not sj.is_file():
        print(f"ERROR: {sj} not found", file=sys.stderr)
        return 2
    recs = json.loads(sj.read_text())
    print(f"loaded {len(recs)} records from {sj}")

    if a.only_groups.strip():
        keep = {int(g) for g in a.only_groups.split(",") if g.strip()}
    elif a.limit_groups > 0:
        seen = []
        for r in recs:
            g = int(r["group_id"])
            if g not in seen:
                seen.append(g)
            if len(seen) >= a.limit_groups:
                break
        keep = set(seen)
    else:
        keep = None
    if keep is not None:
        recs = [r for r in recs if int(r["group_id"]) in keep]
        print(f"  restricted to {len(keep)} parents -> {len(recs)} records")

    out = a.out_csv or (a.run_dir / "boltz_records.csv")
    rows, n_fail = [], 0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, r in enumerate(recs):
            xyz = td / "m.xyz"
            _write_xyz(xyz, r["atomic_numbers"], r["positions"],
                       int(r.get("charge", 0)))
            E_h, _ = _xtb_single_point_energy(td, xyz, int(r.get("charge", 0)))
            if E_h is None:
                n_fail += 1
                continue
            E_eV = float(E_h) * HARTREE_TO_EV
            rows.append(dict(group_id=int(r["group_id"]),
                             pert_id=int(r["pert_id"]),
                             n_atoms=len(r["atomic_numbers"]),
                             charge=int(r.get("charge", 0)),
                             log_p_theta=float(r["log_p_theta"]),
                             E_hartree=float(E_h),
                             E_eV=E_eV,
                             negE_kT=-E_eV / a.kT_eV))
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(recs)} done ({n_fail} xTB failures)",
                      flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["group_id", "pert_id", "n_atoms", "charge", "log_p_theta",
            "E_hartree", "E_eV", "negE_kT"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} rows, {n_fail} xTB failures)")
    print("fig2 will now use real points; re-run make_experiment_figures.py "
          "--only fig2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
