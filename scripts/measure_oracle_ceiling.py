"""Oracle ceiling: how well can ANY model possibly score on our headline metric?

The BGFM energy arm is trained against the OMol25 eSEN teacher, but evaluated
against an INDEPENDENT potential (GFN2-xTB) to avoid oracle circularity.  That
independence has a price: even a model that reproduced the teacher's Boltzmann
distribution *exactly* would not get r = 1 against xTB, because eSEN and xTB
disagree about relative conformer energies.

This script measures that ceiling.  For the same geometries used in the eval, it
scores every structure with

  * the teacher eSEN (fairchem OMol25, envs/omol25), and
  * GFN2-xTB (reused from an existing boltz_*records.csv -- no re-run needed),

then feeds  u = -E_eSEN / kT   (the log-density a PERFECT student would have, up
to the per-molecule constant that all our metrics are invariant to)  and
e = E_xTB  through exactly the same calibration machinery as
``analyze_calibration.py``.  The resulting r / slope / NRV are the maximum any
student could achieve on the reported metric.

Run in envs/omol25 (fairchem + ase; torch 2.8):

    /n/holylabs/woo_lab/Lab/yulili/bgfm/envs/omol25/bin/python \
        scripts/measure_oracle_ceiling.py \
        --samples_json .../boltz1/boltzmann_samples.json \
        --records_csv .../boltz_records.csv \
        --out_dir /n/holylabs/woo_lab/Lab/yulili/bgfm/runs/eval_ours/ceiling

eSEN energies are cached to ``esen_energies.jsonl`` incrementally, so the job is
resumable after preemption -- delete the file to force a recompute.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_calibration import (  # noqa: E402
    group_metrics, _summ, _f, KB_EV_PER_K,
)

DEFAULT_CKPT = "/n/holylabs/woo_lab/Lab/yulili/bgfm/checkpoints/omol25/esen_sm_conserving_all.pt"
EV_TO_KCAL = 23.060548


# ------------------------------------------------------------------ eSEN scoring
def load_calculator(ckpt, device="cpu"):
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    pred = load_predict_unit(ckpt, device=device)
    try:
        return FAIRChemCalculator(pred, task_name="omol")
    except TypeError:
        return FAIRChemCalculator(pred)


def score_esen(records, ckpt, device, cache_path, flush_every=20, limit=0):
    """Return {(group_id, pert_id): E_eV}. Resumable via a JSONL cache."""
    import ase

    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue  # truncated final line from a killed job
                if d.get("E_eV") is not None:
                    cache[(int(d["group_id"]), int(d["pert_id"]))] = float(d["E_eV"])
        print(f"[ceiling] resumed {len(cache)} cached eSEN energies from {cache_path}")

    todo = [r for r in records if (int(r["group_id"]), int(r["pert_id"])) not in cache]
    if limit:
        todo = todo[:limit]
    if not todo:
        return cache

    calc = load_calculator(ckpt, device=device)
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or ".", exist_ok=True)
    t0 = time.time()
    n_fail = 0
    with open(cache_path, "a") as fh:
        for i, r in enumerate(todo):
            gid, pid = int(r["group_id"]), int(r["pert_id"])
            try:
                atoms = ase.Atoms(numbers=r["atomic_numbers"],
                                  positions=np.asarray(r["positions"], dtype=float))
                atoms.info["charge"] = int(r.get("charge", 0))
                atoms.info["spin"] = int(r.get("spin", 1))
                atoms.calc = calc
                e = float(atoms.get_potential_energy())
                if not math.isfinite(e):
                    raise ValueError("non-finite energy")
                cache[(gid, pid)] = e
                fh.write(json.dumps({"group_id": gid, "pert_id": pid, "E_eV": e}) + "\n")
            except Exception as exc:
                n_fail += 1
                fh.write(json.dumps({"group_id": gid, "pert_id": pid, "E_eV": None,
                                     "error": f"{type(exc).__name__}: {str(exc)[:120]}"}) + "\n")
            if (i + 1) % flush_every == 0:
                fh.flush()
                os.fsync(fh.fileno())
                rate = (i + 1) / (time.time() - t0)
                eta = (len(todo) - i - 1) / max(rate, 1e-9)
                print(f"[ceiling] eSEN {i+1}/{len(todo)}  {rate:.2f}/s  ETA {eta/60:.1f} min "
                      f"(fail {n_fail})", flush=True)
    print(f"[ceiling] eSEN done: {len(cache)} energies, {n_fail} failures, "
          f"{time.time()-t0:.0f}s")
    return cache


# --------------------------------------------------------------------- xTB reuse
def load_xtb(records_csv):
    out = {}
    with open(records_csv, newline="") as f:
        for row in csv.DictReader(f):
            ok = str(row.get("xtb_ok", "True")).strip().lower() in ("true", "1", "yes")
            if not ok:
                continue
            try:
                e = float(row["E_eV"])
            except (TypeError, ValueError, KeyError):
                continue
            if math.isfinite(e):
                out[(int(float(row["group_id"])), int(float(row["pert_id"])))] = e
    return out


# ------------------------------------------------------------------------ report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_json", required=True,
                    help="Stage-1 boltzmann_samples.json (geometries; log_p ignored)")
    ap.add_argument("--records_csv", required=True,
                    help="boltz_*records.csv holding the already-computed xTB E_eV")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--kT_eV", type=float, default=1.0,
                    help="kT for the hypothetical perfect student, u = -E_eSEN/kT")
    ap.add_argument("--drop_reference", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="only score this many NEW geometries (subset / dry run)")
    ap.add_argument("--n_groups", type=int, default=0,
                    help="restrict to the first N parent molecules")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    recs = json.load(open(a.samples_json))
    if a.n_groups:
        keep = sorted({int(r["group_id"]) for r in recs})[: a.n_groups]
        keep = set(keep)
        recs = [r for r in recs if int(r["group_id"]) in keep]
    if a.drop_reference:
        recs = [r for r in recs if int(r["pert_id"]) != 0]
    print(f"[ceiling] {len(recs)} geometries, "
          f"{len({r['group_id'] for r in recs})} parent molecules")

    esen = score_esen(recs, a.ckpt, a.device,
                      os.path.join(a.out_dir, "esen_energies.jsonl"), limit=a.limit)
    xtb = load_xtb(a.records_csv)
    print(f"[ceiling] xTB energies available for {len(xtb)} geometries")

    # group -> arrays
    groups = {}
    for r in recs:
        k = (int(r["group_id"]), int(r["pert_id"]))
        if k in esen and k in xtb:
            groups.setdefault(k[0], {"esen": [], "xtb": [], "pert": []})
            groups[k[0]]["esen"].append(esen[k])
            groups[k[0]]["xtb"].append(xtb[k])
            groups[k[0]]["pert"].append(k[1])

    per_group, rows = {}, []
    de_all = []
    for gid, d in sorted(groups.items()):
        Ee = np.asarray(d["esen"], float)
        Ex = np.asarray(d["xtb"], float)
        if Ee.size < 3:
            continue
        # A perfect student's log-density, up to the per-molecule constant that
        # NRV / slope / r are all invariant to.
        u = -Ee / a.kT_eV
        m = group_metrics(u, Ex, a.kT_eV)
        if m is None:
            continue
        # relative-energy agreement (centre each group; absolute offsets differ by
        # ~1e4 eV because the two methods use different references)
        dE = (Ee - Ee.mean()) - (Ex - Ex.mean())
        m["rel_mae_eV"] = float(np.mean(np.abs(dE)))
        m["rel_rmse_eV"] = float(np.sqrt(np.mean(dE ** 2)))
        m["rel_mae_kcal"] = m["rel_mae_eV"] * EV_TO_KCAL
        m["std_E_esen_eV"] = float(np.std(Ee, ddof=1))
        de_all.append(dE)
        per_group[gid] = m
        rows.append({"group_id": gid, "n_pert": m["n_pert"],
                     "pearson_r": m["pearson_r"], "spearman_r": m["spearman_r"],
                     "slope": m["slope"], "nrv": m["nrv"],
                     "T_eff_eV": m["T_eff_eV"],
                     "rel_mae_eV": m["rel_mae_eV"],
                     "std_E_xtb_eV": m["std_E_eV"], "std_E_esen_eV": m["std_E_esen_eV"]})

    keys = ["pearson_r", "spearman_r", "r2", "slope", "nrv", "nrv_min",
            "T_eff_eV", "rel_mae_eV", "rel_mae_kcal", "rel_rmse_eV",
            "std_E_eV", "std_E_esen_eV"]
    agg = {}
    for k in keys:
        agg.update(_summ([m[k] for m in per_group.values()], k))
    slopes = np.array([m["slope"] for m in per_group.values()], float)
    med_slope = float(np.median(slopes)) if slopes.size else float("nan")

    summary = {
        "what": "ceiling = teacher eSEN scored against independent GFN2-xTB",
        "interpretation": ("These are the values a student that matched the eSEN "
                           "Boltzmann distribution EXACTLY would obtain on the "
                           "reported (xTB-based) metric."),
        "samples_json": os.path.abspath(a.samples_json),
        "records_csv": os.path.abspath(a.records_csv),
        "ckpt": a.ckpt,
        "kT_eV": a.kT_eV,
        "drop_reference": bool(a.drop_reference),
        "n_geometries_scored": len(esen),
        "n_groups": len(per_group),
        "median_slope": med_slope,
        "T_eff_eV_from_median_slope": (a.kT_eV / med_slope
                                       if abs(med_slope) > 1e-12 else float("inf")),
        "frac_r_gt_0.8": float(np.mean([m["pearson_r"] > 0.8 for m in per_group.values()]))
        if per_group else float("nan"),
        "frac_r_gt_0.5": float(np.mean([m["pearson_r"] > 0.5 for m in per_group.values()]))
        if per_group else float("nan"),
        **agg,
    }

    with open(os.path.join(a.out_dir, "ceiling_summary.json"), "w") as f:
        json.dump({"summary": summary,
                   "per_group": {str(k): v for k, v in per_group.items()}}, f, indent=2)
    with open(os.path.join(a.out_dir, "ceiling_per_group.csv"), "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

    print("\n===== ORACLE CEILING (eSEN teacher vs independent GFN2-xTB) =====")
    print(f"groups                      : {summary['n_groups']}")
    print(f"pearson r   mean / median   : {_f(summary['pearson_r_mean'])} / "
          f"{_f(summary['pearson_r_median'])}   "
          f"[95% CI {_f(summary['pearson_r_ci95_lo'])}, {_f(summary['pearson_r_ci95_hi'])}]")
    print(f"spearman    mean / median   : {_f(summary['spearman_r_mean'])} / "
          f"{_f(summary['spearman_r_median'])}")
    print(f"slope       mean / median   : {_f(summary['slope_mean'],3)} / {_f(med_slope,3)}")
    print(f"NRV         mean / median   : {_f(summary['nrv_mean'])} / {_f(summary['nrv_median'])}")
    print(f"NRV_min (=1-r^2) mean       : {_f(summary['nrv_min_mean'])}")
    print(f"rel-energy MAE (eV / kcal)  : {_f(summary['rel_mae_eV_mean'],4)} / "
          f"{_f(summary['rel_mae_kcal_mean'],2)}")
    print(f"per-group std E_xtb / E_eSEN: {_f(summary['std_E_eV_mean'],3)} / "
          f"{_f(summary['std_E_esen_eV_mean'],3)} eV")
    print(f"frac groups r>0.8 / r>0.5   : {_f(summary['frac_r_gt_0.8'],2)} / "
          f"{_f(summary['frac_r_gt_0.5'],2)}")
    print(f"\nwrote {a.out_dir}/ceiling_summary.json")


if __name__ == "__main__":
    main()
