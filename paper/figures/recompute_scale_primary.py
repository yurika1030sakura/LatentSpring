#!/usr/bin/env python
"""Restrict the scale diagnostics (Table 2) to the primary-endpoint parent set.

Closes placeholder Q1.  Table 2 as first assembled was computed over all 120
evaluation parents, while Table 1's primary endpoint is the 93 parents that lie
outside the energy term's perturbation-shard pool.  The two tables therefore
described different populations.

Per-group calibration metrics do not depend on which other groups are present,
so no re-scoring is needed: this script re-aggregates the per_group blocks that
scripts/analyze_calibration.py already wrote, restricted to
paper/figures/out/primary_endpoint.json:disjoint_group_ids.

Aggregation follows scripts/analyze_calibration.py: median over parents for the
heavy-tailed quantities (NRV, NRV_min, slope, scale ratio) and the mean over
parents for the correlation, then mean +- SD and +- SEM across seeds.  Both
dispersions are printed so the manuscript can label its error bars correctly.

Output: paper/figures/out/scale_primary.json
"""
from __future__ import annotations

import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CAL = os.path.join(ROOT, "runs", "eval_ours", "calibration")
PRIMARY = os.path.join(HERE, "out", "primary_endpoint.json")
OUT = os.path.join(HERE, "out", "scale_primary.json")

ARMS = {
    # "a1_fm_only" (no suffix) is the fifth, later-scored control seed: it trained
    # to the full budget with finite weights but was never submitted to this round's
    # evaluation until it was recovered and scored under the identical protocol.
    "a1_fm_only": ["a1_fm_only_s2", "a1_fm_only_s3", "a1_fm_only_s4", "a1_fm_only_s5",
                   "a1_fm_only"],
    "a3_energy": ["a3_energy_only_s1", "a3_energy_only_s2", "a3_energy_only_s3",
                  "a3_energy_only_s5"],
    "a6_scrambled": ["a6_energy_only_shuffled_s2", "a6_energy_only_shuffled_s3",
                     "a6_energy_only_shuffled_stab_s5"],
}

# quantity -> how it is aggregated over parents.  This mirrors what
# scripts/analyze_calibration.py reports and what Table 2 prints: medians for the
# heavy-tailed quantities, MEANS for nrv_min and the correlation.  Note that
# mean_m[1 - r_m^2] is NOT 1 - (mean_m r_m)^2; the table must say which it prints.
AGG = {
    "nrv": "median",
    "nrv_min": "mean",
    "nrv_min_median": "median_of_nrv_min",
    "slope": "median",
    "scale_ratio": "median",
    "T_eff_eV": "from_median_slope",
    "pearson_r": "mean",
}


def per_seed(arm_json, gids, kT=1.0):
    pg = arm_json["per_group"]
    sel = [pg[str(g)] for g in gids if str(g) in pg]
    out = {"n_groups": len(sel)}
    for q, how in AGG.items():
        if how == "from_median_slope":
            med_slope = st.median([m["slope"] for m in sel])
            out[q] = kT / med_slope if abs(med_slope) > 1e-12 else float("inf")
        elif how == "median_of_nrv_min":
            out[q] = st.median([m["nrv_min"] for m in sel])
        elif how == "median":
            out[q] = st.median([m[q] for m in sel])
        else:
            out[q] = st.fmean([m[q] for m in sel])
    out["frac_nrv_lt_1"] = st.fmean([1.0 if m["nrv"] < 1.0 else 0.0 for m in sel])
    return out


def summarise(values):
    n = len(values)
    mean = st.fmean(values)
    sd = st.stdev(values) if n > 1 else 0.0
    return {"mean": mean, "sd": sd, "sem": sd / (n ** 0.5) if n else float("nan"),
            "n_seeds": n, "per_seed": values}


def paired_wilcoxon(blob, gids, arm_a, arm_b, quantities):
    """Per-parent seed-median values, paired Wilcoxon arm_a - arm_b."""
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return None
    out = {}
    for q in quantities:
        da, db = [], []
        for g in gids:
            va = [blob["arms"][s]["per_group"][str(g)][q]
                  for s in arm_a if str(g) in blob["arms"][s]["per_group"]]
            vb = [blob["arms"][s]["per_group"][str(g)][q]
                  for s in arm_b if str(g) in blob["arms"][s]["per_group"]]
            if not va or not vb:
                continue
            da.append(st.median(va))
            db.append(st.median(vb))
        diff = [x - y for x, y in zip(da, db)]
        stat, p = wilcoxon(da, db)
        out[q] = {"n_parents": len(diff), "median_delta": st.median(diff),
                  "mean_delta": st.fmean(diff), "p": float(p),
                  "median_a": st.median(da), "median_b": st.median(db)}
    return out


def main():
    gids = json.load(open(PRIMARY))["disjoint_group_ids"]
    result = {"n_parents_primary": len(gids), "source": "runs/eval_ours/calibration",
              "note": "per-group metrics re-aggregated on the 93 disjoint parents"}
    for setting, fname in (("withref", "calibration_withref.json"),
                           ("dropref", "calibration_dropref.json")):
        blob = json.load(open(os.path.join(CAL, fname)))
        block = {}
        for arm, seeds in ARMS.items():
            rows = {}
            for s in seeds:
                if s not in blob["arms"]:
                    continue
                rows[s] = {
                    "primary_93": per_seed(blob["arms"][s], gids),
                    "all_120": per_seed(blob["arms"][s], range(120)),
                }
            agg = {}
            for pop in ("primary_93", "all_120"):
                agg[pop] = {q: summarise([rows[s][pop][q] for s in rows])
                            for q in list(AGG) + ["frac_nrv_lt_1"]}
            block[arm] = {"per_seed": rows, "aggregate": agg}
        block["wilcoxon_energy_minus_fm_primary93"] = paired_wilcoxon(
            blob, gids, ARMS["a3_energy"], ARMS["a1_fm_only"],
            ["nrv", "nrv_min", "slope", "scale_ratio", "pearson_r"])
        block["wilcoxon_energy_minus_fm_all120"] = paired_wilcoxon(
            blob, list(range(120)), ARMS["a3_energy"], ARMS["a1_fm_only"],
            ["nrv", "nrv_min", "slope", "scale_ratio", "pearson_r"])
        result[setting] = block
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(result, open(OUT, "w"), indent=1)

    for setting in ("withref", "dropref"):
        print(f"\n=== {setting} ===")
        hdr = f"{'arm':14s} {'pop':10s} " + " ".join(f"{q:>12s}" for q in AGG)
        print(hdr)
        for arm in ARMS:
            for pop in ("all_120", "primary_93"):
                a = result[setting][arm]["aggregate"][pop]
                cells = " ".join(
                    f"{a[q]['mean']:7.3f}+-{a[q]['sd']:.3f}" for q in AGG)
                print(f"{arm:14s} {pop:10s} {cells}")
        for arm in ARMS:
            a = result[setting][arm]["aggregate"]
            print(f"  {arm}: SEM(primary) nrv={a['primary_93']['nrv']['sem']:.3f} "
                  f"r={a['primary_93']['pearson_r']['sem']:.3f} "
                  f"Teff={a['primary_93']['T_eff_eV']['sem']:.3f} "
                  f"nrv<1 frac={a['primary_93']['frac_nrv_lt_1']['mean']:.3f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
