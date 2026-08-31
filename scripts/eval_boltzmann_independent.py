"""Independent-potential Boltzmann-fidelity eval (Table 3).

Reads a Stage-1 ``boltzmann_samples.json`` (flat list of records with
``group_id``, ``atomic_numbers``, ``positions``, ``charge``, ``log_p_theta``)
and scores how well the model's density matches the Boltzmann distribution of an
INDEPENDENT potential -- GFN2-xTB, NOT the OMol25 eSEN the model trains against.
This removes the oracle-circularity confound in eval_boltzmann_stage2.py.

Per molecule group (perturbations of one held-out molecule) it computes:
  - pearson_r / R^2 / slope of  log p_theta  vs  -E/kT      (R^2 is kT-invariant)
  - ESS_frac = (sum w)^2 / (n * sum w^2),  w = exp(-E/kT) / p_theta      (in [0,1])
and reports a pooled (mean-centred-per-group) R^2 as the headline, plus mean/median
per-group r and mean ESS_frac. Energies come from a single-point xtb --gfn 2 call
reused from eval_xtb_relaxation.py. Run in envs/flowmol (xtb on PATH).
"""
import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_xtb_relaxation import _xtb_single_point_energy, _write_xyz, _xtb_binary  # noqa: E402

HARTREE_TO_EV = 27.211386245988

try:
    from scipy.stats import spearmanr as _spearmanr
except Exception:
    _spearmanr = None


def _logsumexp(a):
    a = np.asarray(a, dtype=float)
    m = np.max(a)
    if not np.isfinite(m):
        return float("-inf")
    return float(m + np.log(np.sum(np.exp(a - m))))


def _ess_frac(neg_e_over_kT, log_p):
    """ESS fraction for weights w = exp(-E/kT)/p_theta. Invariant to a constant
    shift in log w (so the unknown log-Z / normalisation offsets cancel)."""
    log_w = np.asarray(neg_e_over_kT, dtype=float) - np.asarray(log_p, dtype=float)
    n = log_w.size
    if n == 0:
        return float("nan")
    log_sum_w = _logsumexp(log_w)
    log_sum_w2 = _logsumexp(2.0 * log_w)
    if not (math.isfinite(log_sum_w) and math.isfinite(log_sum_w2)):
        return float("nan")
    ess = math.exp(2.0 * log_sum_w - log_sum_w2)
    return ess / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_json", required=True,
                    help="Stage-1 boltzmann_samples.json")
    ap.add_argument("--out_csv", required=True, help="per-group CSV")
    ap.add_argument("--out_json", required=True, help="summary JSON")
    ap.add_argument("--kT_eV", type=float, default=1.0,
                    help="kT in eV for ESS (R^2 is kT-invariant). Config uses 1.0.")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap number of records (0 = all), for quick dry-runs")
    a = ap.parse_args()

    if _xtb_binary() is None:
        print("[independent-boltz] ERROR: xtb not on PATH (activate envs/flowmol).")
        sys.exit(2)

    recs = json.load(open(a.samples_json))
    if a.limit:
        recs = recs[: a.limit]

    # 1. Independent xtb single-point energy (eV) per geometry.
    n_ok = 0
    for r in recs:
        with tempfile.TemporaryDirectory() as td:
            xyz = Path(td) / "m.xyz"
            _write_xyz(xyz, r["atomic_numbers"], r["positions"], int(r.get("charge", 0)))
            E_hartree, _ = _xtb_single_point_energy(Path(td), xyz, int(r.get("charge", 0)))
        if E_hartree is None:
            r["_xtb_ok"] = False
            continue
        r["_E_eV"] = float(E_hartree) * HARTREE_TO_EV
        r["_xtb_ok"] = True
        n_ok += 1

    # 2. Group by molecule and correlate.
    groups = {}
    for r in recs:
        if r.get("_xtb_ok") and ("log_p_theta" in r):
            groups.setdefault(r["group_id"], []).append(r)

    rows = []
    pooled_x, pooled_y = [], []      # mean-centred per group, then pooled
    per_r, per_r2, per_ess = [], [], []
    for gid, rs in sorted(groups.items()):
        if len(rs) < 3:
            continue
        logp = np.array([r["log_p_theta"] for r in rs], dtype=float)
        negE_kT = np.array([-r["_E_eV"] / a.kT_eV for r in rs], dtype=float)
        if np.std(logp) < 1e-9 or np.std(negE_kT) < 1e-9:
            continue
        r_p = float(np.corrcoef(logp, negE_kT)[0, 1])
        r2 = r_p ** 2
        slope = float(np.polyfit(negE_kT, logp, 1)[0])
        sp = float(_spearmanr(logp, negE_kT).correlation) if _spearmanr else float("nan")
        ess = _ess_frac(negE_kT, logp)
        rows.append({"group_id": gid, "n_pert": len(rs), "pearson_r": r_p,
                     "spearman_r": sp, "r2": r2, "slope": slope, "ess_frac": ess})
        per_r.append(r_p); per_r2.append(r2); per_ess.append(ess)
        pooled_x.extend((logp - logp.mean()).tolist())
        pooled_y.extend((negE_kT - negE_kT.mean()).tolist())

    px, py = np.array(pooled_x), np.array(pooled_y)
    pooled_r2 = float(np.corrcoef(px, py)[0, 1] ** 2) if px.size > 2 else float("nan")

    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group_id", "n_pert", "pearson_r",
                                          "spearman_r", "r2", "slope", "ess_frac"])
        w.writeheader()
        for row in rows:
            w.writerow(row)

    # Persist the PER-RECORD (log p, E) pairs next to the per-group summary.
    #
    # Why: the per-group CSV throws away the raw pairs, so any downstream
    # robustness check -- e.g. recomputing r after dropping pert_id==0, which is
    # the UNPERTURBED reference conformer and acts as a leverage point -- forces a
    # full re-run of every xTB single point (~12k calls, hours). Dumping them here
    # costs nothing and makes those checks a few seconds.
    rec_path = str(a.out_csv).replace(".csv", "_records.csv")
    if rec_path == str(a.out_csv):
        rec_path = str(a.out_csv) + ".records.csv"
    with open(rec_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group_id", "pert_id", "n_atoms", "charge",
                                          "log_p_theta", "E_eV", "negE_kT", "xtb_ok"])
        w.writeheader()
        for r in recs:
            if "log_p_theta" not in r:
                continue
            e_ev = r.get("_E_eV")
            w.writerow({
                "group_id": r.get("group_id"),
                "pert_id": r.get("pert_id"),
                "n_atoms": len(r.get("atomic_numbers", []) or []),
                "charge": r.get("charge", 0),
                "log_p_theta": r["log_p_theta"],
                "E_eV": e_ev,
                "negE_kT": (-e_ev / a.kT_eV) if e_ev is not None else "",
                "xtb_ok": bool(r.get("_xtb_ok", False)),
            })
    print(f"[boltz-indep] wrote per-record pairs -> {rec_path}")

    summary = {
        "potential": "gfn2-xtb (independent of eSEN training oracle)",
        "kT_eV": a.kT_eV,
        "n_records": len(recs),
        "n_xtb_ok": n_ok,
        "n_groups": len(rows),
        "pooled_r2": pooled_r2,
        "mean_pearson_r": float(np.mean(per_r)) if per_r else float("nan"),
        "median_pearson_r": float(np.median(per_r)) if per_r else float("nan"),
        "mean_r2": float(np.mean(per_r2)) if per_r2 else float("nan"),
        "frac_r_gt_0.5": float(np.mean(np.array(per_r) > 0.5)) if per_r else float("nan"),
        "frac_r_gt_0.8": float(np.mean(np.array(per_r) > 0.8)) if per_r else float("nan"),
        "mean_ess_frac": float(np.nanmean(per_ess)) if per_ess else float("nan"),
        "median_ess_frac": float(np.nanmedian(per_ess)) if per_ess else float("nan"),
    }
    json.dump(summary, open(a.out_json, "w"), indent=2)
    print(f"[independent-boltz] {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
