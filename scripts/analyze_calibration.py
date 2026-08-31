"""Calibration metrics for Boltzmann fidelity (P2).

Upgrades the headline metric from Pearson correlation -- which is invariant to
BOTH an additive offset and a multiplicative rescaling of log p, and therefore
cannot see that a model is at the wrong temperature -- to a *calibration* metric.

Given per-record pairs (log p_theta, E_eval) grouped by parent molecule
(``boltz_*records.csv`` produced by ``eval_boltzmann_independent.py``), per group
m with K geometries we report:

  NRV_m  = Var_k[ log p_theta + beta E ] / Var_k[ beta E ]        beta = 1/kT
      Normalised residual variance. 0 == perfect Boltzmann, 1 == log p carries
      no usable signal (or exactly cancels), >1 == worse than a constant.
      Invariant to the per-molecule intercept (the unknown log Z_m), sensitive
      to the wrong temperature -- which is the whole point.

  slope_m = OLS slope of  log p_theta  on  (-E/kT).  Ideal = 1.

  T_eff_m = kT / slope_m.  The temperature the model's density *actually*
      implies.  Algebraically T_eff = -Var(E)/Cov(log p, E), independent of the
      kT used to form beta, so it is a property of the model, not of the
      analysis choice.

  pearson / spearman (reported, but secondary).

Useful decomposition.  Writing a = Var(u)/Var(E), b = Cov(u,E)/Var(E) with
u = log p:
      NRV(kT) = a kT^2 + 2 b kT + 1
which is a parabola in kT, minimised at kT*_NRV = -b/a with value 1 - r^2.
So `nrv_min = 1 - r^2` is the *best NRV any temperature rescaling could buy*:
it separates "wrong temperature" (NRV >> nrv_min) from "wrong shape"
(nrv_min itself large).  Both are reported.

Usage
-----
    python scripts/analyze_calibration.py \
        --records LABEL=/path/to/boltz_records.csv [--records LABEL2=...] \
        --kT_eV 1.0 --out_json out.json --out_md out.md [--drop_reference]

Pure numpy/scipy: runs in either conda env.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

try:
    from scipy.stats import spearmanr as _spearmanr
except Exception:  # pragma: no cover
    _spearmanr = None

KB_EV_PER_K = 8.617333262e-5  # Boltzmann constant, eV/K
ROOM_T_EV = 298.15 * KB_EV_PER_K  # ~0.025693 eV


# --------------------------------------------------------------------------- IO
def load_records(path, drop_reference=False, reference_pert_id=0):
    """Read a boltz_*records.csv -> {group_id: (log_p array, E_eV array)}."""
    groups = defaultdict(lambda: ([], []))
    n_rows = n_bad = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            n_rows += 1
            ok = str(row.get("xtb_ok", "True")).strip().lower() in ("true", "1", "yes")
            if not ok:
                n_bad += 1
                continue
            try:
                lp = float(row["log_p_theta"])
                e = float(row["E_eV"])
            except (TypeError, ValueError, KeyError):
                n_bad += 1
                continue
            if not (math.isfinite(lp) and math.isfinite(e)):
                n_bad += 1
                continue
            if drop_reference and int(float(row.get("pert_id", -1))) == reference_pert_id:
                continue
            gid = int(float(row["group_id"]))
            groups[gid][0].append(lp)
            groups[gid][1].append(e)
    out = {g: (np.asarray(u, float), np.asarray(e, float)) for g, (u, e) in groups.items()}
    return out, {"n_rows": n_rows, "n_dropped": n_bad}


# ---------------------------------------------------------------- core metrics
def group_metrics(u, e, kT_eV, min_k=3):
    """Calibration metrics for one parent molecule.

    u : log p_theta (K,)          e : evaluation energy in eV (K,)
    """
    K = u.size
    if K < min_k:
        return None
    var_e = float(np.var(e, ddof=1))
    var_u = float(np.var(u, ddof=1))
    if var_e < 1e-12 or not math.isfinite(var_e):
        return None
    cov_ue = float(np.cov(u, e, ddof=1)[0, 1])

    a = var_u / var_e                       # Var(u)/Var(E)
    b = cov_ue / var_e                      # Cov(u,E)/Var(E); Boltzmann => -1/kT
    # sign convention: r = corr(log p, -E), so a Boltzmann-consistent model has r=+1
    r = -cov_ue / math.sqrt(var_u * var_e) if var_u > 1e-30 else float("nan")

    def nrv_at(kT):
        return a * kT * kT + 2.0 * b * kT + 1.0

    # slope of u on x = -E/kT
    slope = -b * kT_eV
    # T_eff = kT/slope = -1/b  (kT-invariant)
    T_eff_eV = (-1.0 / b) if abs(b) > 1e-30 else float("inf")

    kT_star = (-b / a) if a > 1e-30 else float("nan")   # argmin of NRV(kT)
    nrv_min = 1.0 - (r * r) if math.isfinite(r) else float("nan")

    if _spearmanr is not None and K >= 3:
        sp = float(_spearmanr(u, -e).correlation)
    else:
        sp = float("nan")

    # Decomposition: slope = r * scale_ratio, with
    #   scale_ratio = std(log p) / std(beta E)
    # This separates the two ways to be mis-calibrated. r says whether log p moves
    # in the RIGHT DIRECTION; scale_ratio says whether it moves by the RIGHT AMOUNT.
    # Pearson r is blind to scale_ratio entirely -- that blindness is the reason
    # for this script.
    scale_ratio = math.sqrt(var_u) / (math.sqrt(var_e) / kT_eV) if var_e > 0 else float("nan")

    return {
        "n_pert": int(K),
        "scale_ratio": scale_ratio,   # ideal 1.0
        "var_logp": var_u,
        "var_E_eV2": var_e,
        "std_E_eV": math.sqrt(var_e),
        "nrv": nrv_at(kT_eV),
        "nrv_roomT": nrv_at(ROOM_T_EV),
        "nrv_min": nrv_min,               # == 1 - r^2, best over any kT rescale
        "kT_star_eV": kT_star,            # kT minimising NRV
        "slope": slope,                   # ideal 1.0
        "T_eff_eV": T_eff_eV,
        "T_eff_K": T_eff_eV / KB_EV_PER_K if math.isfinite(T_eff_eV) else float("inf"),
        "pearson_r": r,                   # corr(log p, -E)
        "spearman_r": sp,
        "r2": r * r if math.isfinite(r) else float("nan"),
    }


def _boot_ci(vals, n_boot=2000, alpha=0.05, seed=0, agg=np.mean):
    v = np.asarray([x for x in vals if math.isfinite(x)], float)
    if v.size < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    stats = agg(v[idx], axis=1)
    return (float(np.percentile(stats, 100 * alpha / 2)),
            float(np.percentile(stats, 100 * (1 - alpha / 2))))


def _summ(vals, name, n_boot=2000):
    v = np.asarray([x for x in vals if math.isfinite(x)], float)
    if v.size == 0:
        # keep the key set identical to the populated case so downstream
        # formatting never KeyErrors on an empty/degenerate arm
        return {f"{name}_{k}": (float("nan") if k != "n" else 0)
                for k in ("mean", "trimmean10", "median", "q25", "q75",
                          "ci95_lo", "ci95_hi", "n")}
    lo, hi = _boot_ci(v, n_boot=n_boot)
    # NRV / T_eff are heavy-tailed (a group whose Var(E) is tiny, or whose slope is
    # ~0, blows the plain mean up by orders of magnitude), so a 10%-trimmed mean is
    # reported alongside; the MEDIAN is the metric to quote.
    k = int(0.1 * v.size)
    vs = np.sort(v)
    trimmed = vs[k: v.size - k] if v.size - 2 * k >= 1 else vs
    return {
        f"{name}_mean": float(np.mean(v)),
        f"{name}_trimmean10": float(np.mean(trimmed)),
        f"{name}_median": float(np.median(v)),
        f"{name}_q25": float(np.percentile(v, 25)),
        f"{name}_q75": float(np.percentile(v, 75)),
        f"{name}_ci95_lo": lo,
        f"{name}_ci95_hi": hi,
        f"{name}_n": int(v.size),
    }


def analyze_one(path, kT_eV, drop_reference, n_boot=2000):
    groups, io = load_records(path, drop_reference=drop_reference)
    per_group, skipped = {}, 0
    for gid, (u, e) in sorted(groups.items()):
        m = group_metrics(u, e, kT_eV)
        if m is None:
            skipped += 1
            continue
        per_group[gid] = m

    keys = ["nrv", "nrv_roomT", "nrv_min", "slope", "scale_ratio", "T_eff_eV",
            "T_eff_K", "pearson_r", "spearman_r", "r2", "kT_star_eV",
            "std_E_eV"]
    agg = {}
    for k in keys:
        agg.update(_summ([m[k] for m in per_group.values()], k, n_boot=n_boot))

    slopes = np.array([m["slope"] for m in per_group.values()], float)
    nrvs = np.array([m["nrv"] for m in per_group.values()], float)
    # T_eff from the *median* slope is far more stable than the median of per-group
    # T_eff, because slope ~ 0 sends individual T_eff to +-inf.
    med_slope = float(np.median(slopes)) if slopes.size else float("nan")
    T_eff_from_med_slope = (kT_eV / med_slope) if abs(med_slope) > 1e-12 else float("inf")

    # Pooled (per-group mean-centred, then stacked) calibration -- one number
    # over all geometries, so groups with tiny Var(E) cannot dominate.
    pu, pe = [], []
    for gid, (u, e) in sorted(groups.items()):
        if gid in per_group:
            pu.append(u - u.mean())
            pe.append(e - e.mean())
    if pu:
        PU = np.concatenate(pu)
        PE = np.concatenate(pe)
        pooled = group_metrics(PU, PE, kT_eV)
    else:
        pooled = None

    return {
        "records_csv": os.path.abspath(path),
        "kT_eV": kT_eV,
        "drop_reference": bool(drop_reference),
        "io": io,
        "n_groups": len(per_group),
        "n_groups_skipped": skipped,
        "frac_slope_positive": float(np.mean(slopes > 0)) if slopes.size else float("nan"),
        "frac_nrv_lt_1": float(np.mean(nrvs < 1.0)) if nrvs.size else float("nan"),
        "median_slope": med_slope,
        "T_eff_eV_from_median_slope": T_eff_from_med_slope,
        "T_eff_K_from_median_slope": (T_eff_from_med_slope / KB_EV_PER_K
                                      if math.isfinite(T_eff_from_med_slope) else float("inf")),
        "pooled": pooled,
        **agg,
        "per_group": {str(k): v for k, v in per_group.items()},
    }


# ------------------------------------------------------------------- reporting
def _f(x, n=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a" if x is None or math.isnan(x) else ("inf" if x > 0 else "-inf")
    if abs(x) >= 1e5:
        return f"{x:.3g}"
    return f"{x:.{n}f}"


def markdown_table(results, kT_eV):
    hdr = ("| arm | n | NRV median | NRV trim-mean | NRV_min (=1-r^2) | slope median | "
           "scale-ratio med | T_eff (eV) | T_eff (K) | r mean | rho mean |")
    sep = "|---" * 11 + "|"
    lines = [hdr, sep]
    for label, res in results.items():
        lines.append(
            "| {lab} | {n} | {nmed} | {nm} | {nmin} | {sl} | {sr} | {te} | {tek} | {r} | {sp} |".format(
                lab=label, n=res["n_groups"],
                nm=_f(res["nrv_trimmean10"]), nmed=_f(res["nrv_median"]),
                nmin=_f(res["nrv_min_mean"]),
                sl=_f(res["median_slope"], 4),
                sr=_f(res["scale_ratio_median"], 3),
                te=_f(res["T_eff_eV_from_median_slope"], 2),
                tek=_f(res["T_eff_K_from_median_slope"], 0),
                r=_f(res["pearson_r_mean"]), sp=_f(res["spearman_r_mean"])))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", action="append", default=[], required=True,
                    help="LABEL=path (path may be a glob). Repeatable.")
    ap.add_argument("--kT_eV", type=float, default=1.0,
                    help="kT used for beta in NRV/slope (training config uses 1.0 eV).")
    ap.add_argument("--drop_reference", action="store_true",
                    help="drop pert_id==0 (the unperturbed reference conformer).")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", default="")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--no_per_group", action="store_true",
                    help="omit per-group dict from JSON (smaller file)")
    a = ap.parse_args()

    specs = []
    for item in a.records:
        if "=" in item:
            label, pat = item.split("=", 1)
        else:
            label, pat = os.path.basename(os.path.dirname(item)), item
        matches = sorted(glob.glob(pat)) or ([pat] if os.path.exists(pat) else [])
        if not matches:
            print(f"[calib] WARNING: no match for {pat}")
        for i, p in enumerate(matches):
            lab = label if len(matches) == 1 else f"{label}[{os.path.basename(os.path.dirname(p))}]"
            specs.append((lab, p))

    results = {}
    for label, path in specs:
        try:
            results[label] = analyze_one(path, a.kT_eV, a.drop_reference, n_boot=a.n_boot)
            r = results[label]
            print(f"[calib] {label}: n={r['n_groups']}  NRVmed={_f(r['nrv_median'])} "
                  f"(trim {_f(r['nrv_trimmean10'])})  NRVmin={_f(r['nrv_min_mean'])}  "
                  f"slope_med={_f(r['median_slope'],4)}  scale={_f(r['scale_ratio_median'],3)}  "
                  f"T_eff={_f(r['T_eff_eV_from_median_slope'],2)} eV  "
                  f"r={_f(r['pearson_r_mean'])}")
        except Exception as exc:  # keep going over a batch of arms
            print(f"[calib] FAILED {label} ({path}): {type(exc).__name__}: {exc}")

    out = {
        "kT_eV": a.kT_eV,
        "drop_reference": bool(a.drop_reference),
        "room_T_eV": ROOM_T_EV,
        "arms": {},
    }
    for k, v in results.items():
        vv = dict(v)
        if a.no_per_group:
            vv.pop("per_group", None)
        out["arms"][k] = vv

    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)) or ".", exist_ok=True)
    with open(a.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[calib] wrote {a.out_json}")

    table = markdown_table(results, a.kT_eV)
    print("\n" + table)
    if a.out_md:
        os.makedirs(os.path.dirname(os.path.abspath(a.out_md)) or ".", exist_ok=True)
        with open(a.out_md, "w") as f:
            f.write(f"# Calibration metrics (kT = {a.kT_eV} eV, "
                    f"drop_reference={bool(a.drop_reference)})\n\n")
            f.write(table + "\n")
        print(f"[calib] wrote {a.out_md}")


if __name__ == "__main__":
    main()
