"""P3 analysis: is corr(log p_theta, -E) just a geometry-distance artefact?

The per-group Boltzmann correlation is computed over Gaussian perturbations of a
reference geometry.  Those perturbations do NOT all sit at the same distance from
the reference, and BOTH variables are monotone in that distance:

    log p_theta  falls off away from the data manifold
    -E/kT        falls off away from the local minimum

so a model that learned only "far from the manifold => low density" scores a
positive r without any Boltzmann structure at all.  This script quantifies and
removes that channel:

  1. raw per-group Pearson r(log p, -E/kT)                      -- the headline number
  2. r(log p, RMSD) and r(-E/kT, RMSD)                          -- how big is the confound
  3. PARTIAL correlation r(log p, -E/kT | RMSD) per group       -- the corrected number
  4. RMSD-stratified correlation (bin by RMSD, correlate inside bins)
  5. OLS with RMSD as a covariate:  log p = a + b*(-E/kT) + c*RMSD + group FE
  6. bootstrap CI over groups for the mean partial r

For data produced by ``eval_boltzmann_controlled.py --perturb_mode fixed_rmsd`` the
within-group RMSD variance is exactly zero, so the partial correlation is
undefined *and unnecessary*: the raw r is already RMSD-controlled by construction.
Those groups are flagged (``rmsd_constant``) and their raw r is reported as the
controlled r.

Inputs
------
``--records_csv``  per-record CSV written by ``eval_boltzmann_independent.py``
                   (``*_records.csv`` / ``boltz_records.csv``): needs group_id,
                   pert_id, log_p_theta and one of negE_kT / E_eV.
``--meta_csv``     optional ``perturbation_meta.csv`` from the controlled Stage 1
                   (supplies ``rmsd_actual``).
``--samples_json`` optional Stage-1 JSON; RMSD is taken from ``rmsd_actual`` if
                   present, else recomputed (Kabsch) against the group's pert_id 0.

Usage
-----
    python scripts/analyze_partial_correlation.py \
        --records_csv <dir>/boltz_records.csv \
        --samples_json <dir>/boltz1/boltzmann_samples.json \
        --out_json <dir>/partial_corr.json --out_csv <dir>/partial_corr_groups.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from scipy import stats as _st
except Exception:                                       # pragma: no cover
    _st = None

RMSD_CONST_TOL = 1e-6


# ---------------------------------------------------------------------------

def _pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def partial_corr(x, y, z):
    """r(x, y | z) via the correlation-matrix formula. nan if degenerate."""
    rxy, rxz, ryz = _pearson(x, y), _pearson(x, z), _pearson(y, z)
    if not all(map(np.isfinite, (rxy, rxz, ryz))):
        return float("nan")
    den = math.sqrt(max(1e-15, (1 - rxz ** 2) * (1 - ryz ** 2)))
    return float((rxy - rxz * ryz) / den)


def _corr_p(r, n, k_ctrl=0):
    """Two-sided p for a (partial) correlation with n samples, k controls."""
    df = n - 2 - k_ctrl
    if df <= 0 or not np.isfinite(r) or abs(r) >= 1.0:
        return float("nan")
    t = r * math.sqrt(df / max(1e-15, 1 - r ** 2))
    if _st is None:
        return float("nan")
    return float(2 * _st.t.sf(abs(t), df))


def _ols(y, X):
    """Least squares with intercept prepended. Returns (beta, se, t)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    dof = max(1, len(y) - A.shape[1])
    s2 = float(resid @ resid) / dof
    try:
        cov = s2 * np.linalg.pinv(A.T @ A)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(A.shape[1], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    return beta, se, t


# ---------------------------------------------------------------------------

def load_records(records_csv: Path, meta_csv: Path | None,
                 samples_json: Path | None, kT_eV: float):
    """Return dict group_id -> list of dicts with log_p, negE_kT, rmsd, pert_id."""
    rows = []
    with open(records_csv) as f:
        for r in csv.DictReader(f):
            if str(r.get("xtb_ok", "True")).lower() in ("false", "0"):
                continue
            lp = r.get("log_p_theta", "")
            if lp in ("", None):
                continue
            neg = r.get("negE_kT", "")
            if neg in ("", None):
                e = r.get("E_eV", "")
                if e in ("", None):
                    continue
                neg = -float(e) / kT_eV
            rows.append({
                "group_id": int(float(r["group_id"])),
                "pert_id": int(float(r["pert_id"])),
                "log_p": float(lp),
                "negE_kT": float(neg),
                "n_atoms": int(float(r.get("n_atoms", 0) or 0)),
            })

    rmsd_lookup: dict[tuple[int, int], float] = {}
    extra: dict[tuple[int, int], dict] = {}
    if meta_csv is not None and Path(meta_csv).exists():
        with open(meta_csv) as f:
            for r in csv.DictReader(f):
                key = (int(float(r["group_id"])), int(float(r["pert_id"])))
                rmsd_lookup[key] = float(r["rmsd_actual"])
                extra[key] = {"scale": r.get("scale"), "mode": r.get("perturb_mode"),
                              "mol_id": r.get("mol_id")}
    elif samples_json is not None and Path(samples_json).exists():
        recs = json.load(open(samples_json))
        by_group: dict[int, list] = {}
        for rec in recs:
            by_group.setdefault(int(rec["group_id"]), []).append(rec)
        need_kabsch = any("rmsd_actual" not in rec for rec in recs)
        if need_kabsch:
            from cfm_mol.geom_perturb import kabsch_rmsd
        for gid, rs in by_group.items():
            ref = next((r for r in rs if int(r["pert_id"]) == 0), None)
            for rec in rs:
                key = (gid, int(rec["pert_id"]))
                if rec.get("rmsd_actual") is not None:
                    rmsd_lookup[key] = float(rec["rmsd_actual"])
                elif ref is not None:
                    rmsd_lookup[key] = kabsch_rmsd(
                        np.asarray(rec["positions"], float),
                        np.asarray(ref["positions"], float))
                extra[key] = {"scale": rec.get("scale"),
                              "mode": rec.get("perturb_mode"),
                              "mol_id": rec.get("mol_id")}
    else:
        raise SystemExit("need --meta_csv or --samples_json to supply RMSD")

    groups: dict[int, list] = {}
    n_missing = 0
    for row in rows:
        key = (row["group_id"], row["pert_id"])
        if key not in rmsd_lookup:
            n_missing += 1
            continue
        row["rmsd"] = rmsd_lookup[key]
        row.update({k: v for k, v in extra.get(key, {}).items()})
        groups.setdefault(row["group_id"], []).append(row)
    if n_missing:
        print(f"[partial] WARNING: {n_missing} records had no RMSD match and were "
              f"dropped (group_id/pert_id mismatch between the two files?)")
    return groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records_csv", type=Path, required=True)
    ap.add_argument("--meta_csv", type=Path, default=None)
    ap.add_argument("--samples_json", type=Path, default=None)
    ap.add_argument("--kT_eV", type=float, default=1.0)
    ap.add_argument("--drop_reference", action="store_true",
                    help="Exclude pert_id == 0 (the unperturbed reference, a "
                         "high-leverage point) from every statistic.")
    ap.add_argument("--n_rmsd_bins", type=int, default=4)
    ap.add_argument("--n_bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out_json", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, default=None)
    a = ap.parse_args()

    groups = load_records(a.records_csv, a.meta_csv, a.samples_json, a.kT_eV)
    if a.drop_reference:
        groups = {g: [r for r in rs if r["pert_id"] != 0] for g, rs in groups.items()}
    groups = {g: rs for g, rs in groups.items() if len(rs) >= 4}
    if not groups:
        raise SystemExit("no usable groups (need >= 4 records each)")

    rows_out = []
    raw_r, part_r, r_lp_rmsd, r_e_rmsd = [], [], [], []
    n_const = 0
    pooled = {"lp": [], "e": [], "rmsd": [], "rmsd_raw": [], "gid": []}

    for gid, rs in sorted(groups.items()):
        lp = np.array([r["log_p"] for r in rs], float)
        ne = np.array([r["negE_kT"] for r in rs], float)
        rm = np.array([r["rmsd"] for r in rs], float)
        rmsd_const = float(np.std(rm)) < RMSD_CONST_TOL * max(1.0, float(np.mean(rm)))
        r0 = _pearson(lp, ne)
        r_lr = _pearson(lp, rm)
        r_er = _pearson(ne, rm)
        if rmsd_const:
            n_const += 1
            rp = r0                       # RMSD held fixed by construction
        else:
            rp = partial_corr(lp, ne, rm)
        slope = float(np.polyfit(ne, lp, 1)[0]) if np.std(ne) > 1e-12 else float("nan")
        # slope of -E after partialling out RMSD (multiple regression)
        if rmsd_const:
            b_ctrl = slope
        else:
            beta, se, t = _ols(lp, np.column_stack([ne, rm]))
            b_ctrl = float(beta[1])
        rows_out.append({
            "group_id": gid, "n": len(rs), "mol_id": rs[0].get("mol_id", ""),
            "scale": rs[0].get("scale", ""), "mode": rs[0].get("mode", ""),
            "rmsd_mean": float(np.mean(rm)), "rmsd_std": float(np.std(rm)),
            "rmsd_constant": bool(rmsd_const),
            "pearson_r": r0, "partial_r_given_rmsd": rp,
            "r_logp_rmsd": r_lr, "r_negE_rmsd": r_er,
            "slope": slope, "slope_given_rmsd": b_ctrl,
            "p_raw": _corr_p(r0, len(rs)), "p_partial": _corr_p(rp, len(rs), 1),
        })
        for v, k in ((lp, "lp"), (ne, "e"), (rm, "rmsd")):
            pooled[k].extend((v - v.mean()).tolist())
        # Raw (un-demeaned) RMSD, appended in the SAME order as the demeaned
        # arrays -- the stratification below indexes both, so any ordering
        # mismatch would silently bin the wrong records.
        pooled["rmsd_raw"].extend(rm.tolist())
        pooled["gid"].extend([gid] * len(rs))
        if np.isfinite(r0):
            raw_r.append(r0)
        if np.isfinite(rp):
            part_r.append(rp)
        if np.isfinite(r_lr):
            r_lp_rmsd.append(r_lr)
        if np.isfinite(r_er):
            r_e_rmsd.append(r_er)

    plp = np.array(pooled["lp"]); pe = np.array(pooled["e"]); prm = np.array(pooled["rmsd"])

    # --- pooled (group-demeaned) statistics -------------------------------
    pooled_raw = _pearson(plp, pe)
    pooled_partial = (partial_corr(plp, pe, prm) if np.std(prm) > 1e-12 else pooled_raw)
    if np.std(prm) > 1e-12:
        beta, se, t = _ols(plp, np.column_stack([pe, prm]))
        pooled_beta_E, pooled_t_E = float(beta[1]), float(t[1])
        pooled_beta_rmsd, pooled_t_rmsd = float(beta[2]), float(t[2])
    else:
        beta, se, t = _ols(plp, pe.reshape(-1, 1))
        pooled_beta_E, pooled_t_E = float(beta[1]), float(t[1])
        pooled_beta_rmsd, pooled_t_rmsd = float("nan"), float("nan")

    # --- RMSD-stratified: bin on the RAW rmsd, correlate group-demeaned vars
    raw_rmsd_all = np.asarray(pooled["rmsd_raw"], dtype=float)
    strat = []
    if np.std(raw_rmsd_all) > 1e-12 and a.n_rmsd_bins > 1:
        edges = np.quantile(raw_rmsd_all, np.linspace(0, 1, a.n_rmsd_bins + 1))
        edges[0] -= 1e-9; edges[-1] += 1e-9
        binid = np.digitize(raw_rmsd_all, edges[1:-1])
        for b in range(a.n_rmsd_bins):
            m = binid == b
            if m.sum() < 5:
                continue
            strat.append({
                "bin": b,
                "rmsd_lo": float(edges[b]), "rmsd_hi": float(edges[b + 1]),
                "n": int(m.sum()),
                "rmsd_mean": float(raw_rmsd_all[m].mean()),
                "r_pooled_demeaned": _pearson(plp[m], pe[m]),
            })

    # --- bootstrap over GROUPS (the unit of independence) ------------------
    rng = np.random.default_rng(a.seed)
    boot_raw, boot_part, boot_delta = [], [], []
    arr_raw = np.array(raw_r); arr_part = np.array(part_r)
    if a.n_bootstrap and arr_part.size > 2:
        n = min(arr_raw.size, arr_part.size)
        for _ in range(a.n_bootstrap):
            idx = rng.integers(0, arr_part.size, arr_part.size)
            boot_part.append(float(arr_part[idx].mean()))
            idx2 = rng.integers(0, arr_raw.size, arr_raw.size)
            boot_raw.append(float(arr_raw[idx2].mean()))
        # paired delta (raw - partial) over the same groups
        paired = np.array([(r["pearson_r"], r["partial_r_given_rmsd"])
                           for r in rows_out
                           if np.isfinite(r["pearson_r"])
                           and np.isfinite(r["partial_r_given_rmsd"])])
        if paired.size:
            for _ in range(a.n_bootstrap):
                idx = rng.integers(0, paired.shape[0], paired.shape[0])
                s = paired[idx]
                boot_delta.append(float(s[:, 0].mean() - s[:, 1].mean()))

    def ci(v):
        if not v:
            return [float("nan"), float("nan")]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    summary = {
        "label": a.label,
        "records_csv": str(a.records_csv),
        "kT_eV": a.kT_eV,
        "drop_reference": bool(a.drop_reference),
        "n_groups": len(rows_out),
        "n_groups_rmsd_constant": n_const,
        "n_records": int(plp.size),
        "mean_pearson_r": float(np.mean(raw_r)) if raw_r else float("nan"),
        "median_pearson_r": float(np.median(raw_r)) if raw_r else float("nan"),
        "mean_partial_r_given_rmsd": float(np.mean(part_r)) if part_r else float("nan"),
        "median_partial_r_given_rmsd": (float(np.median(part_r)) if part_r
                                        else float("nan")),
        "mean_r_logp_rmsd": float(np.mean(r_lp_rmsd)) if r_lp_rmsd else float("nan"),
        "mean_r_negE_rmsd": float(np.mean(r_e_rmsd)) if r_e_rmsd else float("nan"),
        "ci95_mean_pearson_r": ci(boot_raw),
        "ci95_mean_partial_r": ci(boot_part),
        "ci95_delta_raw_minus_partial": ci(boot_delta),
        "pooled_demeaned_r": pooled_raw,
        "pooled_demeaned_partial_r": pooled_partial,
        "pooled_slope_negE": pooled_beta_E,
        "pooled_t_negE": pooled_t_E,
        "pooled_slope_rmsd": pooled_beta_rmsd,
        "pooled_t_rmsd": pooled_t_rmsd,
        "rmsd_strata": strat,
        "interpretation": (
            "If mean_partial_r_given_rmsd collapses toward 0 while mean_pearson_r "
            "is positive, the headline correlation is a geometry-distance artefact. "
            "If n_groups_rmsd_constant == n_groups the RMSD was fixed by "
            "construction and mean_pearson_r is ALREADY the controlled number."),
    }

    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(a.out_json, "w"), indent=2)
    if a.out_csv:
        with open(a.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            for r in rows_out:
                w.writerow(r)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
