#!/usr/bin/env python
"""One-glance dashboard over every advisor-requested experiment (P0/P1/P2/P3/P6).

Scans the product directories under $STORE and prints a single text report.
Anything that has not landed yet is shown as `pending` rather than omitted, so
the table doubles as a to-do list.

    P0  six-cell x five-seed ablation      runs/p0/, runs/eval_ours/clean_p0_*
    P2  calibration (NRV / slope / T_eff)  derived on the fly from boltz_records.csv
    P3  controlled perturbations           runs/eval_p3/*/{boltz.json,partial_corr.json}
    P6  likelihood-estimator validation    runs/eval_ours/p6_estimator/*/report.json
    P1  global (between-basin) ensemble    runs/eval_ours/global_ensemble*/*/global_ensemble_results.json

Read-only: it never launches or modifies anything. Safe to run from a login
node or from a short SLURM job chained behind the evals.

    python scripts/summarize_advisor_experiments.py
    python scripts/summarize_advisor_experiments.py --out $STORE/runs/ADVISOR_DASHBOARD.txt
    python scripts/summarize_advisor_experiments.py --no_calibration   # fast, skips CSV math
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from glob import glob

STORE_DEFAULT = "/n/holylabs/woo_lab/Lab/yulili/bgfm"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

P0_CELLS = [
    ("A_fmonly", "FM only (control)"),
    ("B_flat", "energy zeroed -> Var(log p)"),
    ("C_scram", "scrambled energies"),
    ("D_energy", "true energies"),
    ("E_force", "clean force, no distillation"),
    ("F_both", "force + energy"),
]
P0_SEEDS = [1, 2, 3, 4, 5]

# Legacy 30k-step ablation arms that already have products (the current headline).
LEGACY_ARMS = [
    "a3_energy_only", "a3_energy_only_s2", "a3_energy_only_s3", "a3_energy_only_s5",
    "a1_fm_only", "a1_fm_only_s2", "a1_fm_only_s3", "a1_fm_only_s4", "a1_fm_only_s5",
    "a6_energy_only_shuffled_s2", "a6_energy_only_shuffled_s3",
    "a6_energy_only_shuffled_stab_s5",
]

PENDING = "pending"


# ----------------------------------------------------------------- helpers ---
def fmt(x, n=3, width=None):
    if x is None:
        s = PENDING
    elif isinstance(x, str):
        s = x
    elif isinstance(x, float) and not math.isfinite(x):
        s = "nan"
    elif isinstance(x, float):
        s = f"{x:.{n}f}"
    else:
        s = str(x)
    return s.rjust(width) if width else s


def table(rows, headers, aligns=None):
    """Render a fixed-width text table (no external deps)."""
    if not rows:
        return "  (nothing found)\n"
    cols = len(headers)
    aligns = aligns or ["l"] + ["r"] * (cols - 1)
    cells = [[("" if c is None else str(c)) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for r in cells:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    def line(vals):
        out = []
        for i, v in enumerate(vals):
            out.append(v.ljust(widths[i]) if aligns[i] == "l" else v.rjust(widths[i]))
        return "  " + "  ".join(out).rstrip()
    sep = "  " + "  ".join("-" * w for w in widths)
    return "\n".join([line(list(headers)), sep] + [line(r) for r in cells]) + "\n"


def jload(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def newest_ckpt(rundir):
    pats = [
        os.path.join(rundir, "lightning_logs", "version_*", "checkpoints", "midstep-step=*.ckpt"),
        os.path.join(rundir, "lightning_logs", "version_*", "checkpoints", "*.ckpt"),
    ]
    best, best_mt = None, -1
    for p in pats:
        for f in glob(p):
            mt = os.path.getmtime(f)
            if mt > best_mt:
                best, best_mt = f, mt
        if best:
            break
    return best


def ckpt_step(path):
    if not path:
        return None
    m = re.search(r"step=(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


# ------------------------------------------------------------- calibration ---
_CAL = {"fn": None, "tried": False}


def calibration(records_csv, kT_eV=1.0, drop_reference=True):
    """P2 metrics for one records CSV. Returns dict or None."""
    if not (records_csv and os.path.exists(records_csv)):
        return None
    if not _CAL["tried"]:
        _CAL["tried"] = True
        try:
            sys.path.insert(0, os.path.join(REPO, "scripts"))
            from analyze_calibration import analyze_one  # noqa: E402
            _CAL["fn"] = analyze_one
        except Exception as exc:  # numpy/scipy missing, or script moved
            _CAL["fn"] = None
            _CAL["err"] = str(exc)
    if _CAL["fn"] is None:
        return None
    try:
        return _CAL["fn"](records_csv, kT_eV, drop_reference, n_boot=200)
    except Exception:
        return None


def _mean_r_from_csv(records_csv, drop_reference=True):
    """Fallback per-group Pearson r when scipy/analyze_calibration is unavailable."""
    if not (records_csv and os.path.exists(records_csv)):
        return None, 0
    groups = {}
    try:
        with open(records_csv) as fh:
            for row in csv.DictReader(fh):
                if drop_reference and str(row.get("pert_id", "")) == "0":
                    continue
                try:
                    lp = float(row["log_p_theta"])
                    ne = float(row["negE_kT"])
                except Exception:
                    continue
                if not (math.isfinite(lp) and math.isfinite(ne)):
                    continue
                groups.setdefault(row["group_id"], []).append((lp, ne))
    except Exception:
        return None, 0
    rs = []
    for vals in groups.values():
        if len(vals) < 3:
            continue
        xs = [v[0] for v in vals]
        ys = [v[1] for v in vals]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        sxx = sum((a - mx) ** 2 for a in xs)
        syy = sum((b - my) ** 2 for b in ys)
        if sxx <= 0 or syy <= 0:
            continue
        rs.append(sxy / math.sqrt(sxx * syy))
    return (sum(rs) / len(rs) if rs else None), len(rs)


# ------------------------------------------------------------------ scans ----
def scan_boltzmann_dir(d, do_cal=True):
    """One n=120-style eval product dir -> {r, nrv, slope, n_groups, ...}."""
    out = {"dir": d, "r": None, "n_groups": None, "nrv": None, "slope": None,
           "t_eff": None, "nrv_min": None, "state": PENDING}
    if not os.path.isdir(d):
        return out
    bj = None
    for name in ("boltz_independent.json", "boltz.json"):
        cand = os.path.join(d, name)
        if os.path.exists(cand):
            bj = jload(cand)
            break
    if bj:
        out["r"] = bj.get("mean_pearson_r")
        out["n_groups"] = bj.get("n_groups")
        out["state"] = "done"
    rc = None
    for name in ("boltz_records.csv", "boltz_independent_records.csv"):
        cand = os.path.join(d, name)
        if os.path.exists(cand):
            rc = cand
            break
    out["records_csv"] = rc
    if rc and out["r"] is None:
        r, ng = _mean_r_from_csv(rc)
        if r is not None:
            out["r"], out["n_groups"], out["state"] = r, ng, "done(csv)"
    if do_cal and rc:
        cal = calibration(rc)
        if cal:
            out["nrv"] = cal.get("nrv_median")
            out["slope"] = cal.get("median_slope")
            out["t_eff"] = cal.get("T_eff_eV_from_median_slope")
            p = cal.get("pooled") or {}
            out["nrv_min"] = p.get("nrv_min")
    if out["state"] == PENDING:
        # distinguish "running" (stage-1 partial output) from "not started"
        if glob(os.path.join(d, "boltz1", "*.json")):
            out["state"] = "stage1-only"
        elif os.listdir(d):
            out["state"] = "running"
    return out


def scan_p0(store, do_cal):
    rows = []
    for cell, desc in P0_CELLS:
        for seed in P0_SEEDS:
            arm = f"p0_{cell}_s{seed}"
            rundir = os.path.join(store, "runs", "p0", arm)
            ck = newest_ckpt(rundir) if os.path.isdir(rundir) else None
            train = f"step={ckpt_step(ck)}" if ck else (
                "started" if os.path.isdir(rundir) else PENDING)
            ev = scan_boltzmann_dir(os.path.join(store, "runs", "eval_ours", f"clean_{arm}"), do_cal)
            rows.append([arm, desc if seed == 1 else "", train, ev["state"],
                         fmt(ev["r"]), fmt(ev["nrv"]), fmt(ev["slope"]),
                         fmt(ev["n_groups"], 0)])
    return rows


def _legacy_dir(store, arm):
    """The n=120 eval dir for a legacy arm. The seed-1 runs are stored with an
    explicit `_s1` suffix in eval_ours even though the run dir has none."""
    base = os.path.join(store, "runs", "eval_ours")
    for cand in (f"wide_{arm}", f"wide_{arm}_s1", f"clean_{arm}", f"abl_{arm}"):
        p = os.path.join(base, cand)
        if os.path.isdir(p):
            return p
    return os.path.join(base, f"wide_{arm}")


def scan_legacy(store, do_cal):
    rows = []
    for arm in LEGACY_ARMS:
        ev = scan_boltzmann_dir(_legacy_dir(store, arm), do_cal)
        rows.append([arm, ev["state"], fmt(ev["r"]), fmt(ev["nrv"]),
                     fmt(ev["slope"]), fmt(ev["t_eff"], 2), fmt(ev["nrv_min"]),
                     fmt(ev["n_groups"], 0)])
    return rows


P3_MODES = ["fixed_rmsd", "normal_mode", "torsion", "bond_angle", "gaussian"]
P3_EXPECT_ARMS = [
    "a3_energy_only", "a3_energy_only_s2", "a3_energy_only_s3", "a3_energy_only_s5",
    "a1_fm_only", "a1_fm_only_s2", "a1_fm_only_s3", "a1_fm_only_s5",
]


def _split_p3_name(name):
    """`<tag>__<mode>__<scales><suffix>` -> (tag, mode, scales_with_suffix)."""
    parts = name.split("__")
    if len(parts) < 3:
        return name, "?", "?"
    return parts[0], parts[1], "__".join(parts[2:])


def scan_p3(store):
    rows, seen = [], set()
    for d in sorted(glob(os.path.join(store, "runs", "eval_p3", "*"))):
        if not os.path.isdir(d):
            continue
        tag, mode, scales = _split_p3_name(os.path.basename(d))
        bj = jload(os.path.join(d, "boltz.json")) or {}
        pc = jload(os.path.join(d, "partial_corr.json")) or {}
        state = "done" if bj else ("running" if os.listdir(d) else PENDING)
        seen.add((tag, mode))
        rows.append([
            tag, mode, scales, state,
            fmt(bj.get("mean_pearson_r")),
            fmt(pc.get("mean_partial_r_given_rmsd")),
            fmt(pc.get("pooled_slope_negE")),
            fmt(bj.get("n_groups"), 0), fmt(bj.get("n_records"), 0),
            "yes" if pc.get("n_groups_rmsd_constant") else "no",
        ])
    for tag in P3_EXPECT_ARMS:
        for mode in ("fixed_rmsd", "normal_mode", "torsion"):
            if (tag, mode) not in seen:
                rows.append([tag, mode, "-", PENDING, PENDING, PENDING, PENDING,
                             "-", "-", "-"])
    return rows


def p3_paired(store):
    """Pair each energy-arm P3 cell with the same-seed FM-only cell."""
    pairs = {"a3_energy_only": "a1_fm_only",
             "a3_energy_only_s2": "a1_fm_only_s2",
             "a3_energy_only_s3": "a1_fm_only_s3",
             "a3_energy_only_s5": "a1_fm_only_s5"}
    idx = {}
    for d in sorted(glob(os.path.join(store, "runs", "eval_p3", "*"))):
        if not os.path.isdir(d):
            continue
        tag, mode, scales = _split_p3_name(os.path.basename(d))
        bj = jload(os.path.join(d, "boltz.json"))
        if bj and bj.get("mean_pearson_r") is not None:
            idx[(tag, mode, scales)] = bj["mean_pearson_r"]
    rows = []
    for (etag, mode, scales), er in sorted(idx.items()):
        ctag = pairs.get(etag)
        if not ctag:
            continue
        cr = idx.get((ctag, mode, scales))
        if cr is None:
            continue
        rows.append([f"{mode} @ {scales}", etag.replace("a3_energy_only", "E"),
                     fmt(er), fmt(cr), fmt(er - cr)])
    return rows


def scan_p6(store):
    cell_rows, arm_rows = [], []
    roots = sorted(glob(os.path.join(store, "runs", "eval_ours", "p6_estimator", "*")))
    for d in roots:
        if not os.path.isdir(d):
            continue
        arm = os.path.basename(d)
        rep = jload(os.path.join(d, "report.json"))
        if not rep:
            n_cells = len(glob(os.path.join(d, "cells", "*.json")))
            arm_rows.append([arm, "running" if n_cells else PENDING, str(n_cells),
                             PENDING, PENDING, PENDING])
            continue
        cells = rep.get("cells", {})
        # headline = 12 steps / 2 probes (what the published eval uses)
        head = cells.get("steps12_probes2", {})
        # spread of mean r across ODE resolutions == the stability question
        by_steps = {}
        for name, c in cells.items():
            m = re.match(r"steps(\d+)_probes(\d+)", name)
            if not m:
                continue
            by_steps.setdefault(int(m.group(1)), {})[int(m.group(2))] = c
        rs = [c.get("mean_per_group_r") for st in by_steps.values()
              for c in st.values() if c.get("mean_per_group_r") is not None]
        spread = (max(rs) - min(rs)) if len(rs) > 1 else None
        arm_rows.append([arm, "done", str(len(cells)),
                         fmt(head.get("mean_per_group_r")),
                         fmt(head.get("estimator_sd_mean"), 2),
                         fmt(spread)])
        for steps in sorted(by_steps):
            for probes in sorted(by_steps[steps]):
                c = by_steps[steps][probes]
                cell_rows.append([arm, str(steps), str(probes),
                                  fmt(c.get("logp_mean"), 1),
                                  fmt(c.get("estimator_sd_mean"), 2),
                                  fmt(c.get("within_group_signal_sd_mean"), 2),
                                  fmt(c.get("snr_signal_over_noise"), 2),
                                  fmt(c.get("mean_per_group_r")),
                                  fmt(c.get("sem_per_group_r"))])
    if not arm_rows:
        arm_rows.append(["(none)", PENDING, "-", PENDING, PENDING, PENDING])
    return arm_rows, cell_rows


def _stat(v, key="median"):
    """eval_global_ensemble.py stores each summary field as
    {mean, median, sem, n}; older/simpler fields are bare numbers."""
    if isinstance(v, dict):
        return v.get(key, v.get("mean"))
    return v


def scan_p1(store):
    rows = []
    roots = sorted(glob(os.path.join(store, "runs", "eval_ours", "global_ensemble*", "*")))
    for d in roots:
        if not os.path.isdir(d):
            continue
        label = f"{os.path.basename(os.path.dirname(d))}/{os.path.basename(d)}"
        res = jload(os.path.join(d, "global_ensemble_results.json"))
        if not res:
            partial = os.path.join(d, "basin_logp_partial.json")
            state = "logp-partial" if os.path.exists(partial) else (
                "running" if os.path.isdir(d) and os.listdir(d) else PENDING)
            rows.append([label, state, "-", PENDING, PENDING, PENDING, PENDING,
                         PENDING, PENDING])
            continue
        s = res.get("summary", {})
        rows.append([
            label, "done", fmt(_stat(s.get("n_systems")), 0),
            fmt(_stat(s.get("r_pointwise"))),
            fmt(_stat(s.get("r_pointwise_partial_Rg"))),
            fmt(_stat(s.get("nrv_pointwise")), 1),
            fmt(_stat(s.get("nrv_basin_mass")), 2),
            fmt(_stat(s.get("frac_model_top1_is_data_basin"))),
            fmt(_stat(s.get("pooled_basin_pair_r"))),
        ])
    if not rows:
        rows.append(["(none)", PENDING, "-", PENDING, PENDING, PENDING, PENDING,
                     PENDING, PENDING])
    return rows


def scan_ceiling(store):
    rows = []
    for d in sorted(glob(os.path.join(store, "runs", "eval_ours", "ceiling*"))):
        res = jload(os.path.join(d, "ceiling_summary.json"))
        if not res:
            continue
        s = res.get("summary", res)
        rows.append([os.path.basename(d),
                     fmt(s.get("n_groups"), 0),
                     fmt(s.get("pearson_r_mean")),
                     fmt(s.get("pearson_r_median")),
                     fmt(s.get("slope_median") or s.get("median_slope")),
                     fmt(s.get("nrv_median")),
                     fmt(s.get("T_eff_eV_median"), 2)])
    return rows


# -------------------------------------------------------------- job status ---
def job_states(jid_file):
    if not os.path.exists(jid_file):
        return [], {}
    entries = []
    with open(jid_file) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split(None, 1)
            if not parts[0].isdigit():
                continue
            entries.append((parts[0], parts[1] if len(parts) > 1 else ""))
    if not entries:
        return [], {}
    jids = [e[0] for e in entries]
    states = {}
    try:
        out = subprocess.run(
            ["squeue", "-h", "-o", "%i %T", "--jobs", ",".join(jids)],
            capture_output=True, text=True, timeout=60).stdout
        for ln in out.splitlines():
            p = ln.split()
            if len(p) >= 2:
                states[p[0].split("_")[0]] = p[1]
    except Exception:
        pass
    missing = [j for j in jids if j not in states]
    if missing:
        try:
            out = subprocess.run(
                ["sacct", "-nX", "-P", "-o", "JobID,State", "-j", ",".join(missing)],
                capture_output=True, text=True, timeout=90).stdout
            for ln in out.splitlines():
                p = ln.split("|")
                if len(p) >= 2:
                    states.setdefault(p[0].split(".")[0].split("_")[0], p[1].split()[0])
        except Exception:
            pass
    return entries, states


# ------------------------------------------------------------------- main ----
def build_report(store, do_cal=True, jid_file=None):
    L = []
    w = L.append
    w("=" * 100)
    w("BGFM ADVISOR EXPERIMENT DASHBOARD")
    w(f"generated : {datetime.now().isoformat(timespec='seconds')}")
    w(f"store     : {store}")
    w(f"calibration (P2) metrics: {'on' if do_cal else 'off (--no_calibration)'}"
      "   |   all metrics drop the unperturbed reference conformer")
    w("=" * 100)

    w("")
    w("### P0 -- six-cell x five-seed ablation (r = per-group Pearson vs -E_xTB/kT; NRV/slope = P2 calibration)")
    w(table(scan_p0(store, do_cal),
            ["arm", "cell meaning", "train", "eval", "r", "NRV(med)", "slope(med)", "n_grp"],
            ["l", "l", "l", "l", "r", "r", "r", "r"]))

    w("### LEGACY 30k-step arms (current headline; n=120 wide evals)")
    w(table(scan_legacy(store, do_cal),
            ["arm", "eval", "r", "NRV(med)", "slope(med)", "T_eff(eV)", "NRV_min", "n_grp"],
            ["l", "l", "r", "r", "r", "r", "r", "r"]))

    cr = scan_ceiling(store)
    if cr:
        w("### P2 -- oracle ceiling (teacher eSEN scored against independent GFN2-xTB)")
        w(table(cr, ["ceiling run", "n_grp", "r mean", "r med", "slope med",
                     "NRV med", "T_eff(eV)"],
                ["l", "r", "r", "r", "r", "r", "r"]))

    w("### P3 -- controlled perturbations (geometry-distance confound)")
    w(table(scan_p3(store),
            ["arm", "mode", "scales", "state", "r", "partial r|RMSD", "slope",
             "n_grp", "n_rec", "RMSD fixed"],
            ["l", "l", "l", "l", "r", "r", "r", "r", "r", "r"]))
    pp = p3_paired(store)
    if pp:
        w("### P3 -- paired energy-arm minus FM-only (same geometries, same seed)")
        w(table(pp, ["mode @ scales", "seed", "r energy", "r fm-only", "delta"],
                ["l", "l", "r", "r", "r"]))

    arm_rows, cell_rows = scan_p6(store)
    w("### P6 -- likelihood-estimator validation (per arm)")
    w(table(arm_rows,
            ["arm", "state", "n_cells", "r @12steps/2probes", "est.SD", "r spread over grid"],
            ["l", "l", "r", "r", "r", "r"]))
    if cell_rows:
        w("### P6 -- full (ODE steps x Hutchinson probes) grid")
        w(table(cell_rows,
                ["arm", "steps", "probes", "log p mean", "est.SD", "signal SD",
                 "SNR", "mean r", "sem r"],
                ["l", "r", "r", "r", "r", "r", "r", "r", "r"]))

    w("### P1 -- global (between-basin) ensemble   [all values are per-system MEDIANS]")
    w(table(scan_p1(store),
            ["run", "state", "n_sys", "r pointwise", "r|Rg", "NRV within",
             "NRV basin-mass", "top1 hit", "r basin-pair"],
            ["l", "l", "r", "r", "r", "r", "r", "r", "r"]))

    if jid_file:
        entries, states = job_states(jid_file)
        w("### SUBMITTED JOBS (from advisor_jids.txt)")
        if not entries:
            w("  (no job-id file yet)\n")
        else:
            rows = [[j, states.get(j, "unknown"), note] for j, note in entries]
            counts = {}
            for r in rows:
                counts[r[1]] = counts.get(r[1], 0) + 1
            w(table(rows, ["jobid", "state", "purpose"], ["r", "l", "l"]))
            w("  totals: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
            w("")

    w("=" * 100)
    w("legend: pending = product dir absent; running = dir exists but no final JSON;")
    w("        stage1-only = log p computed, xTB/correlation step still missing.")
    w("        NRV = Var_k[log p + bE]/Var_k[bE]  (0 = perfect Boltzmann, >=1 = worse than a constant)")
    w("=" * 100)
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=os.environ.get("STORE", STORE_DEFAULT))
    ap.add_argument("--out", default=None, help="also write the report here")
    ap.add_argument("--jid_file", default=None,
                    help="default: <store>/logs/advisor_jids.txt")
    ap.add_argument("--no_calibration", action="store_true",
                    help="skip the per-CSV NRV/slope math (much faster)")
    args = ap.parse_args()

    jid_file = args.jid_file or os.path.join(args.store, "logs", "advisor_jids.txt")
    rep = build_report(args.store, do_cal=not args.no_calibration, jid_file=jid_file)
    print(rep)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        tmp = args.out + ".tmp"
        with open(tmp, "w") as fh:
            fh.write(rep + "\n")
        os.replace(tmp, args.out)  # atomic: readers never see a half-written table
        print(f"\n[written] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
