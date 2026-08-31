#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_experiment_figures.py -- BGFM / HBFM experiment figures for ICLR 2027.

Produces three figures, PDF (for the paper) + PNG (for quick viewing):

  fig2_boltzmann_scatter        log q_theta  vs  -E_xTB/kT, grouped by parent
                               molecule, one regression line per group, plus a
                               per-group-r distribution panel that also makes
                               the Simpson's-paradox point (pooled r^2 is tiny
                               while the mean per-group r is large).
  fig3_ablation_bars           ablation bars with SEM error bars and individual
                               seed dots; the Y-scrambling pair is highlighted.
  fig4_mechanism_decomposition waterfall splitting the +0.355 total energy-term
                               gain into a density-regularisation component and
                               a Boltzmann-information component.

DESIGN RULES ENCODED HERE (see paper/RESEARCH_figures.md for the reasoning)
  * ICLR 2027 is a SINGLE 5.5in-wide text column, 10pt Times, 11pt leading.
    FULL_W = 5.5in, HALF_W = 2.65in.  Figures are drawn at final size and
    never rescaled in LaTeX (use `width=\\linewidth` only on FULL_W figures).
  * All in-figure text is 7--8pt so it matches the 10pt body / 9pt caption.
  * Okabe--Ito colour-blind-safe palette, with the SAME semantic colour
    assignment as figures/fig1_method.tex.
  * Colour is never the only channel: bars also differ by hatch, series also
    differ by marker.
  * Every number that is not yet real is drawn in RED and tagged PLACEHOLDER.

DATA INTERFACE
  Real artefacts live under
    $BGFM_RUNS (default /n/holylabs/woo_lab/Lab/yulili/bgfm/runs/eval_ours)
  and are read from:
    <tag>/boltz_independent.csv   per-GROUP metrics, written by
                                  scripts/eval_boltzmann_independent.py
                                  cols: group_id,n_pert,pearson_r,spearman_r,
                                        r2,slope,ess_frac
    <tag>/boltz_independent.json  the run-level summary (mean/median r, ...)
    <tag>/boltz_records.csv       OPTIONAL per-PERTURBATION file, needed for a
                                  fully real fig2 scatter.  Columns (any of):
                                    group_id, pert_id, log_p_theta,
                                    and one of {negE_kT, E_eV, E_hartree}
                                  This file does NOT exist yet -- see
                                  dump_boltz_records.py in this directory.
  When a needed artefact is missing the script still runs: it synthesises the
  missing piece, keeps every real quantity it can (per-group r and slope are
  taken from the real CSV), and stamps the affected panel in red.

USAGE
  PY=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/omol25/bin/python
  $PY make_experiment_figures.py                      # all three, into ./out
  $PY make_experiment_figures.py --only fig3
  $PY make_experiment_figures.py --outdir /tmp/f --fmt png
  $PY make_experiment_figures.py --strict-real        # fail if any placeholder
  $PY make_experiment_figures.py --split seeds        # fig4: recompute split

Dependencies: numpy + matplotlib only (pandas is NOT installed in
envs/omol25).  scipy is used if present, otherwise pure-numpy fallbacks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
RUNS = Path(os.environ.get(
    "BGFM_RUNS", "/n/holylabs/woo_lab/Lab/yulili/bgfm/runs/eval_ours"))
HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "out"

# ---------------------------------------------------------------------------
# palette -- Okabe & Ito (2008).  Keep in sync with fig1_method.tex.
# ---------------------------------------------------------------------------
PALETTE = {
    "orange":    "#E69F00",   # physics oracle / DFT labels
    "sky":       "#56B4E9",
    "green":     "#009E73",   # L_energy  (global shape)  <- the good result
    "yellow":    "#F0E442",
    "blue":      "#0072B2",   # model / flow matching
    "vermilion": "#D55E00",   # L_force   (local gradient) <- the bad result
    "purple":    "#CC79A7",   # negative control (Y-scrambled)
    "grey":      "#7F7F7F",   # no-physics baseline
    "greyd":     "#555555",
    "greyl":     "#BFBFBF",
    "red":       "#E31A1C",   # PLACEHOLDER ONLY.  Never a data colour.
}
PH_RED = PALETTE["red"]

# figure widths, inches
FULL_W = 5.50
TWO3_W = 3.60
HALF_W = 2.65


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Times",
                       "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "axes.titleweight": "bold",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "figure.dpi": 130,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.012,
        "pdf.fonttype": 42,      # embed TrueType, no Type-3 (ICLR/arXiv safe)
        "ps.fonttype": 42,
        "hatch.linewidth": 0.5,
    })


# ---------------------------------------------------------------------------
# tiny stats helpers (no scipy dependency required)
# ---------------------------------------------------------------------------
def sem(x) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return float("nan")
    return float(np.std(x, ddof=1) / math.sqrt(x.size))


def welch(a, b):
    """Welch t-test on two small samples.  Returns (t, dof, p_two_sided)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size < 2 or b.size < 2:
        return float("nan"), float("nan"), float("nan")
    va, vb = np.var(a, ddof=1) / a.size, np.var(b, ddof=1) / b.size
    t = (a.mean() - b.mean()) / math.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), dof))
    except Exception:
        # normal approximation; fine for reporting an order of magnitude
        p = float(math.erfc(abs(t) / math.sqrt(2)))
    return float(t), float(dof), p


def permutation_p(A, B):
    """Exact two-sided permutation test over seed relabellings.

    Added 2026-08-03.  With 3--4 seeds per arm the Welch t's normality
    assumption is untestable, so the manuscript reports this alongside it; the
    figure must print the same statistic as Table 1.  Returns (ge, total, p).
    """
    import itertools
    pool = list(A) + list(B)
    obs = abs(np.mean(A) - np.mean(B))
    ge = tot = 0
    for combo in itertools.combinations(range(len(pool)), len(A)):
        a = [pool[i] for i in combo]
        b = [pool[i] for i in range(len(pool)) if i not in combo]
        tot += 1
        if abs(np.mean(a) - np.mean(b)) >= obs - 1e-12:
            ge += 1
    return ge, tot, ge / tot


def fmt_p(p):
    """p-value string that reads well in mathtext."""
    if not np.isfinite(p):
        return "p=n/a"
    if p < 1e-3:
        return "p<10^{-3}"
    if p < 1e-2:
        return f"p={p:.4f}".rstrip("0")
    return f"p={p:.3f}".rstrip("0")


def ols(x, y):
    """Least-squares line.  Returns (slope, intercept)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m), float(c)


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def read_csv_cols(path: Path):
    """Read a small CSV into {column: np.array(float or object)}."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    out = {}
    for k in rows[0].keys():
        if k is None:
            continue
        vals = [r.get(k, "") for r in rows]
        try:
            out[k] = np.array([float(v) if v not in ("", None) else np.nan
                               for v in vals], dtype=float)
        except (TypeError, ValueError):
            out[k] = np.array(vals, dtype=object)
    return out


def load_groups(tag: str):
    """Per-group metrics for one run tag, or None."""
    p = RUNS / tag / "boltz_independent.csv"
    if not p.is_file():
        return None
    d = read_csv_cols(p)
    if "pearson_r" not in d:
        return None
    return d


def load_summary(tag: str):
    p = RUNS / tag / "boltz_independent.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def load_records(tag: str):
    """Per-perturbation (group_id, log_p_theta, negE_kT) or None.

    This is the artefact fig2 really wants.  scripts/eval_boltzmann_independent.py
    computes these numbers internally but currently only persists the per-group
    aggregate, so the file normally does not exist -- see dump_boltz_records.py.
    """
    p = RUNS / tag / "boltz_records.csv"
    if not p.is_file():
        return None
    d = read_csv_cols(p)
    if "group_id" not in d or "log_p_theta" not in d:
        return None
    if "negE_kT" in d:
        neg = d["negE_kT"]
    elif "E_eV" in d:
        neg = -d["E_eV"] / KT_EV
    elif "E_hartree" in d:
        neg = -d["E_hartree"] * 27.211386245988 / KT_EV
    else:
        return None
    return {"group_id": d["group_id"], "log_p": d["log_p_theta"],
            "negE_kT": neg}


KT_EV = 1.0   # the eval and the configs both use kT = 1.0 eV; r^2 is kT-invariant


# ---------------------------------------------------------------------------
# the ablation registry
#
# `seeds_fallback` holds the REAL per-seed mean per-group Pearson r that were
# measured and are quoted in the paper.  They were cross-checked against
# <tag>/boltz_independent.json on 2026-08-03 and agree to 4 decimals, so the
# figure is correct with or without filesystem access.
# ---------------------------------------------------------------------------
ABLATIONS = [
    # The control arm has FIVE trained seeds.  "wide_a1_fm_only" (no suffix) is
    # the fifth: it trained to 30,000 steps with finite weights but was not
    # scored under the primary protocol when the other four were.  It has since
    # been scored under the identical protocol (120 parents, n_ode = 12, same
    # evaluation seed) and is now part of the arm wherever the arm is drawn.
    # Its permissive fallback value is the all-120/with-reference number from
    # out/primary_endpoint.json, the same quantity the other four carry.
    dict(key="fm_only",
         label="no\nphysics",
         tags=["wide_a1_fm_only_s2", "wide_a1_fm_only_s3",
               "wide_a1_fm_only_s4", "wide_a1_fm_only_s5",
               "wide_a1_fm_only"],
         seeds_fallback=[0.0750, 0.0829, 0.0647, 0.0771, 0.1434],
         color=PALETTE["grey"], hatch="", n_eval=120, note=""),
    dict(key="energy_shuffled",
         label="energy\nscrambled",
         tags=["wide_a6_energy_only_shuffled_s2",
               "wide_a6_energy_only_shuffled_s3",
               "wide_a6_energy_only_shuffled_stab_s5"],
         seeds_fallback=[0.2843, 0.3155, 0.4172],
         color=PALETTE["purple"], hatch="xxx", n_eval=120, note=""),
    dict(key="energy_true",
         label="energy\ntrue",
         tags=["wide_a3_energy_only_s1", "wide_a3_energy_only_s2",
               "wide_a3_energy_only_s3", "wide_a3_energy_only_s5"],
         seeds_fallback=[0.3768, 0.4025, 0.5246, 0.4181],
         color=PALETTE["green"], hatch="", n_eval=120, note=""),
    # ------------------------------------------------------------------
    # The earlier, smaller round.  REVISION 2026-08-03 (reviewer finding):
    # the previous version of this figure drew force_only (+0.109, n=30)
    # immediately beside fm_only (+0.075, n=120) and omitted the n=30
    # round's OWN baseline, so the eye read "force helps" -- the exact
    # inverse of the claim, and the cross-round comparison SPINE RULING R2
    # forbids.  The n=30 cells now form a separate, self-contained group
    # carrying their own baseline and their own energy cell, and both of
    # the n=30 control seeds are shown (0.223 and 0.126; the manuscript
    # previously quoted only the higher one).
    # ------------------------------------------------------------------
    dict(key="fm_only_n30",
         label="no\nphysics",
         tags=["abl_a1_fm_only", "abl_a1_fm_only_s2"],
         seeds_fallback=[0.2232, 0.1260],
         color=PALETTE["grey"], hatch="///", n_eval=30,
         note="2 seeds"),
    dict(key="force_only",
         label="force\nonly",
         tags=["abl_a2_force_only"],
         seeds_fallback=[0.1093],
         color=PALETTE["vermilion"], hatch="///", n_eval=30,
         note="1 seed"),
    dict(key="force_energy",
         label="force\n+energy",
         tags=["abl_a4_force_energy"],
         seeds_fallback=[0.2941],
         color=PALETTE["blue"], hatch="///", n_eval=30,
         note="1 seed"),
    dict(key="energy_only_n30",
         label="energy\nonly",
         tags=["abl_a3_energy_only", "abl_a3_energy_only_s2"],
         seeds_fallback=[0.4203, 0.4241],
         color=PALETTE["green"], hatch="///", n_eval=30,
         note="2 seeds"),
]

# The Y-scrambling contrast at larger n, AS MEASURED (inventory row Q5, closed
# 2026-08-12).  The predicted n=240 values an earlier draft carried here
# (delta=0.13, t=3.9, p=0.05) are REMOVED for the same reason the manuscript
# removed them from its results table: a prediction better than the measurement
# does not belong beside one.  Currently unused by any figure; kept so that this
# script's printed inventory matches tab:placeholders rather than contradicting it.
MEASURED_YSCRAMBLE = dict(delta=0.092, t=1.77, p=0.147, n=120)  # not significant


# ---------------------------------------------------------------------------
# PRIMARY ENDPOINT (added 2026-08-03, REVIEWER-RESPONSE PASS)
#
# The manuscript's pre-specified endpoint applies BOTH evaluation corrections at
# once: restrict to the 93 parents disjoint from the energy term's shard pool AND
# drop the unperturbed reference geometry from every group.  That cell cannot be
# computed from <tag>/boltz_independent.csv (which is already aggregated per
# group over all 9 geometries), so it is produced by
#     paper/figures/recompute_disjoint_noreference.py
# in envs/flowmol (it needs torch to build the val-molecule signature table) and
# cached here as out/primary_endpoint.json.  Figure 3 reads that cache so that
# the figure and Table 1 cannot drift apart.  If the cache is absent the figure
# falls back to the permissive metric and says so, loudly, in the ledger.
# ---------------------------------------------------------------------------
PRIMARY_ENDPOINT_JSON = DEFAULT_OUT / "primary_endpoint.json"
PRIMARY_KEY = "disjoint|noref"


def load_primary_endpoint():
    try:
        import json as _json
        return _json.loads(PRIMARY_ENDPOINT_JSON.read_text())["cells"]
    except Exception:
        return None


_PRIMARY = load_primary_endpoint()


def resolve_ablation(spec: dict, quiet=False):
    """Return (seed_means, source) where source is 'primary', 'disk' or 'fallback'."""
    if _PRIMARY is not None and spec.get("n_eval") == 120:
        vals = [_PRIMARY[t][PRIMARY_KEY] for t in spec["tags"] if t in _PRIMARY]
        if len(vals) == len(spec["tags"]):
            return np.array(vals, float), "primary"
    got, missing = [], []
    for tag in spec["tags"]:
        s = load_summary(tag)
        if s and "mean_pearson_r" in s:
            got.append(float(s["mean_pearson_r"]))
        else:
            g = load_groups(tag)
            if g is not None:
                got.append(float(np.nanmean(g["pearson_r"])))
            else:
                missing.append(tag)
    if missing or len(got) != len(spec["tags"]):
        if not quiet:
            print(f"    [{spec['key']}] using hard-coded real values "
                  f"(missing on disk: {', '.join(missing) or 'n/a'})")
        return np.array(spec["seeds_fallback"], float), "fallback"
    got = np.array(got, float)
    ref = np.array(spec["seeds_fallback"], float)
    if got.size == ref.size and np.max(np.abs(np.sort(got) - np.sort(ref))) > 5e-3:
        print(f"    !! [{spec['key']}] DISK DISAGREES WITH THE PAPER NUMBERS: "
              f"disk={np.sort(got).round(4).tolist()} "
              f"paper={np.sort(ref).round(4).tolist()}")
    return got, "disk"


# ---------------------------------------------------------------------------
# placeholder plumbing
# ---------------------------------------------------------------------------
class Ledger:
    """Collects every placeholder used, so the caller can audit the figure."""

    def __init__(self):
        self.items = []

    def add(self, figure, what, real_status):
        self.items.append((figure, what, real_status))

    def report(self):
        if not self.items:
            print("\nPLACEHOLDER INVENTORY: none -- every number is real.")
            return
        print("\nPLACEHOLDER INVENTORY (mirror these into the manuscript's "
              "PLACEHOLDER INVENTORY block):")
        for fig, what, status in self.items:
            print(f"  - {fig}: {what}\n      current real status: {status}")


LEDGER = Ledger()


def stamp(ax, text="PLACEHOLDER", loc="upper left", fs=6.0):
    """Red PLACEHOLDER stamp with a dashed red box."""
    xy = {"upper left": (0.02, 0.975, "left", "top"),
          "upper right": (0.98, 0.975, "right", "top"),
          "lower left": (0.02, 0.025, "left", "bottom"),
          "lower right": (0.98, 0.025, "right", "bottom")}[loc]
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, color=PH_RED,
            fontsize=fs, ha=xy[2], va=xy[3], zorder=50, linespacing=1.15,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=PH_RED,
                      lw=0.6, ls=(0, (2, 1.4)), alpha=0.95))


def save(fig, outdir: Path, name: str, fmts):
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in fmts:
        p = outdir / f"{name}.{fmt}"
        fig.savefig(p, format=fmt)
        paths.append(p)
    plt.close(fig)
    for p in paths:
        print(f"  wrote {p}  ({p.stat().st_size/1024:.0f} kB)")
    return paths


# ---------------------------------------------------------------------------
# fig2 : log p vs -E/kT, grouped
# ---------------------------------------------------------------------------
def synth_group(r_target, slope_target, n, rng, sx=30.0):
    """Points whose SAMPLE Pearson r and SAMPLE OLS slope equal the targets.

    Used only to *illustrate* real, measured (r, slope) pairs when the
    per-perturbation file is unavailable.  Gram--Schmidt construction gives an
    exact sample correlation, so nothing about the illustrated statistic is
    invented -- only the individual point positions are.
    """
    r = float(np.clip(r_target, -0.995, 0.995))
    x = rng.standard_normal(n)
    x = (x - x.mean()) / (x.std() + 1e-12)
    e = rng.standard_normal(n)
    e = e - x * float(e @ x) / float(x @ x)
    e = (e - e.mean()) / (e.std() + 1e-12)
    y = r * x + math.sqrt(max(0.0, 1.0 - r * r)) * e
    xs = x * sx
    # slope = r * sy / sx  =>  sy = slope * sx / r
    sy = abs(slope_target) * sx / abs(r) if abs(r) > 1e-3 else 20.0
    sy = float(np.clip(sy, 2.0, 400.0))
    return xs, y * sy


def fig2_boltzmann_scatter(outdir: Path, fmts, tag_energy="wide_a3_energy_only_s1",
                           tag_base="wide_a1_fm_only_s2", seed=0):
    print("fig2_boltzmann_scatter")
    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(FULL_W, 2.05))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.30],
                          wspace=0.42, left=0.075, right=0.985,
                          bottom=0.285, top=0.855)

    recs = load_records(tag_energy)
    groups = load_groups(tag_energy)
    mode = ("records" if recs is not None
            else "synth-from-real-r" if groups is not None else "synth")
    print(f"    scatter panels: mode={mode}")

    # choose three example parents spanning the real spread (cf. EnFlow Fig 4:
    # "molecules with different correlation strengths and noise levels")
    if groups is not None:
        rs = groups["pearson_r"]
        order = np.argsort(rs)
        pick_idx = [order[int(0.20 * (len(order) - 1))],
                    order[int(0.55 * (len(order) - 1))],
                    order[int(0.92 * (len(order) - 1))]]
        picks = [(int(groups["group_id"][i]), float(rs[i]),
                  float(groups["slope"][i]),
                  int(groups["n_pert"][i]) if "n_pert" in groups else 9)
                 for i in pick_idx]
    else:
        picks = [(0, 0.18, 0.20, 9), (1, 0.42, 0.35, 9), (2, 0.74, 0.62, 9)]

    axes = []
    for j, (gid, r_real, slope_real, npert) in enumerate(picks):
        ax = fig.add_subplot(gs[0, j])
        axes.append(ax)
        if recs is not None:
            m = recs["group_id"] == gid
            X = recs["negE_kT"][m]
            Y = recs["log_p"][m]
            X = X - X.mean()
            Y = Y - Y.mean()
        else:
            X, Y = synth_group(r_real, slope_real, max(npert, 6), rng)

        # colour by Boltzmann rank (cividis is CVD-optimised, unlike jet/viridis
        # for red-green deficiency it stays monotone in perceived lightness)
        rank = np.argsort(np.argsort(X))
        ax.scatter(X, Y, c=rank, cmap="cividis", s=13, lw=0.4,
                   edgecolors="white", zorder=3)
        m, c = ols(X, Y)
        xs = np.array([X.min(), X.max()])
        ax.plot(xs, m * xs + c, color=PALETTE["greyd"], lw=0.9,
                ls=(0, (3, 1.6)), zorder=2)
        ax.set_title(f"parent {gid}", pad=2.5, fontsize=7.5)
        ax.text(0.045, 0.93, f"$r={r_real:+.2f}$", transform=ax.transAxes,
                fontsize=7, va="top", color=PALETTE["greyd"])
        ax.axhline(0, color=PALETTE["greyl"], lw=0.4, zorder=1)
        ax.axvline(0, color=PALETTE["greyl"], lw=0.4, zorder=1)
        ax.tick_params(labelsize=6.5)
        if j == 0:
            ax.set_ylabel(r"$\log q_\theta$  (centred)", labelpad=1.5)
        if j == 1:
            ax.set_xlabel(r"$-E_{\mathrm{GFN2\text{-}xTB}}/kT$  (centred within parent)",
                          labelpad=1.5)

    if recs is None:
        msg = ("PLACEHOLDER point positions in the three left panels —\n"
               "their per-parent $r$ and OLS slope are REAL (measured)"
               if groups is not None else
               "PLACEHOLDER — fully synthetic\n(no eval CSV found)")
        fig.text(0.42, 0.010, msg, color=PH_RED, fontsize=5.4,
                 linespacing=1.25,
                 ha="center", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=PH_RED,
                           lw=0.6, ls=(0, (2, 1.4))))
        LEDGER.add(
            "fig2 panels 1-3",
            "individual (log q, -E/kT) points are synthesised",
            "per-parent r and OLS slope ARE real (from "
            f"{tag_energy}/boltz_independent.csv); only the point positions "
            "are drawn to match them.  Run dump_boltz_records.py to make the "
            "panel fully real.")

    # ---- panel 4: distribution of per-group r --------------------------
    ax = fig.add_subplot(gs[0, 3])
    # REVISION 2026-08-03: select by KEY, not by list position.  Appending the
    # n=30 cells to ABLATIONS made ABLATIONS[-1] the n=30 energy cell, and the
    # panel silently plotted +0.422 (n=30, 2 seeds) where the manuscript quotes
    # +0.430 (n=120, 4 seeds).
    by_key = {a["key"]: a for a in ABLATIONS}
    series = [
        ("energy", by_key["energy_true"]["tags"], PALETTE["green"], "-"),
        ("no physics", by_key["fm_only"]["tags"], PALETTE["grey"], "--"),
    ]
    bins = np.linspace(-1, 1, 25)
    any_real = False
    pooled = {}
    ymax = 0.0
    for lbl, tags, col, ls in series:
        vals, pr2 = [], []
        for t in tags:
            g = load_groups(t)
            if g is not None:
                vals.append(g["pearson_r"])
                s = load_summary(t)
                if s and "pooled_r2" in s:
                    pr2.append(float(s["pooled_r2"]))
        if vals:
            any_real = True
            v = np.concatenate(vals)
            pooled[lbl] = (float(np.mean(pr2)) if pr2 else float("nan"),
                           float(v.mean()))
            h, _ = np.histogram(v, bins=bins, density=True)
            ymax = max(ymax, float(h.max()))
            ax.hist(v, bins=bins, density=True, histtype="stepfilled",
                    color=col, alpha=0.22, lw=0)
            ax.hist(v, bins=bins, density=True, histtype="step",
                    color=col, lw=1.1, ls=ls,
                    # 3 decimals: the manuscript and the Fig. 2 caption quote
                    # +0.430 and +0.075, and a 2-decimal legend printed "+0.07",
                    # which read as a different number.
                    label=f"{lbl}  ($\\bar r={v.mean():+.3f}$)")
            ax.axvline(v.mean(), color=col, lw=0.9, ls=":")
    ax.set_xlabel("per-parent Pearson $r$", labelpad=1.5)
    ax.set_ylabel("density", labelpad=1.5)
    ax.set_xlim(-1, 1)
    if ymax > 0:
        ax.set_ylim(0, ymax * 1.62)      # headroom for the legend + note
    ax.set_title("all held-out parents", pad=2.5, fontsize=7.5)
    ax.legend(loc="upper left", handlelength=1.2, borderpad=0.05,
              labelspacing=0.22, fontsize=6.0, handletextpad=0.5)
    ax.tick_params(labelsize=6.5)
    if pooled:
        # the Simpson's-paradox point: pooling all parents destroys the signal
        txt = ",  ".join(f"{k} {v[0]:.2f}" for k, v in pooled.items())
        ax.text(0.035, 0.70,
                "pooled over all parents,\n"
                f"$r^2$: {txt}\n→ grouping is mandatory",
                transform=ax.transAxes, ha="left", va="top", fontsize=5.4,
                color=PALETTE["greyd"], linespacing=1.3)
    if not any_real:
        stamp(ax, "PLACEHOLDER\n(no CSVs found)", loc="upper right", fs=5.4)
        LEDGER.add("fig2 panel 4", "per-group r histogram",
                   "boltz_independent.csv not found under " + str(RUNS))

    return save(fig, outdir, "fig2_boltzmann_scatter", fmts)


# ---------------------------------------------------------------------------
# fig3 : ablation bars
# ---------------------------------------------------------------------------
def fig3_ablation_bars(outdir: Path, fmts):
    print("fig3_ablation_bars")
    resolved = []
    for spec in ABLATIONS:
        # REVISION 2026-08-03 (mandated item 4e): panel (a) previously placed the
        # n=30 round's cells beside the n=120 round's cells behind a dashed
        # divider.  A divider is not a substitute for not doing it: the bars are
        # still read against each other.  The n=30 round is now OUT of the
        # figure entirely and lives only in Table A3, whose "compare with" column
        # states the single legitimate comparator per cell.
        if spec.get("n_eval") != 120:
            continue
        seeds, src = resolve_ablation(spec)
        resolved.append(dict(spec, seeds=seeds, src=src,
                             mean=float(np.mean(seeds)), sem=sem(seeds)))
    on_primary = all(r["src"] == "primary" for r in resolved)
    if not on_primary:
        LEDGER.add("fig3 panel (a)",
                   "bars drawn on the PERMISSIVE metric (all 120 parents, "
                   "reference geometry included)",
                   "out/primary_endpoint.json is missing -- run "
                   "figures/recompute_disjoint_noreference.py in envs/flowmol")

    base = next(r for r in resolved if r["key"] == "fm_only")
    true = next(r for r in resolved if r["key"] == "energy_true")
    shuf = next(r for r in resolved if r["key"] == "energy_shuffled")

    t_be, _, p_be = welch(true["seeds"], base["seeds"])
    t_ts, _, p_ts = welch(true["seeds"], shuf["seeds"])
    d_be = true["mean"] - base["mean"]
    d_ts = true["mean"] - shuf["mean"]

    # ---- headline annotation -------------------------------------------
    # REVISION 2026-08-03: the annotation is now COMPUTED from whatever metric
    # the bars are drawn on, instead of printing a hard-coded rounding of the
    # permissive contrast.  On the primary endpoint, with all five control seeds
    # scored, this reads Delta=+0.169, t=5.82, exact permutation p=0.008 (1/126).
    # The four-control-seed reading it replaced was Delta=+0.167, t=5.16, 2/70.
    _ge, _tot, _pperm = permutation_p(list(true["seeds"]), list(base["seeds"]))
    HEADLINE_TEXT = (rf"$\Delta={d_be:+.3f}$,  $t={t_be:.2f}$,  "
                     rf"perm. $p={_pperm:.3f}$")

    fig = plt.figure(figsize=(FULL_W, 2.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.35, 1.0], wspace=0.30,
                          left=0.075, right=0.988, bottom=0.215, top=0.905)

    def note_below(ax, i, text):
        """Put a note underneath the (two-line) tick label, not on top of it."""
        ax.annotate(text, xy=(i, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -26), textcoords="offset points",
                    ha="center", va="top", fontsize=5.4,
                    color=PALETTE["greyd"], annotation_clip=False)

    # ---------------- (a) all ablation cells ----------------------------
    ax = fig.add_subplot(gs[0, 0])
    xs = np.arange(len(resolved))
    # REVISION 2026-08-03: axis limits rescaled -- the primary-endpoint values
    # top out near 0.43, where the old YMAX of 0.98 left the annotations floating.
    YMAX_A = 0.72
    for i, r in enumerate(resolved):
        ax.bar(i, r["mean"], width=0.66, color=r["color"], alpha=0.80,
               edgecolor=r["color"], lw=0.8, hatch=r["hatch"], zorder=2)
        if np.isfinite(r["sem"]):
            ax.errorbar(i, r["mean"], yerr=r["sem"], color="black", lw=0.9,
                        capsize=2.2, capthick=0.9, zorder=4)
        # individual seeds: with 3--4 seeds the raw points are more honest than
        # an error bar on its own
        jitter = np.linspace(-0.14, 0.14, len(r["seeds"]))
        ax.scatter(i + jitter, r["seeds"], s=7, facecolor="white",
                   edgecolor="black", lw=0.55, zorder=5)
        top = r["mean"] + (r["sem"] if np.isfinite(r["sem"]) else 0.0)
        ax.text(i, max(top, max(r["seeds"])) + 0.020, f"{r['mean']:+.3f}",
                ha="center", va="bottom", fontsize=6.4)
        if r["note"]:
            note_below(ax, i, r["note"])

    # highlight the Y-scrambling pair
    i_s = [r["key"] for r in resolved].index("energy_shuffled")
    i_t = [r["key"] for r in resolved].index("energy_true")
    lo, hi = min(i_s, i_t) - 0.46, max(i_s, i_t) + 0.46
    BAND_TOP = 0.470
    ax.add_patch(Rectangle((lo, -0.005), hi - lo, BAND_TOP + 0.005,
                           facecolor=PALETTE["yellow"], alpha=0.20, lw=0.7,
                           edgecolor=PALETTE["orange"], ls=(0, (3, 1.6)),
                           zorder=1))
    ax.text((lo + hi) / 2, BAND_TOP + 0.012, "Y-scrambling control",
            ha="center", va="bottom", fontsize=6.2,
            color=PALETTE["orange"], fontweight="bold")

    # REVISION 2026-08-03: the round divider and the two round labels were
    # removed together with the n=30 cells themselves (see the filter at the top
    # of this function).  Only one evaluation round is plotted, so there is
    # nothing to separate.
    ax.text(0.5, YMAX_A - 0.055,
            "primary endpoint: 93 disjoint parents, reference geometry excluded"
            if on_primary else
            "PERMISSIVE metric (all 120 parents, reference included)",
            ha="center", va="top", fontsize=6.0,
            color=PALETTE["greyd"] if on_primary else PH_RED)

    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in resolved], fontsize=6.2)
    ax.set_ylabel("mean per-parent Boltzmann $r$", labelpad=2)
    ax.set_ylim(-0.02, YMAX_A)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("(a) ablation of the two physics terms", loc="left", pad=3,
                 fontsize=8)
    ax.tick_params(axis="x", length=0, pad=2)

    # the headline contrast: a plain horizontal significance bracket
    y_br = 0.590
    ax.plot([0, 0, i_t, i_t], [y_br - 0.022, y_br, y_br, y_br - 0.022],
            lw=0.8, color="black", clip_on=False, zorder=6)
    ax.text(i_t / 2, y_br + 0.012, HEADLINE_TEXT,
            ha="center", va="bottom", fontsize=6.4)

    # REVISION 2026-08-03: the force bracket was removed with the n=30 cells.
    # The force result is reported in the text and in Table A3 only, where its
    # single legitimate comparator is named.

    # ---------------- (b) Y-scrambling zoom -----------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    YMAX_B = 0.72
    pair = [shuf, true]
    for i, r in enumerate(pair):
        ax2.bar(i, r["mean"], width=0.52, color=r["color"], alpha=0.80,
                edgecolor=r["color"], lw=0.8, hatch=r["hatch"], zorder=2)
        ax2.errorbar(i, r["mean"], yerr=r["sem"], color="black", lw=0.9,
                     capsize=2.4, capthick=0.9, zorder=4)
        jitter = np.linspace(-0.11, 0.11, len(r["seeds"]))
        ax2.scatter(i + jitter, r["seeds"], s=9, facecolor="white",
                    edgecolor="black", lw=0.6, zorder=5)
        ax2.text(i, max(r["mean"] + r["sem"], max(r["seeds"])) + 0.016,
                 f"{r['mean']:+.3f}", ha="center", va="bottom", fontsize=6.6)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["energy labels\nY-scrambled",
                         "true geometry\u2013energy\npairing"], fontsize=6.6)
    ax2.set_ylim(0, YMAX_B)
    ax2.set_ylabel("mean per-parent $r$", labelpad=2)
    ax2.set_title("(b) does the pairing matter?", loc="left", pad=3, fontsize=8)
    ax2.tick_params(axis="x", length=0, pad=2)

    y0 = 0.480
    _ges, _tots, _pperm_ts = permutation_p(list(true["seeds"]), list(shuf["seeds"]))
    _tsc_val, _, _psc = welch(shuf["seeds"], base["seeds"])
    ax2.plot([0, 0, 1, 1], [y0 - 0.018, y0, y0, y0 - 0.018], lw=0.8,
             color="black")
    ax2.text(0.5, y0 + 0.010,
             "true vs. scrambled\n"
             f"$\\Delta={d_ts:+.3f}$, $t={t_ts:.2f}$, perm. $p={_pperm_ts:.3f}$\n"
             "scrambled vs. no physics\n"
             f"$\\Delta={shuf['mean']-base['mean']:+.3f}$, $t={_tsc_val:.2f}$ (n.s.)",
             ha="center", va="bottom", fontsize=5.3, linespacing=1.35)
    # REVISION 2026-08-03: the red "projected n=240" box was REMOVED from this
    # panel.  It existed because at the permissive metric the Y-scrambling
    # contrast was not significant (Delta=+0.093, t=1.85) and the manuscript was
    # writing towards a larger evaluation.  On the primary endpoint the contrast
    # IS measured and IS significant (Delta=+0.143, t=4.73, exact permutation
    # p=0.029), so printing a projection beside it would be misleading rather
    # than conservative.  The n=240 evaluation is still running and the
    # corresponding \pending row survives in Table 1 and in the placeholder
    # inventory, which is where an unfinished job belongs.

    fig.text(0.985, 0.012,
             "Y-scrambling permutes the geometry\u2013energy pairing inside each "
             "parent; the energy multiset and the within-parent energy std "
             "(77.6 kcal/mol) are unchanged.",
             ha="right", va="bottom", fontsize=5.4, color=PALETTE["greyd"])

    print(f"    energy vs no-physics : Delta={d_be:+.4f} t={t_be:.2f} p={p_be:.2e}")
    print(f"    true vs Y-scrambled  : Delta={d_ts:+.4f} t={t_ts:.2f} p={p_ts:.3f}")
    return save(fig, outdir, "fig3_ablation_bars", fmts)


# ---------------------------------------------------------------------------
# fig4 : mechanism decomposition
# ---------------------------------------------------------------------------
def fig4_mechanism_decomposition(outdir: Path, fmts, split="facts"):
    print(f"fig4_mechanism_decomposition (split={split})")
    seeds = {}
    for spec in ABLATIONS:
        s, _ = resolve_ablation(spec, quiet=True)
        seeds[spec["key"]] = s
    base = float(np.mean(seeds["fm_only"]))
    true = float(np.mean(seeds["energy_true"]))
    shuf3 = float(np.mean(seeds["energy_shuffled"]))
    total = true - base

    # Two splits are computable and the manuscript now prints BOTH rather than
    # choosing one (revision 2026-08-03): the 2-seed scrambled mean
    # (0.284, 0.315) -> 0.2999 is the CONFIGURATION-MATCHED comparison and gives
    # 63/37; the 3-seed mean (0.3390) includes the differently-configured
    # stab_s5 seed and gives 74/26.  --split seeds remains the default because
    # it uses all data on record; the figure and the text both say that neither
    # is a clean measurement while the val shard is unscrambled.
    shuf2 = float(np.mean(sorted(seeds["energy_shuffled"])[:2]))
    reg_facts = shuf2 - base
    reg_seeds = shuf3 - base
    reg = reg_facts if split == "facts" else reg_seeds
    info = total - reg
    fr = 100.0 * reg / total
    fi = 100.0 * info / total
    print(f"    total={total:+.3f}  regularisation={reg:+.3f} ({fr:.0f}%)  "
          f"Boltzmann info={info:+.3f} ({fi:.0f}%)")
    print(f"    [split=facts uses the 2-seed scrambled mean {shuf2:.4f}; "
          f"split=seeds uses the 3-seed mean {shuf3:.4f} -> "
          f"{100*reg_seeds/total:.0f}%/{100*(total-reg_seeds)/total:.0f}%]")

    fig = plt.figure(figsize=(TWO3_W, 2.42))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.150, right=0.985, bottom=0.255, top=0.865)

    steps = [
        ("no physics\nbaseline", 0.0, base, PALETTE["grey"], "", True),
        ("density\nregularisation", base, reg, PALETTE["purple"], "xxx", False),
        ("Boltzmann\ninformation", base + reg, info, PALETTE["green"], "", False),
        ("$+\\,\\mathcal{L}_{\\mathrm{energy}}$\n(true pairing)", 0.0, true,
         PALETTE["green"], "", True),
    ]
    for i, (lbl, bot, h, col, hatch, absolute) in enumerate(steps):
        ax.bar(i, h, bottom=bot, width=0.62, color=col,
               alpha=0.85 if absolute else 0.70, edgecolor=col, lw=0.8,
               hatch=hatch, zorder=3)
        if absolute:
            ax.text(i, bot + h + 0.016, f"{bot + h:.3f}", ha="center",
                    va="bottom", fontsize=6.8, fontweight="bold")
        else:
            # SPINE ruling R4: the manuscript quotes the total gain as +0.355
            # (the raw seed means give +0.3556); print the manuscript rounding
            # so the figure and the text agree.
            ax.text(i, bot + h + 0.016, f"{h:+.3f}\n({100*h/total:.0f}% of "
                    f"+0.355)", ha="center", va="bottom", fontsize=6.4,
                    linespacing=1.15)
    # waterfall connectors
    for i in range(3):
        y = steps[i][1] + steps[i][2]
        ax.plot([i + 0.31, i + 1 - 0.31], [y, y], lw=0.7, ls=(0, (2.4, 1.6)),
                color=PALETTE["greyd"], zorder=2)
    ax.plot([2 + 0.31, 3 - 0.31], [true, true], lw=0.7, ls=(0, (2.4, 1.6)),
            color=PALETTE["greyd"], zorder=2)

    ax.set_xticks(range(4))
    ax.set_xticklabels([s[0] for s in steps], fontsize=6.5)
    ax.set_ylabel("mean per-parent Boltzmann $r$", labelpad=2)
    ax.set_ylim(0, 0.62)
    ax.set_title("what the energy term actually buys", loc="left", pad=3,
                 fontsize=8)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.axhline(base, color=PALETTE["grey"], lw=0.6, ls=":", zorder=1)

    ax.legend(handles=[
        Patch(facecolor=PALETTE["purple"], alpha=0.70, hatch="xxx",
              edgecolor=PALETTE["purple"], lw=0.8,
              label="survives Y-scrambling: generic\ndensity regularisation"),
        Patch(facecolor=PALETTE["green"], alpha=0.70,
              edgecolor=PALETTE["green"], lw=0.8,
              label="destroyed by Y-scrambling:\nBoltzmann information"),
    ], loc="upper left", fontsize=5.9, handlelength=1.1, handleheight=1.3,
        borderpad=0.15, labelspacing=0.45)

    # REVISION 2026-08-03: the scrambling control has two known defects (only
    # the training-split shard was scrambled; the third scrambled seed uses a
    # different estimator configuration).  Both bias the purple block upward,
    # so the split is a direction and not a measurement.  Say so on the figure
    # itself -- a caption cannot carry a caveat this load-bearing.
    ax.text(0.985, 0.965,
            "control is incomplete: only one of the two shards was scrambled,\n"
            "and one of its three seeds is not configuration-matched.\n"
            "Both biases inflate the purple block, so read this as a direction.",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.2,
            color=PALETTE["greyd"], linespacing=1.3)

    if split == "facts":
        fig.text(0.5, 0.008,
                 f"PLACEHOLDER split: derived from the 2-seed Y-scrambled\n"
                 f"mean {shuf2:.3f}; with all 3 scrambled seeds ({shuf3:.3f})\n"
                 f"it becomes {100*reg_seeds/total:.0f}%/"
                 f"{100*(total-reg_seeds)/total:.0f}%",
                 ha="center", va="bottom", fontsize=5.2, color=PH_RED,
                 linespacing=1.3,
                 bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=PH_RED,
                           lw=0.55, ls=(0, (2, 1.4))))
        LEDGER.add("fig4",
                   f"the {fr:.0f}%/{fi:.0f}% regularisation/Boltzmann split",
                   f"it is derived from the 2-seed Y-scrambled mean "
                   f"({shuf2:.4f}).  With all 3 scrambled seeds "
                   f"({shuf3:.4f}) the split is "
                   f"{100*reg_seeds/total:.0f}%/"
                   f"{100*(total-reg_seeds)/total:.0f}%.  Re-derive once the "
                   f"n=240 scrambled seeds land, and make the main text agree.")
    else:
        # RULING R1: this is the split the manuscript reports (74/26).  It is a
        # real measurement, so it gets a plain grey note, not a red placeholder.
        fig.text(0.5, 0.012,
                 f"Split computed from all {len(seeds['energy_shuffled'])} "
                 f"Y-scrambled seeds (mean {shuf3:.3f}); an earlier 2-seed mean "
                 f"({shuf2:.3f}) gave "
                 f"{100*reg_facts/total:.0f}%/"
                 f"{100*(total-reg_facts)/total:.0f}%.",
                 ha="center", va="bottom", fontsize=5.2,
                 color=PALETTE["greyd"], linespacing=1.3)
    return save(fig, outdir, "fig4_mechanism_decomposition", fmts)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fmt", default="pdf,png",
                    help="comma list of pdf,png,svg (default pdf,png)")
    ap.add_argument("--only", default="", help="fig2 | fig3 | fig4")
    ap.add_argument("--runs", type=Path, default=None,
                    help="override the eval_ours directory")
    ap.add_argument("--split", choices=("facts", "seeds"), default="seeds",
                    help="fig4: 'seeds' (DEFAULT, SPINE ruling R1) computes the "
                         "74/26 split the manuscript reports, from all "
                         "scrambled seeds on record; 'facts' reproduces the "
                         "superseded 63/37 split (2-seed scrambled mean)")
    ap.add_argument("--strict-real", action="store_true",
                    help="exit non-zero if any placeholder was needed")
    a = ap.parse_args(argv)

    global RUNS
    if a.runs is not None:
        RUNS = a.runs
    fmts = [f.strip() for f in a.fmt.split(",") if f.strip()]

    set_style()
    print(f"eval_ours root : {RUNS}  (exists={RUNS.is_dir()})")
    print(f"output dir     : {a.outdir}")
    want = a.only.strip().lower()
    if want in ("", "fig2"):
        fig2_boltzmann_scatter(a.outdir, fmts)
    if want in ("", "fig3"):
        fig3_ablation_bars(a.outdir, fmts)
    if want in ("", "fig4"):
        fig4_mechanism_decomposition(a.outdir, fmts, split=a.split)
    LEDGER.report()
    if a.strict_real and LEDGER.items:
        print("\nSTRICT: placeholders present -> exit 1")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
