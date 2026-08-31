#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig3_grid.py -- Figure 3: the matched six-cell causal grid.

WHAT THIS DRAWS
  (a) Per-cell mean per-parent Pearson r between log q_theta and -E_xTB/kT,
      with EVERY completed seed shown as a raw point, the cell mean as a bar,
      the standard error over seeds as a whisker, and the
      launched / completed / diverged triple printed under every cell.
      A grey band marks the seed range of cell A (the no-physics baseline) so
      that "A, B and C are indistinguishable" is read off the points, not from
      a table.
  (b) The six seed-level EFFECT SIZES as a forest plot: the difference between
      two cell means over completed seeds, with the propagated standard error
      of that difference, sqrt(sem_a^2 + sem_b^2), as a DESCRIPTIVE whisker.
      No inferential statistic is drawn anywhere in this figure: no Welch t,
      no p-value, no "significant"/"n.s." label.  The whisker is an SEM, not a
      confidence interval, and the panel says so.

DATA SOURCE -- ONE FILE PER RUN, NOTHING ELSE
  $BGFM_RUNS/clean_p0_<CELL>_s<N>/boltz_independent_records.csv
      cols: group_id,pert_id,n_atoms,charge,log_p_theta,E_eV,negE_kT,xtb_ok
  Recipe (the primary-endpoint convention for this grid):
      drop pert_id == 0 (the unperturbed reference), keep xtb_ok,
      per-parent Pearson r(log q_theta, -E/kT) -- the CSV column is named
      log_p_theta, the quantity is the clamped positional-flow density
      log q_theta -- meaned over the 120 parents.

  Launched / completed / diverged is a census over
      $BGFM_RUNS/../p0/p0_<CELL>_s<N>/lightning_logs/version_*/checkpoints/
  A completed run has midstep-step=30000.ckpt.

THREE TRAPS, ALL AVOIDED BY CONSTRUCTION (see the hard-coded CELLS table)
  (i)   the verified force cell is clean_p0_E_force_*, NOT clean_p0_E2_force_*;
  (ii)  clean_p0_D_energy_s6/s7/s9 and p0_F_both_s6..s10 are post-hoc relaunches
        outside the matched grid -- no seed is ever replaced, so seeds are hard
        restricted to s1..s5 of the six named cells;
  (iii) <tag>/boltz_independent.json's mean_pearson_r is reference-INCLUDED and
        is never read by this script.

INTEGRITY GATE
  Every per-seed r and every contrast recomputed here is checked against the
  independently verified values before anything is drawn.  A mismatch aborts.
  The Welch t values survive in the gate and in the audit JSON, where their job
  is to catch a pipeline change; they are never rendered into the image.

USAGE
  PY=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/omol25/bin/python
  $PY fig3_grid.py                    # -> out/fig3_grid.{pdf,png} + _data.json
  $PY fig3_grid.py --outdir /tmp/f --fmt png
"""

from __future__ import annotations

import argparse
import csv
import glob
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

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "out"
RUNS = Path(os.environ.get(
    "BGFM_RUNS", "/n/holylabs/woo_lab/Lab/yulili/bgfm/runs/eval_ours"))
P0_RUNS = Path(os.environ.get("BGFM_P0_RUNS", str(RUNS.parent / "p0")))

# --------------------------------------------------------------------------
# palette -- Okabe & Ito (2008), same semantic assignment as fig1_method.tex
# and make_experiment_figures.py.  Colour is never the only channel: every
# cell also carries its own marker shape, and cell F also carries a hatch.
# --------------------------------------------------------------------------
PALETTE = {
    "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "blue": "#0072B2", "vermilion": "#D55E00", "purple": "#CC79A7",
    "grey": "#7F7F7F", "greyd": "#555555", "greyl": "#BFBFBF",
}
# --------------------------------------------------------------------------
# DRAWN SIZE == PLACED SIZE (figure-editor pass, reviewer item 8).
#
# The float used to place this PDF at 0.76\linewidth while the canvas was drawn
# at the full 5.5 in column width, so every in-figure size rendered at 76% of
# what the code below asks for: an 8 pt label arrived on the page as 6.1 pt and
# the small notes as 4.6 pt.  Two things are fixed together here.
#
#   1. The float now includes the PDF at width=\linewidth (figures/fig3_grid.tex),
#      which is what its own header comment always said it should do.  So
#      PLACED_FRACTION is 1.00 and every point size below is a true point size
#      on the page, directly comparable with the 10 pt body text.
#   2. The canvas HEIGHT is set to what the figure already occupied on the page
#      (3.05 in drawn x 0.76 = 2.318 in rendered), so the float's vertical
#      footprint is unchanged and the nine-page main text is unaffected.  The
#      enlargement is bought entirely from the 24% of the column width that the
#      old placement threw away as margin.
#
# If the float is ever put back to a fractional width, set PLACED_FRACTION to
# that fraction; FULL_W then follows and the sizes stay honest.
# --------------------------------------------------------------------------
PLACED_FRACTION = 1.00
TEXTWIDTH_IN = 5.50
FULL_W = PLACED_FRACTION * TEXTWIDTH_IN          # 5.50 in
FULL_H = 2.30                                    # was 3.05 x 0.76 = 2.318 placed

# --------------------------------------------------------------------------
# the matched grid.  seeds are the COMPLETED seeds (a diverged run has no
# checkpoint and therefore no evaluation); `launched` is what was submitted.
# --------------------------------------------------------------------------
CELLS = [
    dict(key="A", run="A_fmonly", seeds=[1, 2, 3, 4, 5], launched=5,
         label="A", desc="FM only\nno energy",
         color=PALETTE["grey"], marker="o", hatch="", edge=PALETTE["grey"]),
    dict(key="B", run="B_flat", seeds=[2, 3, 4, 5], launched=5,
         label="B", desc="zero\nenergies",
         color=PALETTE["sky"], marker="s", hatch="", edge=PALETTE["sky"]),
    dict(key="C", run="C_scram", seeds=[1, 2, 3, 4, 5], launched=5,
         label="C", desc="scrambled\nenergies",
         color=PALETTE["purple"], marker="^", hatch="", edge=PALETTE["purple"]),
    dict(key="D", run="D_energy", seeds=[2, 3, 5], launched=5,
         label="D", desc="true\npairing",
         color=PALETTE["green"], marker="D", hatch="", edge=PALETTE["green"]),
    dict(key="E", run="E_force", seeds=[1, 2, 3, 4, 5], launched=5,
         label="E", desc="force\nonly",
         color=PALETTE["vermilion"], marker="v", hatch="",
         edge=PALETTE["vermilion"]),
    # F carries both terms, so it carries both colours: a vermilion (force)
    # body under a green (energy) hatch.  Its points and label are vermilion
    # because the force term is what distinguishes it from D.
    dict(key="F", run="F_both", seeds=[3, 5], launched=5,
         label="F", desc="force $+$\nenergy",
         color=PALETTE["vermilion"], marker="P", hatch="////",
         edge=PALETTE["green"], accent=PALETTE["vermilion"]),
]

CONTRASTS = [
    ("D", "A", "true pairing vs FM only"),
    ("D", "B", "true pairing vs zero energies"),
    ("D", "C", "true pairing vs scrambled"),
    ("B", "A", "zero energies vs FM only"),
    ("E", "A", "force only vs FM only"),
    ("F", "D", "force $+$ energy vs energy"),
]

# --------------------------------------------------------------------------
# independently verified reference values (the integrity gate).  These are NOT
# used for plotting -- everything plotted is recomputed from the CSVs -- they
# only assert that the recomputation reproduces the verified numbers.
# --------------------------------------------------------------------------
VERIFIED_SEEDS = {
    "A": [0.234, 0.200, 0.232, 0.221, 0.197],
    "B": [0.233, 0.125, 0.147, 0.243],
    "C": [0.232, 0.224, 0.168, 0.125, 0.174],
    "D": [0.452, 0.398, 0.412],
    "E": [0.105, 0.130, 0.044, 0.086, 0.058],
    "F": [0.179, 0.210],
}
VERIFIED_MEAN_SEM = {
    "A": (0.217, 0.008), "B": (0.187, 0.030), "C": (0.185, 0.020),
    "D": (0.421, 0.016), "E": (0.085, 0.016), "F": (0.194, 0.016),
}
VERIFIED_CONTRASTS = {          # (delta, t)
    ("D", "A"): (+0.204, +11.36), ("D", "C"): (+0.236, +9.28),
    ("D", "B"): (+0.234, +6.89), ("B", "A"): (-0.030, -0.96),
    ("E", "A"): (-0.132, -7.60), ("F", "D"): (-0.227, -10.05),
}
VERIFIED_LCD = {"A": (5, 5, 0), "B": (5, 4, 1), "C": (5, 5, 0),
                "D": (5, 3, 2), "E": (5, 5, 0), "F": (5, 2, 3)}


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Times",
                       "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        # every size here is now a TRUE point size on the page: the canvas is
        # drawn at the width the float places it at (see FULL_W above).
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "axes.titleweight": "bold",
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 7.2,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "legend.frameon": False,
        "lines.linewidth": 1.1, "lines.markersize": 3.4,
        "figure.dpi": 130, "savefig.dpi": 400,
        # NOT bbox="tight".  A tight bbox grows the canvas past FULL_W by
        # whatever the outermost annotation overhangs, and LaTeX then shrinks
        # the whole thing back to \linewidth -- which is exactly the 76%-of-
        # requested-size bug this pass exists to remove, only smaller.  The
        # margins below are explicit and every annotation is placed inside them.
        "savefig.bbox": None, "savefig.pad_inches": 0.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "hatch.linewidth": 0.45,
    })


def sem(x) -> float:
    x = np.asarray(x, float)
    if x.size < 2:
        return float("nan")
    return float(np.std(x, ddof=1) / math.sqrt(x.size))


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def seed_r(run: str, seed: int):
    """Mean per-parent Pearson r for one run, from its per-geometry CSV."""
    path = RUNS / f"clean_p0_{run}_s{seed}" / "boltz_independent_records.csv"
    if not path.is_file():
        raise SystemExit(f"MISSING per-geometry artefact: {path}")
    groups: dict[str, list] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if int(row["pert_id"]) == 0:                 # drop the reference
                continue
            if str(row["xtb_ok"]).strip().lower() != "true":
                continue
            groups.setdefault(row["group_id"], []).append(
                (float(row["log_p_theta"]), float(row["negE_kT"])))
    rs = []
    for pts in groups.values():
        a = np.asarray(pts, float)
        if a.shape[0] > 2 and a[:, 0].std() > 0 and a[:, 1].std() > 0:
            rs.append(float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]))
    if not rs:
        raise SystemExit(f"no usable parents in {path}")
    return float(np.mean(rs)), len(rs), str(path)


def census(run: str):
    """(launched, completed, diverged) over seeds s1..s5, from checkpoints."""
    launched = completed = 0
    for s in range(1, 6):
        d = P0_RUNS / f"p0_{run}_s{s}"
        if not d.is_dir():
            continue
        launched += 1
        hits = glob.glob(str(d / "lightning_logs" / "version_*" /
                             "checkpoints" / "midstep-step=30000.ckpt"))
        if hits:
            completed += 1
    return launched, completed, launched - completed


def collect():
    out = {}
    for c in CELLS:
        vals, ngroups, srcs = [], [], []
        for s in c["seeds"]:
            r, n, p = seed_r(c["run"], s)
            vals.append(r)
            ngroups.append(n)
            srcs.append(p)
        v = np.asarray(vals, float)
        lcd = census(c["run"])
        out[c["key"]] = dict(seeds=c["seeds"], r=v, mean=float(v.mean()),
                             sem=sem(v), n_parents=ngroups, sources=srcs,
                             lcd=lcd)
    return out


def verify(D) -> None:
    """Abort unless the recomputation reproduces every verified number."""
    bad = []
    for k, ref in VERIFIED_SEEDS.items():
        got = D[k]["r"]
        if len(got) != len(ref):
            bad.append(f"cell {k}: {len(got)} seeds, expected {len(ref)}")
            continue
        for i, (g, e) in enumerate(zip(got, ref)):
            if abs(g - e) > 5.1e-4:
                bad.append(f"cell {k} seed#{i}: {g:.4f} vs verified {e:.3f}")
    for k, (m, s) in VERIFIED_MEAN_SEM.items():
        if abs(D[k]["mean"] - m) > 5.1e-4 or abs(D[k]["sem"] - s) > 5.1e-4:
            bad.append(f"cell {k}: mean/sem {D[k]['mean']:.4f}/"
                       f"{D[k]['sem']:.4f} vs verified {m}/{s}")
    for k, ref in VERIFIED_LCD.items():
        if tuple(D[k]["lcd"]) != ref:
            bad.append(f"cell {k}: census {D[k]['lcd']} vs verified {ref}")
    for a, b, _ in CONTRASTS:
        d = D[a]["mean"] - D[b]["mean"]
        se = math.hypot(D[a]["sem"], D[b]["sem"])
        t = d / se
        rd, rt = VERIFIED_CONTRASTS[(a, b)]
        if abs(d - rd) > 5.1e-4 or abs(t - rt) > 5.1e-3:
            bad.append(f"contrast {a}-{b}: {d:+.4f}/t={t:+.3f} vs verified "
                       f"{rd:+.3f}/t={rt:+.2f}")
    if bad:
        print("INTEGRITY GATE FAILED -- nothing drawn:", file=sys.stderr)
        for b in bad:
            print("  " + b, file=sys.stderr)
        raise SystemExit(2)
    print("  integrity gate: all per-seed r, cell means/sems, "
          "launched/completed/diverged triples and six contrasts "
          "reproduce the verified values.")


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
def draw(D, outdir: Path, fmts):
    """Draw at the placed width, so every point size below is a page point size.

    FIGURE-EDITOR PASS (reviewer item 8).  Three changes, none of which touches
    a plotted value:
      * the canvas is 5.50 x 2.30 in and the float places it at width=\linewidth,
        so nothing is rescaled and nothing renders below 7 pt;
      * the in-panel line that repeated the population and the evaluator is
        GONE.  The caption already carries "120 held-out parents, reference
        geometry dropped, scored by an independent GFN2-xTB evaluator" verbatim,
        so the line bought nothing and cost the space it sat in;
      * the reclaimed space goes into the seed markers, the cell labels and the
        contrast annotations, which are what a reader actually has to resolve.
    """
    set_style()
    fig = plt.figure(figsize=(FULL_W, FULL_H))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.90, 1.28], wspace=0.175,
                          left=0.088, right=0.990, top=0.905, bottom=0.325)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # ---------------- panel (a) : cells ------------------------------------
    xs = np.arange(len(CELLS), dtype=float)
    base = D["A"]

    # Observed seed range of the THREE no-physics cells (A, B, C), drawn behind
    # everything.  Widening the band from A alone to A-B-C is what lets a reader
    # check the descriptive claim the caption makes -- every completed D seed
    # lies above the observed range of A, B and C -- straight off the plot, with
    # no test statistic standing in for it.
    band_lo = min(float(D[k]["r"].min()) for k in ("A", "B", "C"))
    band_hi = max(float(D[k]["r"].max()) for k in ("A", "B", "C"))
    axA.axhspan(band_lo, band_hi, color=PALETTE["greyl"],
                alpha=0.35, lw=0, zorder=0)
    axA.axhline(base["mean"], color=PALETTE["grey"], lw=0.7, ls=(0, (3, 2)),
                zorder=1)
    axA.annotate("observed seed range of the\nthree no-physics cells A, B, C",
                 xy=(0.52, band_hi + 0.004), xytext=(0.74, 0.418),
                 fontsize=7.0, color=PALETTE["greyd"], ha="center",
                 va="center", linespacing=1.20, zorder=6,
                 arrowprops=dict(arrowstyle="-", lw=0.55,
                                 color=PALETTE["grey"],
                                 shrinkA=2.0, shrinkB=1.0))

    for i, c in enumerate(CELLS):
        d = D[c["key"]]
        acc = c.get("accent", c["edge"])
        axA.bar(xs[i], d["mean"], width=0.62, facecolor=c["color"],
                alpha=0.26, edgecolor=c["edge"], linewidth=0.9,
                hatch=c["hatch"], zorder=2)
        axA.errorbar(xs[i], d["mean"], yerr=d["sem"], fmt="none",
                     ecolor=acc, elinewidth=1.1, capsize=2.6,
                     capthick=1.1, zorder=4)
        # raw seed points, deterministically spread so none is hidden
        n = len(d["r"])
        off = (np.arange(n) - (n - 1) / 2.0) * (0.46 / max(n - 1, 1))
        axA.scatter(xs[i] + off, d["r"], s=26, marker=c["marker"],
                    facecolors="white", edgecolors=acc, linewidths=1.05,
                    zorder=5, clip_on=False)
        dag = r"$^{\dagger}$" if d["lcd"][2] > 0 else ""
        axA.text(xs[i], max(d["mean"] + d["sem"], d["r"].max()) + 0.019,
                 f"{d['mean']:.3f}{dag}", ha="center", va="bottom",
                 fontsize=8.0, color=acc, fontweight="bold", zorder=6)

    axA.set_xticks(xs)
    axA.set_xticklabels([f"{c['label']}\n{c['desc']}" for c in CELLS],
                        fontsize=7.4, linespacing=1.22)
    for lbl in axA.get_xticklabels():
        lbl.set_linespacing(1.22)
    axA.set_xlim(-0.62, 5.62)
    axA.set_ylim(0.0, 0.545)
    axA.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    # two short lines rather than one long one: the axes are 1.36 in tall and a
    # single-line 8 pt label of the full phrase is longer than that, so it used
    # to run off the canvas.  The wording matches the caption's metric sentence.
    axA.set_ylabel("density\u2013energy\ncorrelation $r$", fontsize=8.0,
                   labelpad=2.5, linespacing=1.15)
    # 7.3 pt, not 8: at 8 pt this sentence runs into panel (b)'s title.  It is
    # still the largest text in the panel and the conditioning stays explicit.
    axA.set_title("(a)  Among completed runs, only the true pairing "
                  "improves ordering", fontsize=7.3, loc="left", pad=4.5)
    axA.tick_params(axis="x", length=0, pad=2.5)

    # launched / completed / diverged, one triple under each cell
    for i, c in enumerate(CELLS):
        l, comp, div = D[c["key"]]["lcd"]
        axA.annotate(f"{l}/{comp}/{div}", xy=(xs[i], 0), xycoords=("data",
                     "axes fraction"), xytext=(0, -32.5),
                     textcoords="offset points", ha="center", va="top",
                     fontsize=7.4, color=PALETTE["greyd"],
                     fontweight="bold" if div else "normal",
                     annotation_clip=False)
    axA.annotate("runs launched / completed / diverged"
                 r"      $\dagger$ mean conditional on completion",
                 xy=(0.5, 0), xycoords=("axes fraction", "axes fraction"),
                 xytext=(0, -44.0), textcoords="offset points", ha="center",
                 va="top", fontsize=7.0, color=PALETTE["greyd"], style="italic",
                 annotation_clip=False)
    # NOTE.  The population/evaluator stamp that used to sit here is deliberately
    # absent; it is in the caption, where a reader looks for provenance, and at
    # the size it was drawn it was the least legible object on the page.

    # ---------------- panel (b) : effect sizes -----------------------------
    # DESCRIPTIVE ONLY.  Each row is the difference between two cell means over
    # completed seeds; the whisker is the propagated standard error of that
    # difference, sqrt(sem_a^2 + sem_b^2).  That whisker is an SEM, not a
    # confidence interval, and the panel labels it as an SEM.  No Welch t, no
    # p-value and no significance label is drawn: the sign and the magnitude of
    # each difference, read against the seed clouds in panel (a), are the
    # evidence this figure offers.
    ys = np.arange(len(CONTRASTS))[::-1].astype(float)
    axB.axvline(0.0, color=PALETTE["greyd"], lw=0.8, zorder=1)
    for y, (a, b, name) in zip(ys, CONTRASTS):
        d = D[a]["mean"] - D[b]["mean"]
        se = math.hypot(D[a]["sem"], D[b]["sem"])
        # colour and marker encode the SIGN of the effect, nothing else.
        col = PALETTE["green"] if d > 0 else PALETTE["vermilion"]
        mk = "o" if d > 0 else "v"
        axB.errorbar(d, y, xerr=se, fmt="none", ecolor=col, elinewidth=1.15,
                     capsize=2.2, capthick=1.0, zorder=3)
        axB.scatter([d], [y], s=30, marker=mk, facecolors=col,
                    edgecolors="white", linewidths=0.7, zorder=4)
        # Anchor each row's text to the axes edge on the side the marker is NOT
        # on: positive effects read down the left margin, negative ones down the
        # right, so no label crosses its own marker.
        tx, ha = (-0.42, "left") if d > 0.0 else (0.42, "right")
        axB.text(tx, y + 0.28, f"{a} $-$ {b}", fontsize=7.6,
                 color=PALETTE["greyd"], ha=ha, va="center",
                 fontweight="bold")
        axB.text(tx, y - 0.22, f"{d:+.3f}", fontsize=7.4,
                 color=col, ha=ha, va="center")

    axB.set_ylim(-0.80, len(CONTRASTS) - 0.18)
    axB.set_xlim(-0.42, 0.42)
    axB.set_yticks([])
    axB.spines["left"].set_visible(False)
    axB.set_xticks([-0.2, 0.0, 0.2])
    axB.set_xlabel(r"difference in mean $r$", fontsize=8.0, labelpad=2.0)
    axB.set_title("(b)  effect sizes", fontsize=8.0, loc="left", pad=4.5)
    axB.annotate("helps $\\rightarrow$", xy=(0.42, -0.66), fontsize=7.0,
                 color=PALETTE["green"], ha="right", va="center")
    axB.annotate("$\\leftarrow$ hurts", xy=(-0.42, -0.66), fontsize=7.0,
                 color=PALETTE["vermilion"], ha="left", va="center")
    axB.annotate("whiskers: propagated SEM of the difference",
                 xy=(0.5, 0), xycoords=("axes fraction", "axes fraction"),
                 xytext=(0, -44.0), textcoords="offset points", ha="center",
                 va="top", fontsize=7.0, color=PALETTE["greyd"], style="italic",
                 annotation_clip=False)

    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in fmts:
        p = outdir / f"fig3_grid.{fmt}"
        fig.savefig(p, format=fmt)
        written.append(p)
        print(f"  wrote {p}  ({p.stat().st_size/1024:.0f} kB)")
    plt.close(fig)
    return written


def dump_json(D, outdir: Path):
    rec = {
        "figure": "fig3_grid",
        "endpoint": ("120 held-out parents; pert_id==0 (unperturbed "
                     "reference) dropped; xtb_ok only; per-parent Pearson "
                     "r(log q_theta, -E/kT); mean over parents; "
                     "evaluation n_ode = 12"),
        "source_pattern": str(RUNS / "clean_p0_<CELL>_s<N>" /
                              "boltz_independent_records.csv"),
        "census_pattern": str(P0_RUNS / "p0_<CELL>_s<N>" / "lightning_logs" /
                              "version_*" / "checkpoints" /
                              "midstep-step=30000.ckpt"),
        "excluded": ["clean_p0_E2_force_s1..s5 (different force variant)",
                     "clean_p0_D_energy_s6/s7/s9 (post-hoc relaunch)",
                     "p0_F_both_s6..s10 (post-hoc relaunch)",
                     "boltz_independent.json mean_pearson_r "
                     "(reference-included)"],
        "cells": {}, "contrasts": {},
    }
    for c in CELLS:
        d = D[c["key"]]
        rec["cells"][c["key"]] = dict(
            run=c["run"], seeds=d["seeds"], per_seed_r=[round(v, 6) for v in d["r"]],
            mean=round(d["mean"], 6), sem=round(d["sem"], 6),
            n_parents=d["n_parents"], launched=d["lcd"][0],
            completed=d["lcd"][1], diverged=d["lcd"][2], sources=d["sources"])
    for a, b, name in CONTRASTS:
        delta = D[a]["mean"] - D[b]["mean"]
        se = math.hypot(D[a]["sem"], D[b]["sem"])
        rec["contrasts"][f"{a}-{b}"] = dict(
            description=name, delta=round(delta, 6), welch_se=round(se, 6),
            welch_t=round(delta / se, 4), n_seeds_a=len(D[a]["r"]),
            n_seeds_b=len(D[b]["r"]))
    p = outdir / "fig3_grid_data.json"
    p.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"  wrote {p}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(DEFAULT_OUT))
    ap.add_argument("--fmt", default="pdf,png")
    a = ap.parse_args()
    outdir = Path(a.outdir)
    fmts = [f.strip() for f in a.fmt.split(",") if f.strip()]
    print(f"fig3_grid  runs={RUNS}")
    D = collect()
    verify(D)
    for c in CELLS:
        d = D[c["key"]]
        print(f"  {c['key']}  L/C/D {d['lcd'][0]}/{d['lcd'][1]}/{d['lcd'][2]}"
              f"  r = {d['mean']:.4f} +/- {d['sem']:.4f}  seeds "
              + " ".join(f"{v:.4f}" for v in d["r"]))
    draw(D, outdir, fmts)
    dump_json(D, outdir)


if __name__ == "__main__":
    main()
