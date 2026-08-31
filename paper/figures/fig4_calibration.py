#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
fig4_calibration.py -- builds Figure 4, "value supervision improves ordering
before calibration".

USAGE
    /n/holylabs/woo_lab/Lab/yulili/bgfm/envs/omol25/bin/python \
        /n/home04/yulili/bgfm/paper/figures/fig4_calibration.py

OUTPUT
    figures/out/fig4_calibration.pdf       <- what LaTeX consumes
    figures/out/fig4_calibration.png       <- quick look only
    figures/out/fig4_calibration_data.json <- every number plotted, with the
                                              file it came from

NOTATION
    The measured density is the CLAMPED POSITIONAL-FLOW DENSITY q_theta: the
    endpoint discrete labels are held fixed and only the positional ODE is
    reversed.  It is not the joint generator's conditional density, so the
    drawn annotation reads \log q_\theta.  NRV is
    Var(log q_theta + E/kT) / Var(E/kT), and a constant log q_theta scores 1.

REVISION C6 (correctness fix, not a wording change)
    The previous version overlaid the continuous curve NRV = 1 - r^2 on a plane
    whose points are (mean-over-parents r, median-over-parents NRV).  That curve
    is NOT a valid boundary for such an aggregate: the per-parent identity is
    NRV_min(m) = 1 - r_m^2, and averaging it gives mean_m[1 - r_m^2], which is
    not 1 - (mean_m r_m)^2.  The curve has been REMOVED.  In its place the
    optimally rescaled bound is reported the only way it can be reported
    honestly: as its own aggregate statistic, per seed, in panel (b).
    Verified per-seed NRV_min (mean over the 93 parents, as
    scripts/analyze_calibration.py aggregates it):
        FM-only 0.810 0.848 0.816 0.847 0.803 -> 0.825
        Value   0.782 0.753 0.714 0.737       -> 0.747
    So the optimally rescaled bound IMPROVES while the raw NRV WORSENS.  That
    contrast is the figure's point and it is now drawn rather than implied.

POPULATION (all panels)
    The primary endpoint: the 93 held-out parents disjoint from the value term's
    shard pool, with the unperturbed reference geometry dropped, leaving up to
    eight displaced geometries per parent scored by the independent GFN2-xTB
    evaluator.  5 seeds for FM-only, 4 for value supervision.  Every panel shows
    all raw seed points.

PANELS AND THEIR SOURCE FILES  (nothing here is scraped from prose or a PDF)
    (a) ordering r (x) against raw NRV (y), one point per seed, per arm mean
        with sem crosses, the constant-log-density line at NRV = 1, the ideal
        point (1, 0), and the teacher-evaluator agreement reference.
          figures/out/scale_primary.json
            ["dropref"][arm]["per_seed"][seed]["primary_93"]
            fields "pearson_r" (MEAN over the 93 parents) and
                   "nrv"       (MEDIAN over the 93 parents)
    (b) raw NRV against NRV_min, the same seeds, one connector per seed.
          same records, fields "nrv" and "nrv_min" (nrv_min is the MEAN over the
          93 parents, matching analyze_calibration.py's aggregation).
    (c) implied kT_eff per seed, both arms, against the kT = 1.0 eV target.
          same records, field "T_eff_eV".
    All three are cross-checked at run time against the raw per-parent blocks of
      runs/eval_ours/calibration/calibration_dropref.json
        ["arms"][arm_seed]["per_group"][gid]["pearson_r"|"nrv"|"nrv_min"|"slope"]
    restricted to the 93 ids in figures/out/primary_endpoint.json; the two routes
    agree to 4 decimals and the script asserts it.

THE TEACHER-EVALUATOR AGREEMENT REFERENCE (a reference, never a ceiling)
    Computed here from
      runs/eval_ours/ceiling_full_dropref/ceiling_per_group.csv
    restricted to the same 93 group ids, using exactly the conventions the model
    arms use: r = mean over parents, NRV = median over parents, NRV_min = mean
    over parents of the per-parent 1 - r_m^2 (that per-parent identity is exact
    and is verified at run time against the model arms' stored nrv_min), and
    kT_eff = kT / median slope.  That gives r = 0.9271 (the one verified teacher
    number), NRV = 0.0817, NRV_min = 0.0954, kT_eff = 1.251 eV.  It is the
    the agreement the GFN2-xTB evaluator reaches with the eSEN training teacher
    whose energies it substitutes for the model log density on this population, not
    an attainable bound for a generative model, and it is labelled that way in
    the figure and the caption.

WHAT IS NOT DRAWN
    No 1 - r^2 curve (see REVISION C6).  No oracle NRV from any earlier round or
    PDF; no cross-round comparison; the scrambled control is argued in Figure 2
    and the six-cell grid in Figure 3, so neither appears here.

READING (what the figure must make obvious)
    This is a SCOPE figure for a method that works, not a failure plot.  Three
    of the four quantities move the way the objective intends: ordering r rises
    0.200 -> 0.369, the optimally rescaled residual NRV_min falls 0.825 ->
    0.747, and the implied kT_eff moves 8.02 -> 2.34 eV toward the numerical
    kT = 1.0 eV target.  The fourth, the RAW NRV, rises 1.402 -> 2.144, past the
    value 1 that a constant log-density already scores.  Read together, panels
    (b) and (c) locate the remaining residual in the MAGNITUDE of the density
    response rather than in its ranking, which is why raw calibration is a
    separate target needing its own objective and not a larger weight on this
    one.  No test statistic is drawn anywhere in this figure.

STYLE
    Mirrors figures/make_experiment_figures.py and figures/fig2_ordering.py:
    5.5 in single column, 7-8 pt serif in-figure text, Okabe-Ito palette with
    the fixed semantics grey = FM-only baseline, green = value/energy,
    orange = teacher-evaluator agreement.  Vermilion (force) and the retired
    red never appear here.  Drawn at final size; include with width=\linewidth.
"""

from __future__ import annotations

import csv
import json
import math
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent            # .../paper/figures
OUT = HERE / "out"
REPO = HERE.parent.parent                          # .../bgfm
RUNS = Path(REPO / "runs" / "eval_ours")

F_SCALE = OUT / "scale_primary.json"
F_PRIMARY = OUT / "primary_endpoint.json"
F_CALIB = RUNS / "calibration" / "calibration_dropref.json"
F_CEIL = RUNS / "ceiling_full_dropref" / "ceiling_per_group.csv"

KT_TRAIN_EV = 1.0          # the training target; also calibration_dropref["kT_eV"]

# --------------------------------------------------------------------------
# style (kept byte-identical in spirit to make_experiment_figures.set_style)
# --------------------------------------------------------------------------
PALETTE = {
    "orange":    "#E69F00",   # teacher-evaluator agreement (reference)
    "green":     "#009E73",   # value supervision  <- the proposed arm
    "grey":      "#7F7F7F",   # no-physics baseline (FM-only)
    "greyd":     "#555555",
    "greyl":     "#BFBFBF",
    "ink":       "#222222",
}
# --------------------------------------------------------------------------
# DRAWN SIZE == PLACED SIZE (figure-editor pass, reviewer item 8).
#
# The float used to place this PDF at 0.76\linewidth while the canvas was drawn
# at the full 5.5 in column width, so every in-figure size rendered at 76% of
# what this file asks for: an 8 pt axis label arrived on the page as 6.1 pt and
# the annotations as 4.7 pt.  Two things are fixed together.
#
#   1. The float now includes the PDF at width=\linewidth (see
#      figures/fig4_calibration.tex, whose header always said it should), so
#      PLACED_FRACTION is 1.00 and every point size below is a true point size,
#      directly comparable with the 10 pt body text.
#   2. The canvas HEIGHT is what the figure already occupied on the page
#      (2.72 in drawn x 0.76 = 2.067 in rendered), so the float's vertical
#      footprint does not grow and the nine-page main text is unaffected.  The
#      three panels get bigger out of the 24% of the column width that the
#      fractional placement used to throw away as margin, and out of a more even
#      split between them: panel (c) was 0.61 in wide and is now 0.99 in.
#
# If the float is ever put back to a fractional width, set PLACED_FRACTION to
# that fraction; FULL_W follows and the sizes stay honest.
# --------------------------------------------------------------------------
PLACED_FRACTION = 1.00
TEXTWIDTH_IN = 5.50
FULL_W = PLACED_FRACTION * TEXTWIDTH_IN          # 5.50 in
FULL_H = 2.03                                    # was 2.72 x 0.76 = 2.067 placed


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "axes.titleweight": "bold",
        # every size here is now a TRUE point size on the page (see FULL_W).
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.0,
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
        "lines.linewidth": 1.1,
        "lines.markersize": 3.6,
        "figure.dpi": 130,
        "savefig.dpi": 400,
        # NOT bbox="tight": a tight bbox grows the canvas past FULL_W by whatever
        # the outermost annotation overhangs, and LaTeX then shrinks the whole
        # thing back to \linewidth -- the same "renders smaller than requested"
        # bug this pass removes, only subtler.  Margins below are explicit.
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def sem(x) -> float:
    x = np.asarray(x, float)
    if x.size < 2:
        return float("nan")
    return float(np.std(x, ddof=1) / math.sqrt(x.size))


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
ARMS = [
    # key in scale_primary.json,  label,             colour,             marker
    ("a1_fm_only", "FM-only", PALETTE["grey"], "o"),
    ("a3_energy", "Value (ours)", PALETTE["green"], "s"),
]


def load_arms():
    """Per-seed (r, NRV, NRV_min, kT_eff) on the 93-parent primary endpoint.

    The stored JSON field is named "T_eff_eV"; that is a data-file field name and
    is never drawn.  Every label, annotation and caption reads kT_eff, which is
    what the quantity is: an energy in eV, kT divided by the fitted slope.
    """
    d = json.loads(F_SCALE.read_text())
    assert d["n_parents_primary"] == 93, d["n_parents_primary"]
    out = {}
    for key, label, colour, marker in ARMS:
        per_seed = d["dropref"][key]["per_seed"]
        seeds = sorted(per_seed)
        rec = {"label": label, "colour": colour, "marker": marker,
               "seeds": seeds,
               "r": [per_seed[s]["primary_93"]["pearson_r"] for s in seeds],
               "nrv": [per_seed[s]["primary_93"]["nrv"] for s in seeds],
               "nrv_min": [per_seed[s]["primary_93"]["nrv_min"] for s in seeds],
               "t_eff": [per_seed[s]["primary_93"]["T_eff_eV"] for s in seeds],
               "n_groups": [per_seed[s]["primary_93"]["n_groups"] for s in seeds]}
        assert all(n == 93 for n in rec["n_groups"]), rec["n_groups"]
        out[key] = rec
    return out


def crosscheck_from_raw(arms) -> dict:
    """Recompute (r, NRV, NRV_min) from the per-parent blocks and assert equality.

    r        = MEAN   over the 93 parents of the per-parent Pearson correlation
    NRV      = MEDIAN over the 93 parents of the per-parent NRV
    NRV_min  = MEAN   over the 93 parents of the per-parent NRV_min
    This is the convention recompute_scale_primary.py (and, upstream of it,
    scripts/analyze_calibration.py) uses; verifying it here means the figure
    never depends on a cached aggregate alone.

    The per-parent identity NRV_min(m) = 1 - r_m^2 is also asserted, because
    panel (b) and the teacher reference both rely on it.
    """
    ids = {str(g) for g in json.loads(F_PRIMARY.read_text())["disjoint_group_ids"]}
    assert len(ids) == 93, len(ids)
    raw = json.loads(F_CALIB.read_text())
    assert raw["drop_reference"] is True
    assert abs(raw["kT_eV"] - KT_TRAIN_EV) < 1e-12
    checked = {}
    for key, rec in arms.items():
        for i, seed in enumerate(rec["seeds"]):
            pg = raw["arms"][seed]["per_group"]
            sel = [pg[g] for g in ids if g in pg]
            assert len(sel) == 93, (seed, len(sel))
            # up to 8 displaced geometries: a handful of parents lose one or
            # more to a GFN2-xTB single-point failure.  The reference geometry
            # is already dropped upstream (drop_reference is True), so nothing
            # here can exceed 8.
            assert all(2 <= m["n_pert"] <= 8 for m in sel), \
                sorted({m["n_pert"] for m in sel})
            # per-parent identity that licenses the NRV_min reference marker
            for m in sel:
                assert abs(m["nrv_min"] - (1.0 - m["pearson_r"] ** 2)) < 1e-9
            r_raw = float(np.mean([m["pearson_r"] for m in sel]))
            nrv_raw = float(st.median([m["nrv"] for m in sel]))
            nmin_raw = float(np.mean([m["nrv_min"] for m in sel]))
            te_raw = KT_TRAIN_EV / float(st.median([m["slope"] for m in sel]))
            assert abs(r_raw - rec["r"][i]) < 1e-4, (seed, r_raw, rec["r"][i])
            assert abs(nrv_raw - rec["nrv"][i]) < 1e-4, (seed, nrv_raw, rec["nrv"][i])
            assert abs(nmin_raw - rec["nrv_min"][i]) < 1e-4, \
                (seed, nmin_raw, rec["nrv_min"][i])
            assert abs(te_raw - rec["t_eff"][i]) < 1e-3, (seed, te_raw, rec["t_eff"][i])
            checked[seed] = {"pearson_r_mean_over_parents": r_raw,
                             "nrv_median_over_parents": nrv_raw,
                             "nrv_min_mean_over_parents": nmin_raw,
                             "T_eff_eV_from_median_slope": te_raw}
    return checked


def load_teacher() -> dict:
    """Teacher-evaluator agreement on the same 93 parents, same conventions.

    r is the verified number (0.9271).  NRV, NRV_min and kT_eff are DERIVED HERE
    from ceiling_per_group.csv and are labelled as derived wherever they appear.
    This is a REFERENCE -- the agreement one evaluator reaches with another on
    this population -- not a ceiling a generative model could be expected to hit.
    """
    ids = {str(g) for g in json.loads(F_PRIMARY.read_text())["disjoint_group_ids"]}
    with open(F_CEIL, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if str(r["group_id"]) in ids]
    assert len(rows) == 93, len(rows)
    assert all(2 <= int(r["n_pert"]) <= 8 for r in rows), \
        sorted({int(r["n_pert"]) for r in rows})
    rr = [float(x["pearson_r"]) for x in rows]
    r = float(np.mean(rr))
    nrv = float(st.median([float(x["nrv"]) for x in rows]))
    nmin = float(np.mean([1.0 - v ** 2 for v in rr]))
    te = KT_TRAIN_EV / float(st.median([float(x["slope"]) for x in rows]))
    assert abs(r - 0.9271) < 5e-4, r      # the one verified teacher number
    return {"pearson_r": r, "nrv": nrv, "nrv_min": nmin, "T_eff_eV": te,
            "n_parents": len(rows), "derived": ["nrv", "nrv_min", "T_eff_eV"]}


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------
def panel_a(ax, arms, teacher):
    """Ordering against raw calibration.

    FIGURE-EDITOR PASS.  The panel is 1.78 x 1.48 in instead of 1.38 x 1.15, and
    the annotations are cut to what a reader has to read off the plane: the two
    anchor lines (calibrated floor, constant log-density), the ideal corner, the
    direction of travel, and the teacher point with its two numbers.  The
    parenthetical explaining what the teacher point is and is not now lives in
    the caption, which is where a reader looks for it and where it can be read
    at 9 pt.
    """
    ax.set_title("(a) ordering against raw calibration", loc="left", pad=5,
                 fontsize=7.6)

    xlo, xhi = 0.02, 1.10
    ylo, yhi = -0.34, 3.42
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)

    # constant log-density scores exactly 1
    ax.axhline(1.0, color=PALETTE["greyd"], lw=0.8, ls=(0, (1.6, 1.6)), zorder=1)
    ax.text(1.09, 1.045, "a constant $\\log q_\\theta$ scores 1",
            fontsize=7.0, color=PALETTE["greyd"], ha="right", va="bottom")

    # the calibrated floor
    ax.axhline(0.0, color=PALETTE["ink"], lw=0.7, zorder=1)
    ax.text(0.045, 0.07, "calibrated", fontsize=7.0, color=PALETTE["ink"],
            ha="left", va="bottom")

    # ideal Boltzmann point (definitional, not measured)
    ax.plot([1.0], [0.0], marker="*", ms=9.0, mfc="white",
            mec=PALETTE["ink"], mew=0.9, ls="none", zorder=5, clip_on=False)
    ax.text(1.0, -0.15, "ideal", fontsize=7.0, color=PALETTE["ink"],
            ha="center", va="top")

    # --- arms ---------------------------------------------------------------
    for key, rec in arms.items():
        col = rec["colour"]
        # zorder 7 puts the RAW SEED POINTS above the arm-mean marker (zorder 6).
        # With five control seeds one of them, (r=0.192, NRV=1.388), lands within
        # a marker radius of the control mean (0.200, 1.402) and was completely
        # hidden underneath it when the seeds were drawn first.  Every seed the
        # arm contributes has to be visible, so the raw points draw last.
        ax.plot(rec["r"], rec["nrv"], ls="none", marker=rec["marker"], ms=4.4,
                mfc="white", mec=col, mew=0.9, alpha=0.95, zorder=7)
        mx, my = float(np.mean(rec["r"])), float(np.mean(rec["nrv"]))
        ax.errorbar([mx], [my], xerr=[sem(rec["r"])], yerr=[sem(rec["nrv"])],
                    fmt=rec["marker"], ms=6.2, mfc=col, mec=col, mew=0.9,
                    color=col, elinewidth=1.0, capsize=2.2, zorder=6)

    fm, val = arms["a1_fm_only"], arms["a3_energy"]
    fx, fy = float(np.mean(fm["r"])), float(np.mean(fm["nrv"]))
    vx, vy = float(np.mean(val["r"])), float(np.mean(val["nrv"]))

    ax.annotate("", xy=(vx - 0.012, vy - 0.12), xytext=(fx + 0.012, fy + 0.12),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=PALETTE["ink"],
                                shrinkA=2.0, shrinkB=2.0,
                                connectionstyle="arc3,rad=-0.25"), zorder=3)
    # the reading, parked in the empty upper-right quadrant.  At full size there
    # is no room for it beside the arrow without landing on the value arm's two
    # high-NRV seeds, and a leader line to the arrow only reads as a pointer to
    # the "Value (ours)" label it has to cross.
    # The reading, parked in the empty upper-right quadrant.  It states the
    # direction of travel and where the remaining residual is addressed; it does
    # not read the panel as a failure, because panels (b) and (c) show the
    # rescaled residual and the effective temperature both improving.
    ax.text(1.09, 2.72, "ordering improves;\nraw calibration is a\n"
            "separate target (b, c)",
            fontsize=7.2, color=PALETTE["ink"], ha="right", va="center",
            linespacing=1.20, zorder=8)

    # the control cluster sits just above the constant-log-density rule, so the
    # label is masked rather than moved: anywhere far enough from the rule is
    # also far enough from the cluster to stop naming it.
    ax.text(fx, fy - 0.20, "FM-only", fontsize=7.4, color=PALETTE["grey"],
            ha="center", va="top", zorder=8,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
    ax.text(vx + 0.075, vy + 0.06, "Value (ours)", fontsize=7.4,
            color=PALETTE["green"], ha="left", va="center")

    # --- teacher-evaluator agreement (reference, not a ceiling) -------------
    ax.plot([teacher["pearson_r"]], [teacher["nrv"]], marker="D", ms=5.2,
            mfc="white", mec=PALETTE["orange"], mew=1.0, ls="none", zorder=6)
    # Two lines, centred well below the constant-log-density rule: a third
    # line pushed the block up onto that rule and onto its label.
    ax.annotate("teacher-evaluator agreement\n$r = 0.93$,  NRV $= 0.08$",
                xy=(teacher["pearson_r"], teacher["nrv"] + 0.06),
                xytext=(1.09, 0.52),
                fontsize=7.0, color=PALETTE["orange"], ha="right", va="center",
                linespacing=1.20,
                arrowprops=dict(arrowstyle="-", lw=0.6,
                                color=PALETTE["orange"],
                                shrinkA=1.0, shrinkB=2.0))

    ax.set_xlabel("ordering $r$ (per-parent mean)", labelpad=2.0)
    ax.set_ylabel("raw NRV\n(median over parents)", labelpad=2.5,
                  linespacing=1.15)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 1, 2, 3])


def panel_b(ax, arms, teacher):
    ax.set_title("(b) after optimal rescaling", loc="left", pad=5,
                 fontsize=7.6)

    x_raw, x_min = 0.0, 1.0
    off = {"a1_fm_only": -0.085, "a3_energy": 0.085}

    ax.axhline(1.0, color=PALETTE["greyd"], lw=0.8, ls=(0, (1.6, 1.6)), zorder=1)
    ax.axhline(0.0, color=PALETTE["ink"], lw=0.7, zorder=1)

    for key, rec in arms.items():
        col, dx = rec["colour"], off[key]
        for a, b in zip(rec["nrv"], rec["nrv_min"]):
            ax.plot([x_raw + dx, x_min + dx], [a, b], color=col, lw=0.6,
                    alpha=0.55, zorder=3)
        ax.plot([x_raw + dx] * len(rec["nrv"]), rec["nrv"], ls="none",
                marker=rec["marker"], ms=4.0, mfc="white", mec=col, mew=0.9,
                zorder=4)
        ax.plot([x_min + dx] * len(rec["nrv_min"]), rec["nrv_min"], ls="none",
                marker=rec["marker"], ms=4.0, mfc="white", mec=col, mew=0.9,
                zorder=4)
        for x, vals in ((x_raw + dx, rec["nrv"]), (x_min + dx, rec["nrv_min"])):
            m = float(np.mean(vals))
            ax.plot([x - 0.17, x + 0.17], [m, m], color=col, lw=1.7, zorder=5,
                    solid_capstyle="butt")
            ax.errorbar([x], [m], yerr=[sem(vals)], fmt="none", ecolor=col,
                        elinewidth=1.0, capsize=2.2, zorder=5)

    fm, val = arms["a1_fm_only"], arms["a3_energy"]
    # every printed value is computed here, never transcribed: the arm means are
    # whatever scale_primary.json holds for the seeds listed in ARMS.  The four
    # labels are placed inside the axes at full size, so none of them sits on
    # the y tick column any more.
    fm_nrv, val_nrv = float(np.mean(fm["nrv"])), float(np.mean(val["nrv"]))
    fm_min, val_min = float(np.mean(fm["nrv_min"])), float(np.mean(val["nrv_min"]))
    ax.text(-0.205, fm_nrv, f"{fm_nrv:.2f}",
            fontsize=7.4, color=PALETTE["grey"], ha="right", va="center")
    ax.text(0.205, val_nrv, f"{val_nrv:.2f}",
            fontsize=7.4, color=PALETTE["green"], ha="left", va="center",
            fontweight="bold")
    ax.text(1.245, 0.87, f"{fm_min:.3f}",
            fontsize=7.4, color=PALETTE["grey"], ha="left", va="center")
    ax.text(1.245, 0.42, f"{val_min:.3f}",
            fontsize=7.4, color=PALETTE["green"], ha="left", va="center",
            fontweight="bold")

    # the reference, same two statistics, same conventions
    ax.plot([x_raw - 0.085, x_min + 0.085], [teacher["nrv"], teacher["nrv_min"]],
            color=PALETTE["orange"], lw=0.6, alpha=0.7, zorder=3)
    ax.plot([x_raw - 0.085, x_min + 0.085], [teacher["nrv"], teacher["nrv_min"]],
            ls="none", marker="D", ms=4.0, mfc="white", mec=PALETTE["orange"],
            mew=0.95, zorder=5)
    # Two lines: on one line the label ran under the green 0.747 at x = 1.245.
    ax.text(0.5, 0.22, "teacher-evaluator\nagreement", fontsize=7.0,
            color=PALETTE["orange"], ha="center", va="bottom",
            linespacing=1.20)

    ax.set_xlim(-0.62, 1.72)
    ax.set_ylim(-0.18, 3.42)
    ax.set_xticks([x_raw, x_min])
    ax.set_xticklabels(["raw\nNRV", "best\nrescaling"])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylabel("normalised residual variance", labelpad=2.5)
    ax.tick_params(axis="x", length=0)


def panel_c(ax, arms, teacher):
    # UNITS.  The quantity is kT_eff in eV, an energy, not a temperature: it is
    # kT divided by the fitted slope, and the training target is the numerical
    # kT = 1.0 eV.  Every label in this panel says kT_eff.
    ax.set_title(r"(c) implied $kT_{\mathrm{eff}}$", loc="left", pad=5,
                 fontsize=7.6)

    xs = {"a1_fm_only": 0.0, "a3_energy": 1.0}
    for key, rec in arms.items():
        col, x0 = rec["colour"], xs[key]
        te = np.asarray(rec["t_eff"], float)
        jit = np.linspace(-0.19, 0.19, te.size)
        ax.plot(x0 + jit, te, ls="none", marker=rec["marker"], ms=4.0,
                mfc="white", mec=col, mew=0.9, zorder=4)
        m = float(np.mean(te))
        ax.plot([x0 - 0.30, x0 + 0.30], [m, m], color=col, lw=1.7, zorder=5,
                solid_capstyle="butt")
        ax.errorbar([x0], [m], yerr=[sem(te)], fmt="none", ecolor=col,
                    elinewidth=1.0, capsize=2.2, zorder=5)
        ax.text(x0, 14.2, f"{m:.2f}", fontsize=7.6, color=col, ha="center",
                va="bottom", fontweight="bold")

    ax.axhline(KT_TRAIN_EV, color=PALETTE["ink"], lw=0.9, zorder=2)
    ax.text(-0.44, KT_TRAIN_EV * 0.93, r"target $kT = 1.0$ eV",
            fontsize=7.0, color=PALETTE["ink"], ha="left", va="top")

    ax.set_yscale("log")
    ax.set_ylim(0.72, 21.0)
    ax.set_yticks([1, 2, 5, 10])
    ax.set_yticklabels(["1", "2", "5", "10"])
    ax.minorticks_off()
    ax.set_xlim(-0.48, 1.48)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["FM-only", "Value\n(ours)"])
    ax.set_ylabel(r"$kT_{\mathrm{eff}}$ (eV, log scale)", labelpad=2.5)
    ax.tick_params(axis="x", length=0)


# --------------------------------------------------------------------------
def main():
    set_style()
    arms = load_arms()
    checked = crosscheck_from_raw(arms)
    teacher = load_teacher()

    fig = plt.figure(figsize=(FULL_W, FULL_H))
    # a more even split than the old [1.62, 1.06, 0.72]: panel (c) was 0.61 in
    # wide once the fractional placement was taken into account and is now 0.98.
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.00, 0.80], wspace=0.40,
                          left=0.078, right=0.995, top=0.900, bottom=0.195)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    panel_a(ax_a, arms, teacher)
    panel_b(ax_b, arms, teacher)
    panel_c(ax_c, arms, teacher)

    OUT.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        p = OUT / f"fig4_calibration.{fmt}"
        fig.savefig(p, format=fmt)
        print(f"[fig4] wrote {p}")
    plt.close(fig)

    # ---- provenance dump ---------------------------------------------------
    rec = {
        "figure": ("Figure 4 -- value supervision improves ordering before "
                   "calibration (revision C6)"),
        "revision_c6": ("The continuous 1 - r^2 curve was REMOVED from panel (a): "
                        "the identity is per parent, NRV_min(m) = 1 - r_m^2, and "
                        "mean_m[1 - r_m^2] != 1 - (mean_m r_m)^2, so the curve was "
                        "not a valid boundary for a point built from a mean r and a "
                        "median NRV. The optimally rescaled bound is now reported as "
                        "its own per-seed aggregate in panel (b)."),
        "population": ("93 held-out parents disjoint from the value-term shard pool, "
                       "unperturbed reference geometry dropped, 8 displaced "
                       "geometries per parent, GFN2-xTB evaluator"),
        "conventions": {
            "pearson_r": "MEAN over the 93 parents, then per seed",
            "nrv": "MEDIAN over the 93 parents, then per seed; arm value is the MEAN over seeds",
            "nrv_min": "MEAN over the 93 parents of the per-parent 1 - r_m^2; arm value is the MEAN over seeds",
            "T_eff_eV": "kT / (MEDIAN over parents of the per-parent slope), then per seed",
            "nrv_reference_points": {"ideal_boltzmann": 0.0, "constant_log_density": 1.0},
        },
        "sources": {
            "all_panels_arms": str(F_SCALE),
            "all_panels_arms_path": '["dropref"][arm]["per_seed"][seed]["primary_93"]',
            "panel_a_fields": ["pearson_r", "nrv"],
            "panel_b_fields": ["nrv", "nrv_min"],
            "panel_c_fields": ["T_eff_eV"],
            "crosscheck_raw": str(F_CALIB),
            "group_ids": str(F_PRIMARY) + ' ["disjoint_group_ids"]',
            "teacher_matching_reference": str(F_CEIL),
        },
        "arms": {},
        "teacher_matching_reference": teacher,
        "teacher_note": ("pearson_r is the verified value on this population "
                         "(0.9271). nrv, nrv_min and T_eff_eV were DERIVED for this "
                         "figure from ceiling_per_group.csv restricted to the same 93 "
                         "ids, using the same aggregation conventions as the model "
                         "arms. This is a reference (evaluator-to-evaluator "
                         "agreement), NOT a ceiling for a generative model."),
        "crosscheck_recomputed_from_per_group": checked,
    }
    for key, a in arms.items():
        rec["arms"][key] = {
            "label": a["label"], "seeds": a["seeds"],
            "pearson_r": a["r"], "pearson_r_mean": float(np.mean(a["r"])),
            "pearson_r_sem": sem(a["r"]),
            "nrv": a["nrv"], "nrv_mean_over_seeds": float(np.mean(a["nrv"])),
            "nrv_sem": sem(a["nrv"]),
            "nrv_min": a["nrv_min"],
            "nrv_min_mean_over_seeds": float(np.mean(a["nrv_min"])),
            "nrv_min_sem": sem(a["nrv_min"]),
            "T_eff_eV": a["t_eff"],
            "T_eff_eV_mean_over_seeds": float(np.mean(a["t_eff"])),
            "T_eff_eV_sem": sem(a["t_eff"]),
        }
    p = OUT / "fig4_calibration_data.json"
    p.write_text(json.dumps(rec, indent=2))
    print(f"[fig4] wrote {p}")

    fm, val = rec["arms"]["a1_fm_only"], rec["arms"]["a3_energy"]
    print(f"[fig4] r       FM {fm['pearson_r_mean']:.3f}  ->  value {val['pearson_r_mean']:.3f}")
    print(f"[fig4] NRV     FM {fm['nrv_mean_over_seeds']:.3f}  ->  value {val['nrv_mean_over_seeds']:.3f}")
    print(f"[fig4] NRV_min FM {fm['nrv_min_mean_over_seeds']:.3f}  ->  value {val['nrv_min_mean_over_seeds']:.3f}")
    print(f"[fig4] T_eff   FM {fm['T_eff_eV_mean_over_seeds']:.2f}  ->  value {val['T_eff_eV_mean_over_seeds']:.2f} eV")
    print(f"[fig4] reference r {teacher['pearson_r']:.4f}  NRV {teacher['nrv']:.4f}  "
          f"NRV_min {teacher['nrv_min']:.4f}  T_eff {teacher['T_eff_eV']:.3f} eV  "
          f"(last three derived here)")


if __name__ == "__main__":
    main()
