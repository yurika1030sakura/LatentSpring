#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
figA_resolution.py -- APPENDIX figure: the strictly paired resolution sweep.

USAGE
    /n/holylabs/woo_lab/Lab/yulili/bgfm/envs/omol25/bin/python \
        /n/home04/yulili/bgfm/paper/figures/figA_resolution.py

OUTPUT
    figures/out/figA_resolution.pdf        <- what LaTeX consumes
    figures/out/figA_resolution.png        <- quick look only
    figures/out/figA_resolution_data.json  <- every number drawn

WHAT THIS IS FOR
    The resolution comparison is strictly paired: the same nine checkpoints --
    five flow-matching (s1..s5) and four value (s1, s2, s3, s5) -- were each
    re-scored at n = 4, n = 12 and n = 48 ODE steps on identical parents,
    identical perturbations and a pinned evaluation seed 12345.  The
    per-checkpoint trajectories are the informative object and no table shows
    them, so they are drawn here.

    The two arms move in OPPOSITE directions under the same refinement, and
    panel (b) reads the same records as the gap, which decays
    0.382 -> 0.215 -> 0.037.  Every headline number in the paper is computed at
    n = 12.  The analysis establishes that the estimator is sensitive to solver
    resolution; it does NOT distinguish model-dependent numerical error from an
    effect learned under the four-step training estimator, and neither the
    figure nor its caption claims that it does.

STATISTICAL REPORTING
    Nothing inferential is drawn.  Panel (b) shows the difference of arm means
    over independently trained seeds with the propagated standard error over
    seeds -- a descriptive SEM, not a confidence interval -- and panel (a) shows
    every checkpoint.  The Welch t values remain in the integrity gate and in
    figA_resolution_data.json, where their job is to catch a pipeline change;
    no t value and no significance label reaches the image.

    THIS FIGURE IS FOR THE APPENDIX.  It is not referenced from the main text
    and its float wrapper (figA_resolution.tex) is not \input anywhere yet.

PROVENANCE
    Per-checkpoint values are the author-supplied authoritative re-scoring of the
    records under
        /n/holylabs/woo_lab/Lab/yulili/bgfm/runs/eval_res/
            {a1_fm_only,a3_energy_only}[_s<N>]__ode{4,12,48}/
    They are transcribed here rather than recomputed, per the standing
    instruction not to re-derive them.  The script's integrity gate recomputes
    every AGGREGATE (arm means, gaps, Welch t) from the transcribed
    per-checkpoint values and refuses to draw unless each reproduces the
    authoritative aggregate.

WHAT IS NOT ASSERTED HERE
    The authoritative note describes the flow-matching arm as "four of five
    checkpoints rise".  The transcribed per-checkpoint values support two exact
    statements and neither of them is four of five: three of five rise
    monotonically across all three step counts (s1, s3, s5), and five of five
    end higher at n = 48 than at n = 4.  The figure therefore annotates only
    what its own plotted numbers support, and the discrepancy is flagged rather
    than drawn.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
RECORDS = Path("/n/holylabs/woo_lab/Lab/yulili/bgfm/runs/eval_res")

PALETTE = {
    "green": "#009E73", "grey": "#7F7F7F", "greyd": "#555555",
    "greyl": "#BFBFBF", "ink": "#222222", "orange": "#E69F00",
}

# the appendix float includes this at width=\linewidth, and the canvas is drawn
# at that width, so every point size below is a page point size.
PLACED_FRACTION = 1.00
TEXTWIDTH_IN = 5.50
FULL_W = PLACED_FRACTION * TEXTWIDTH_IN
FULL_H = 2.42

N_STEPS = [4, 12, 48]
DT = [0.2500, 0.0833, 0.0208]

# --------------------------------------------------------------------------
# authoritative per-checkpoint values (n = 4, 12, 48), transcribed, not derived
# --------------------------------------------------------------------------
FM = {                                  # flow-matching control, five seeds
    "s1": [0.141, 0.199, 0.239],
    "s2": [0.226, 0.221, 0.315],
    "s3": [0.176, 0.191, 0.238],
    "s4": [0.205, 0.183, 0.298],
    "s5": [0.197, 0.224, 0.257],
}
VAL = {                                 # value supervision, four seeds
    "s1": [0.528, 0.328, 0.287],
    "s2": [0.614, 0.433, 0.289],
    "s3": [0.654, 0.485, 0.320],
    "s5": [0.489, 0.428, 0.330],
}

# authoritative aggregates the gate must reproduce from the values above
VERIFIED = [                            # n, fm mean, value mean, gap, welch t
    (4, 0.189, 0.571, 0.382, 9.41),
    (12, 0.204, 0.418, 0.215, 6.36),
    (48, 0.269, 0.306, 0.037, 1.94),
]


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Times",
                       "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "axes.titleweight": "bold",
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 7.2,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "legend.frameon": False,
        "lines.linewidth": 1.1, "lines.markersize": 3.6,
        "figure.dpi": 130, "savefig.dpi": 400,
        # explicit, non-tight bbox: drawn size must equal placed size.
        "savefig.bbox": None, "savefig.pad_inches": 0.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def sem(x) -> float:
    x = np.asarray(x, float)
    return float(np.std(x, ddof=1) / math.sqrt(x.size)) if x.size > 1 else float("nan")


def welch_t(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float((a.mean() - b.mean()) /
                 math.sqrt(np.var(a, ddof=1) / a.size + np.var(b, ddof=1) / b.size))


def aggregates():
    out = []
    for j, n in enumerate(N_STEPS):
        f = [v[j] for v in FM.values()]
        v = [v[j] for v in VAL.values()]
        out.append(dict(n=n, dt=DT[j], fm=f, val=v,
                        fm_mean=float(np.mean(f)), val_mean=float(np.mean(v)),
                        fm_sem=sem(f), val_sem=sem(v),
                        gap=float(np.mean(v) - np.mean(f)),
                        t=welch_t(v, f)))
    return out


def verify(A) -> None:
    """Refuse to draw unless every aggregate reproduces the authoritative one."""
    bad = []
    for a, (n, fm_m, val_m, gap, t) in zip(A, VERIFIED):
        assert a["n"] == n
        if abs(a["fm_mean"] - fm_m) > 2e-3:
            bad.append(f"n={n} flow-matching mean {a['fm_mean']:.4f} vs {fm_m}")
        if abs(a["val_mean"] - val_m) > 2e-3:
            bad.append(f"n={n} value mean {a['val_mean']:.4f} vs {val_m}")
        if abs(a["gap"] - gap) > 2e-3:
            bad.append(f"n={n} gap {a['gap']:+.4f} vs {gap:+.3f}")
        # the authoritative t values are quoted from full-precision per-parent
        # correlations; these are recomputed from three-decimal transcriptions,
        # so the gate allows the rounding those transcriptions carry.
        if abs(a["t"] - t) > 6e-2:
            bad.append(f"n={n} Welch t {a['t']:+.3f} vs {t:+.2f}")
    if len(FM) != 5 or len(VAL) != 4:
        bad.append(f"seed counts {len(FM)}/{len(VAL)}, expected 5/4")
    # the two directional facts the figure annotates, checked before it says them
    n_val_mono = sum(1 for v in VAL.values() if v[0] > v[1] > v[2])
    n_fm_end = sum(1 for v in FM.values() if v[2] > v[0])
    if n_val_mono != 4:
        bad.append(f"{n_val_mono} of 4 value checkpoints fall monotonically")
    if n_fm_end != 5:
        bad.append(f"{n_fm_end} of 5 flow-matching checkpoints end higher")
    if bad:
        raise SystemExit("INTEGRITY GATE FAILED -- nothing drawn:\n  "
                         + "\n  ".join(bad))
    print("  integrity gate (not drawn): arm means, gaps and Welch t at "
          "n = 4, 12, 48 all reproduce the authoritative aggregates; 4/4 value "
          "checkpoints fall monotonically and 5/5 flow-matching checkpoints "
          "end higher.  Only the gaps and the SEMs are rendered.")


def draw(A, fmts=("pdf", "png")):
    set_style()
    fig = plt.figure(figsize=(FULL_W, FULL_H))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.42, 1.00], wspace=0.22,
                          left=0.095, right=0.992, top=0.895, bottom=0.215)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    xs = np.arange(3, dtype=float)

    # ---- panel (a): the nine paired checkpoints --------------------------
    axA.axvspan(0.82, 1.18, color=PALETTE["greyl"], alpha=0.30, lw=0, zorder=0)
    axA.annotate("every headline\nnumber in the paper", xy=(1.0, 0.678),
                 fontsize=7.0, color=PALETTE["greyd"], ha="center",
                 va="bottom", linespacing=1.18, zorder=6)

    for tag, v in FM.items():
        axA.plot(xs, v, color=PALETTE["grey"], lw=0.9, marker="o", ms=3.6,
                 mfc="white", mec=PALETTE["grey"], mew=0.9, alpha=0.95,
                 zorder=3)
    for tag, v in VAL.items():
        axA.plot(xs, v, color=PALETTE["green"], lw=0.9, marker="s", ms=3.6,
                 mfc="white", mec=PALETTE["green"], mew=0.9, alpha=0.95,
                 zorder=4)

    # The two callouts double as the arm key: each is drawn in its arm's colour
    # and leads to that arm's lines, so no separate legend competes for the one
    # region of the panel the nine trajectories leave empty.
    axA.annotate("value supervision, 4 checkpoints:\n"
                 "every one falls at every refinement",
                 xy=(2.04, 0.312), xytext=(2.06, 0.560),
                 fontsize=7.0, color=PALETTE["green"], ha="right", va="bottom",
                 linespacing=1.18, annotation_clip=False,
                 arrowprops=dict(arrowstyle="-", lw=0.5,
                                 color=PALETTE["green"], shrinkA=2.0,
                                 shrinkB=2.0))
    axA.annotate("flow matching, 5 checkpoints:\n"
                 "every one ends higher than it started",
                 xy=(2.04, 0.262), xytext=(2.06, 0.160),
                 fontsize=7.0, color=PALETTE["greyd"], ha="right", va="top",
                 linespacing=1.18, annotation_clip=False,
                 arrowprops=dict(arrowstyle="-", lw=0.5,
                                 color=PALETTE["grey"], shrinkA=2.0,
                                 shrinkB=2.0))

    axA.set_xticks(xs)
    axA.set_xticklabels([f"$n = {n}$\n$\\Delta t = {d:.4f}$"
                         for n, d in zip(N_STEPS, DT)], fontsize=7.2,
                        linespacing=1.25)
    axA.set_xlim(-0.16, 2.16)
    axA.set_ylim(0.030, 0.800)
    axA.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    axA.set_ylabel("density–energy\ncorrelation $r$", fontsize=8.0,
                   labelpad=2.5, linespacing=1.15)
    axA.set_title("(a)  the same nine checkpoints, three step counts",
                  fontsize=7.6, loc="left", pad=4.5)
    axA.tick_params(axis="x", length=0, pad=2.5)

    # ---- panel (b): the gap under refinement, descriptively --------------
    # Each point is the difference of the two arm means over independently
    # trained seeds (five control, four value) and each whisker is the
    # propagated standard error over those seeds.  That whisker is an SEM, not
    # a confidence interval.  The Welch t values and the significance labels
    # that used to sit under these points are gone; nothing here is a test.
    gaps = [a["gap"] for a in A]
    ses = [math.hypot(a["val_sem"], a["fm_sem"]) for a in A]
    col = PALETTE["ink"]
    axB.axhline(0.0, color=PALETTE["ink"], lw=0.7, zorder=1)
    axB.plot(xs, gaps, color=PALETTE["ink"], lw=1.0, zorder=3)
    for i, (g, se) in enumerate(zip(gaps, ses)):
        axB.errorbar([xs[i]], [g], yerr=[se], fmt="none", ecolor=col,
                     elinewidth=1.0, capsize=2.4, capthick=1.0, zorder=4)
        axB.plot([xs[i]], [g], marker="o", ms=5.0, mfc=col, mec=col, mew=1.0,
                 ls="none", zorder=5)
        axB.text(xs[i], g + se + 0.030, f"{g:+.3f}", fontsize=7.6, color=col,
                 ha="center", va="bottom", fontweight="bold")
    # One line only: panel (a)'s two callouts already state the seed counts
    # (5 flow-matching checkpoints, 4 value checkpoints), and a second line here
    # runs off the right-hand edge of this narrow panel.
    axB.text(1.02, -0.098, "whiskers: propagated SEM over seeds",
             fontsize=7.0, color=PALETTE["greyd"], ha="center", va="bottom")

    axB.set_xticks(xs)
    axB.set_xticklabels([f"$n = {n}$" for n in N_STEPS], fontsize=7.2)
    axB.set_xlim(-0.38, 2.42)
    axB.set_ylim(-0.115, 0.505)
    axB.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    axB.set_ylabel("gap in $r$ (value $-$ flow matching)", fontsize=8.0,
                   labelpad=2.5)
    axB.set_title("(b)  the gap under refinement", fontsize=7.6, loc="left",
                  pad=4.5)
    axB.tick_params(axis="x", length=0, pad=2.5)

    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in fmts:
        p = OUT / f"figA_resolution.{fmt}"
        fig.savefig(p, format=fmt)
        written.append(p)
        print(f"  wrote {p}  ({p.stat().st_size/1024:.0f} kB)")
    plt.close(fig)
    return written


def dump_json(A):
    rec = {
        "figure": "figA_resolution (appendix)",
        "population": ("the strictly paired resolution sweep: five flow-matching "
                       "checkpoints (s1..s5) and four value checkpoints "
                       "(s1, s2, s3, s5), each re-scored at n = 4, 12 and 48 ODE "
                       "steps on identical parents, identical perturbations and "
                       "pinned evaluation seed 12345"),
        "records_dir": str(RECORDS),
        "transcribed_not_recomputed": True,
        "per_checkpoint": {"flow_matching": FM, "value": VAL},
        "n_steps": N_STEPS, "delta_t": DT,
        "aggregates": [
            dict(n=a["n"], delta_t=a["dt"],
                 fm_mean=round(a["fm_mean"], 6), fm_sem=round(a["fm_sem"], 6),
                 value_mean=round(a["val_mean"], 6),
                 value_sem=round(a["val_sem"], 6),
                 gap=round(a["gap"], 6), welch_t=round(a["t"], 4),
                 n_seeds_fm=len(FM), n_seeds_value=len(VAL))
            for a in A],
        "directional_facts_checked": {
            "value_checkpoints_falling_monotonically": 4,
            "flow_matching_checkpoints_ending_higher": 5,
            "note": ("the authoritative summary phrase 'four of five flow-matching "
                     "checkpoints rise' is not reproducible from these values: "
                     "three of five rise monotonically and five of five end "
                     "higher at n = 48 than at n = 4, so the figure annotates "
                     "only the latter two"),
        },
    }
    p = OUT / "figA_resolution_data.json"
    p.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"  wrote {p}")


def main():
    print("figA_resolution")
    A = aggregates()
    verify(A)
    for a in A:
        print(f"  n = {a['n']:2d}  dt = {a['dt']:.4f}  "
              f"flow matching {a['fm_mean']:.3f} +/- {a['fm_sem']:.3f}  "
              f"value {a['val_mean']:.3f} +/- {a['val_sem']:.3f}  "
              f"gap {a['gap']:+.3f}  t = {a['t']:+.2f}")
    draw(A)
    dump_json(A)


if __name__ == "__main__":
    main()
