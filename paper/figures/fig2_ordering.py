#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
fig2_ordering.py -- builds the held-out ordering figure (FIGURE 3 in the
rendered PDF; label fig:ordering, Sec. 5.2).

USAGE
    /n/holylabs/woo_lab/Lab/yulili/bgfm/envs/omol25/bin/python \
        /n/home04/yulili/bgfm/paper/figures/fig2_ordering.py

OUTPUT
    figures/out/fig2_ordering.pdf       <- what LaTeX consumes
    figures/out/fig2_ordering.png       <- quick look only
    figures/out/fig2_ordering_data.json <- every number printed or plotted,
                                           with the file it came from

NOTATION.  The measured quantity is the CLAMPED POSITIONAL-FLOW DENSITY
q_theta: the endpoint discrete labels are held fixed and only the positional
ODE is reversed.  It is not the joint generator's conditional density, so every
axis label, legend entry and annotation in this figure reads \log q_\theta.
The stored record files keep their original column name `log_p_theta`; that is
a data-file field name, not a label, and it is never drawn.

POPULATION (panels a-c)
    The conservative held-out endpoint: the 93 parents disjoint from the energy
    term's shard pool, with the unperturbed data geometry dropped, scored on
    the (up to 8) displaced geometries by the independent GFN2-xTB evaluator --
    NOT by the eSEN training teacher.  5 seeds flow-matching-only, 4 seeds
    value supervision.

PANELS AND THEIR SOURCE FILES  (nothing here is scraped from prose or a PDF)
    (a) per-seed correlation, both arms, raw seed points
        figures/out/scale_primary.json
          ["dropref"][arm]["per_seed"][seed]["primary_93"]["pearson_r"]
        cross-checked at run time against the raw per-geometry records
        runs/eval_ours/wide_<tag>/boltz_records.csv; the two routes agree to
        4 decimals and the script asserts it.
    (b) parent-level distribution, empirical CDF over the 93 parents
        runs/eval_ours/wide_<tag>/boltz_records.csv, pert_id 0 dropped,
        restricted to the 93 ids in figures/out/primary_endpoint.json,
        per-parent Pearson averaged over the arm's seeds.
    (c) within-parent retrieval, top-1 and top-3, raw seed points
        figures/out/retrieval_utility.json
          ["arms"][arm]["per_seed"][tag]["primary_93"], ["chance_top1"],
          ["chance_top3"], ["oracle"]["primary_93"].
        The ["oracle"] block is the independent GFN2-xTB evaluator scoring the
        eSEN TRAINING TEACHER itself on this population.  It is renamed on load
        to teacher_ref and drawn as a small ensemble-specific reference marker.
        The figure names it the teacher-evaluator agreement reference.
        It is not a ceiling, and no label in the figure calls it one.
    (d) the fixed-RMSD control against the default Gaussian family
        The default-family pair is recomputed from the same records as (a).
        The fixed-RMSD pair is the paper's verified pair, carried here as the
        constants FIXED_RMSD below; see the provenance note attached to them.

STATISTICAL REPORTING
    Nothing inferential is drawn.  The figure reports arm means over
    independently trained seeds, the standard error over those seeds, every raw
    seed point, the absolute gain Delta r, the seed counts, and the
    launched/completed/diverged record.  The Welch t and the exact permutation
    p are still computed, because they are the guard that the arms have not
    silently changed, and they are written to fig2_ordering_data.json; no t
    value, no p-value and no significance label reaches the image.

WHAT CHANGED IN THIS REVISION (figure-editor pass)
    The standardised geometry-level scatter that used to occupy panel (c) is
    gone.  Its fitted coefficient was algebraically the panel-(a) arm mean, so
    it restated a number two other panels already carried, and 3.5k rasterised
    points bought only that.  The fixed-RMSD control takes the space instead:
    it answers the confound that most threatens the ordering claim -- that the
    value arm merely prefers geometries nearer the data manifold -- which no
    other panel addresses.  The scrambled-energy arm is also dropped from panel
    (a): only one of its two shards was scrambled, so it is defective, and cell
    C of the matched grid supersedes it.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent                 # paper/figures
OUT = HERE / "out"
RUNS = Path(os.environ.get(
    "BGFM_RUNS", "/n/holylabs/woo_lab/Lab/yulili/bgfm/runs/eval_ours"))

F_SCALE = OUT / "scale_primary.json"
F_RETR = OUT / "retrieval_utility.json"
F_PRIM = OUT / "primary_endpoint.json"

ARMS = {
    "fm": dict(
        label="FM-only",
        short="FM",
        scale_key="a1_fm_only",
        retr_key="a1_fm_only",
        # the unsuffixed tag is the fifth control seed, recovered and scored
        # under the identical protocol after the first four (paper Sec. B.9).
        seeds=["a1_fm_only_s2", "a1_fm_only_s3",
               "a1_fm_only_s4", "a1_fm_only_s5", "a1_fm_only"],
        wide=["wide_a1_fm_only_s2", "wide_a1_fm_only_s3",
              "wide_a1_fm_only_s4", "wide_a1_fm_only_s5",
              "wide_a1_fm_only"],
    ),
    "value": dict(
        label="Value (ours)",
        short="Value",
        scale_key="a3_energy",
        retr_key="a3_energy",
        seeds=["a3_energy_only_s1", "a3_energy_only_s2",
               "a3_energy_only_s3", "a3_energy_only_s5"],
        wide=["wide_a3_energy_only_s1", "wide_a3_energy_only_s2",
              "wide_a3_energy_only_s3", "wide_a3_energy_only_s5"],
    ),
}

# ---------------------------------------------------------------------------
# Panel (d): the fixed-RMSD control.
#
# PROVENANCE, stated plainly because it differs from every other panel.  These
# four scalars are the paper's verified fixed-RMSD result (protected-number
# register: REVISION_LOG.md, STORYLINE.md Sec. ROBUSTNESS, EVIDENCE_DIGEST.md
# P2; drawn in Sec. 5.5 and App. B).  They are carried as constants rather than
# recomputed because NO artefact under runs/eval_p3 reproduces this exact
# aggregation: the __cpu round (40 parents, 4 seeds/arm) gives 0.3028 / 0.1885
# and the __gpu60 round (120 parents, 4 seeds/arm) gives 0.3505 / 0.2482.
# Plotting either round's per-seed cloud under the verified means would be a
# cross-round splice, so the panel draws the verified arm means only, states
# the seed count, and draws no seed points on the fixed-RMSD side.  The panel
# and the caption both say the two families are separate evaluation rounds and
# that the comparison is between GAPS, not between levels.
# ---------------------------------------------------------------------------
FIXED_RMSD = dict(fm=0.182, value=0.305, delta=0.124, t=4.94, n_seeds=4,
                  source=("verified protected-number register (fixed-RMSD "
                          "shells, within-group displacement magnitude "
                          "constant to numerical precision, 4 seeds per arm); "
                          "no on-disk artefact reproduces this aggregation"))

# The default-family values the panel is checked against (same register; also
# recomputed from records at run time, and the script asserts they agree).
VERIFIED = dict(fm_mean=0.200, fm_sem=0.018, value_mean=0.369, value_sem=0.023,
                delta=0.169, t=5.82, perm_p=0.008,
                top1_fm=0.202, top1_value=0.296,
                top3_fm=0.495, top3_value=0.624,
                chance_top1=0.125, chance_top3=0.375, n_better=67)

PALETTE = {
    "green": "#009E73",     # value / energy
    "grey": "#7F7F7F",      # no-physics flow-matching baseline
    "greyd": "#555555",
    "greyl": "#BFBFBF",
    "orange": "#E69F00",    # teacher-evaluator agreement reference
}
C = {"fm": PALETTE["grey"], "value": PALETTE["green"]}
MK = {"fm": "o", "value": "s"}

# The float includes this PDF at width=0.76\linewidth and \textwidth is
# 5.5 in, so the figure is PLACED at 4.18 in.  Draw it at that width, not at
# 5.5 in: a 5.5 in canvas scaled to 0.76 renders every in-figure size at 76%
# of what it says here -- 8 pt labels arrive as 6.1 pt and the 6 pt notes as
# 4.6 pt.  Drawn at the placed width, the sizes below are the sizes on the
# page.  If the float ever goes back to width=\linewidth, set this to 5.50.
PLACED_FRACTION = 1.00
TEXTWIDTH_IN = 5.50
FULL_W = PLACED_FRACTION * TEXTWIDTH_IN          # 5.50 in


# ---------------------------------------------------------------------------
# style + tiny stats (same conventions as figures/make_experiment_figures.py)
# ---------------------------------------------------------------------------
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
        # NOT bbox="tight".  A tight bbox grows the canvas past the drawn
        # width, so the float scales it down at \includegraphics and every
        # point size lands smaller than requested.  Same fix as fig3_grid.py.
        "savefig.bbox": None, "savefig.pad_inches": 0.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def sem(x) -> float:
    x = np.asarray(x, float)
    return float(np.std(x, ddof=1) / math.sqrt(x.size)) if x.size > 1 else float("nan")


def welch_t(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = np.var(a, ddof=1) / a.size, np.var(b, ddof=1) / b.size
    return float((a.mean() - b.mean()) / math.sqrt(va + vb))


def permutation_p(A, B):
    """Exact two-sided permutation test over seed relabellings."""
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


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    x = x - x.mean()
    y = y - y.mean()
    d = math.sqrt(float(x @ x) * float(y @ y))
    return float(x @ y) / d if d > 0 else float("nan")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def primary_ids() -> set:
    ids = json.loads(F_PRIM.read_text())["disjoint_group_ids"]
    assert len(ids) == 93, f"expected 93 disjoint parents, got {len(ids)}"
    return set(int(i) for i in ids)


def load_records(tag: str, ids: set):
    """{group_id: (log q array, negE_kT array)} on the conservative endpoint."""
    path = RUNS / tag / "boltz_records.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    g = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            gid, pid = int(row["group_id"]), int(row["pert_id"])
            if pid == 0 or gid not in ids:      # drop data geometry, keep the 93
                continue
            g.setdefault(gid, [[], []])
            # `log_p_theta` is the stored column name; the quantity is log q.
            g[gid][0].append(float(row["log_p_theta"]))
            g[gid][1].append(float(row["negE_kT"]))
    return {k: (np.array(v[0]), np.array(v[1])) for k, v in g.items()}


def per_seed_r_from_scale_json():
    d = json.loads(F_SCALE.read_text())["dropref"]
    out = {}
    for key, spec in ARMS.items():
        blk = d[spec["scale_key"]]["per_seed"]
        out[key] = [blk[s]["primary_93"]["pearson_r"] for s in spec["seeds"]]
    return out


def per_seed_r_from_records(ids):
    """Independent route: per-parent Pearson straight off the raw records."""
    out, per_parent = {}, {}
    for key, spec in ARMS.items():
        seed_means, parent_acc = [], {}
        for tag in spec["wide"]:
            rec = load_records(tag, ids)
            assert len(rec) == 93, f"{tag}: {len(rec)} parents, expected 93"
            rs = []
            for gid, (lq, ne) in rec.items():
                r = pearson(lq, ne)
                rs.append(r)
                parent_acc.setdefault(gid, []).append(r)
            seed_means.append(float(np.mean(rs)))
        out[key] = seed_means
        per_parent[key] = {g: float(np.mean(v)) for g, v in parent_acc.items()}
    return out, per_parent


def load_retrieval():
    d = json.loads(F_RETR.read_text())
    out = {}
    for key, spec in ARMS.items():
        blk = d["arms"][spec["retr_key"]]
        agg = blk["aggregate"]["primary_93"]
        out[key] = dict(
            top1=[v["primary_93"]["top1"] for v in blk["per_seed"].values()],
            top3=[v["primary_93"]["top3"] for v in blk["per_seed"].values()],
            top1_mean=agg["top1"]["mean"], top3_mean=agg["top3"]["mean"],
            top1_sem=agg["top1"]["sem"], top3_sem=agg["top3"]["sem"],
        )
    # ["oracle"] is the key in the stored JSON; the quantity is the independent
    # evaluator scoring the eSEN training teacher, i.e. an ensemble-specific
    # teacher-evaluator agreement reference.  Renamed on load so no label can
    # say "oracle".
    out["teacher_ref"] = dict(top1=d["oracle"]["primary_93"]["top1"],
                              top3=d["oracle"]["primary_93"]["top3"])
    out["chance"] = dict(top1=d["chance_top1"], top3=d["chance_top3"])
    return out


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------
def panel_a(ax, rseed, stats):
    order = ["fm", "value"]
    rng = np.random.default_rng(20260813)     # jitter only; no data is random
    for i, key in enumerate(order):
        v = np.asarray(rseed[key], float)
        m, s = float(v.mean()), sem(v)
        ax.errorbar(i, m, yerr=s, fmt="none", ecolor=C[key],
                    elinewidth=1.1, capsize=2.6, capthick=1.1, zorder=3)
        ax.plot([i - 0.20, i + 0.20], [m, m], color=C[key], lw=1.6, zorder=4,
                solid_capstyle="butt")
        jit = rng.uniform(-0.115, 0.115, v.size)
        ax.plot(i + jit, v, MK[key], mfc="none", mec=C[key], mew=0.9, ms=3.8,
                ls="none", zorder=5)
        ax.text(i, 0.026, f"{m:.3f}", ha="center", va="bottom", fontsize=7.0,
                color=C[key])
        ax.text(i, -0.005, f"{v.size} seeds", ha="center", va="top",
                fontsize=7.0, color=PALETTE["greyd"],
                transform=ax.get_xaxis_transform())

    ytop = max(max(rseed["fm"]), max(rseed["value"]))
    yb = ytop + 0.055
    ax.plot([0, 0, 1, 1], [yb - 0.012, yb, yb, yb - 0.012],
            color="black", lw=0.7, clip_on=False)
    # DESCRIPTIVE ONLY.  The gain and the seed counts are drawn; the Welch t and
    # the exact permutation p that used to sit on this bracket are not.  They
    # remain in fig2_ordering_data.json, which is an audit record, not an image.
    ax.text(0.5, yb + 0.008,
            r"gain $\Delta r = +%.3f$" % stats["delta"],
            ha="center", va="bottom", fontsize=7.0, linespacing=1.25)
    ax.text(0.99, 0.13,
            "all %d value seeds exceed\nall %d control seeds"
            % (len(rseed["value"]), len(rseed["fm"])),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.0,
            linespacing=1.25, color=PALETTE["greyd"])

    ax.set_xticks(range(2))
    ax.set_xticklabels([ARMS[k]["label"] for k in order])
    ax.tick_params(axis="x", length=0, pad=9)
    ax.set_xlim(-0.62, 1.62)
    ax.set_ylim(0.0, 0.60)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_ylabel(r"per-parent $r(\log q_\theta,\ -E/kT)$")
    ax.set_title("(a)  per-seed ordering, 93 parents", loc="left")


def panel_b(ax, per_parent, rseed, n_better):
    for key in ("fm", "value"):
        v = np.sort(np.fromiter(per_parent[key].values(), float))
        y = np.arange(1, v.size + 1) / v.size
        ax.step(np.concatenate([v, v[-1:]]), np.concatenate([y, y[-1:]]),
                where="post", color=C[key], lw=1.3, label=ARMS[key]["label"])
        ax.axvline(float(np.mean(rseed[key])), color=C[key], lw=0.7,
                   ls=(0, (1.4, 1.4)), alpha=0.85)
    ax.axvline(0.0, color=PALETTE["greyl"], lw=0.6, zorder=0)
    ax.set_xlim(-0.6, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"per-parent $r$, averaged over seeds")
    ax.set_ylabel("fraction of parents $\\leq r$")
    ax.legend(loc="upper left", handlelength=1.5, borderpad=0.1,
              labelspacing=0.25, fontsize=7.0)
    ax.text(0.035, 0.70, "dotted: arm mean $r$", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.0, color=PALETTE["greyd"])
    ax.text(0.985, 0.06, f"value higher on\n{n_better} of 93 parents",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.0,
            linespacing=1.25, color=PALETTE["greyd"])
    ax.set_title("(b)  distribution over parents", loc="left")


def panel_c(ax, retr):
    """Within-parent retrieval, top-1 and top-3, with chance and the
    ensemble-specific teacher-evaluator agreement reference."""
    groups = [("top1", "top-1"), ("top3", "top-3")]
    keys = ["fm", "value"]
    rng = np.random.default_rng(20260813)
    tickx, ticklab = [], []
    for gi, (gk, glab) in enumerate(groups):
        base = gi * 1.75
        for ki, key in enumerate(keys):
            x = base + ki * 0.55
            m = retr[key][f"{gk}_mean"]
            ax.bar(x, m, width=0.44, color=C[key], alpha=0.28,
                   edgecolor=C[key], lw=0.9, zorder=2)
            ax.errorbar(x, m, yerr=retr[key][f"{gk}_sem"], fmt="none",
                        ecolor=C[key], elinewidth=1.0, capsize=2.4,
                        capthick=1.0, zorder=4)
            v = np.asarray(retr[key][gk], float)
            ax.plot(x + rng.uniform(-0.11, 0.11, v.size), v, MK[key],
                    mfc="none", mec=C[key], mew=0.9, ms=3.4, ls="none",
                    zorder=5)
            # clear the raw seed cloud, not just the error bar: at 7 pt the
            # percentage label used to land on the topmost seed marker.
            ytop = max(m + retr[key][f"{gk}_sem"], float(np.max(v)))
            ax.text(x, ytop + 0.030, f"{100*m:.1f}%",
                    ha="center", va="bottom", fontsize=7.0, color=C[key])
            tickx.append(x)
            ticklab.append(ARMS[key]["short"])
        lo, hi = base - 0.33, base + 0.88
        ax.plot([lo, hi], [retr["chance"][gk]] * 2, color=PALETTE["greyd"],
                lw=0.9, ls=(0, (3, 1.6)), zorder=6)
        ax.text(hi + 0.04, retr["chance"][gk], f"{100*retr['chance'][gk]:.1f}%",
                ha="left", va="center", fontsize=7.0, color=PALETTE["greyd"])
        # the reference is a small marker, deliberately not a spanning rule
        ax.plot([base + 0.275], [retr["teacher_ref"][gk]], marker="D",
                ms=3.4, mfc=PALETTE["orange"], mec=PALETTE["orange"],
                ls="none", zorder=7)
        ax.text(base + 0.275, -0.155, glab, ha="center", va="top",
                fontsize=7.5, transform=ax.get_xaxis_transform())
    ax.set_xticks(tickx)
    ax.set_xticklabels(ticklab, fontsize=7.0)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(-0.55, 3.35)
    ax.set_ylim(0, 1.24)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_ylabel("lowest-energy geometry\nretrieved (%)", linespacing=1.2)
    ax.set_title("(c)  retrieval within a parent", loc="left")
    ax.legend(handles=[
        Line2D([], [], color=PALETTE["orange"], marker="D", ms=3.4, ls="none",
               label="teacher-evaluator agreement reference\n(ensemble-specific, not a ceiling)"),
        Line2D([], [], color=PALETTE["greyd"], lw=0.9, ls=(0, (3, 1.6)),
               label="chance")],
        loc="upper left", handlelength=1.4, borderpad=0.1, labelspacing=0.3,
        fontsize=7.0, bbox_to_anchor=(-0.02, 1.045))


def panel_d(ax, rseed, fixed):
    """Default Gaussian displacements against fixed-RMSD shells.

    Two separate evaluation rounds on different parent sets, so the comparison
    the panel invites is between the two GAPS, not between the two levels, and
    the axis note says so.  The Gaussian family carries its raw seed points;
    the fixed-RMSD family carries arm means only, because no on-disk artefact
    reproduces the verified pair per seed (see FIXED_RMSD above).
    """
    rng = np.random.default_rng(20260819)
    fams = [
        dict(key="gauss", x=0.0,
             means={k: float(np.mean(rseed[k])) for k in ("fm", "value")},
             delta=float(np.mean(rseed["value"]) - np.mean(rseed["fm"])),
             t=VERIFIED["t"], seeds=rseed,
             lab="Gaussian displacements\n(%d and %d seeds)"
                 % (len(rseed["fm"]), len(rseed["value"]))),
        dict(key="rmsd", x=1.70,
             means={"fm": fixed["fm"], "value": fixed["value"]},
             delta=fixed["delta"], t=fixed["t"], seeds=None,
             lab="fixed-RMSD shells\n(%d seeds per arm)" % fixed["n_seeds"]),
    ]
    tickx, ticklab = [], []
    for fam in fams:
        for ki, key in enumerate(("fm", "value")):
            x = fam["x"] + ki * 0.55
            m = fam["means"][key]
            ax.plot([x - 0.19, x + 0.19], [m, m], color=C[key], lw=1.7,
                    zorder=4, solid_capstyle="butt")
            ax.plot([x], [m], MK[key], ms=4.4, mfc=C[key], mec=C[key],
                    ls="none", zorder=5)
            if fam["seeds"] is not None:
                v = np.asarray(fam["seeds"][key], float)
                ax.plot(x + rng.uniform(-0.115, 0.115, v.size), v, MK[key],
                        mfc="none", mec=C[key], mew=0.9, ms=3.4, ls="none",
                        zorder=6)
            ax.text(x, 0.022, f"{m:.3f}", ha="center", va="bottom",
                    fontsize=7.0, color=C[key])
            tickx.append(x)
            ticklab.append(ARMS[key]["short"])
        # gap bracket, in the style of panel (a)
        yb = 0.455
        x0, x1 = fam["x"], fam["x"] + 0.55
        ax.plot([x0, x0, x1, x1], [yb - 0.012, yb, yb, yb - 0.012],
                color="black", lw=0.7)
        # the gap only.  No test statistic is drawn on either family.
        ax.text(0.5 * (x0 + x1), yb + 0.008,
                r"$+%.3f$" % fam["delta"],
                ha="center", va="bottom", fontsize=7.0, linespacing=1.25)
        ax.text(0.5 * (x0 + x1), -0.075, fam["lab"], ha="center", va="top",
                fontsize=7.0, linespacing=1.3,
                transform=ax.get_xaxis_transform())

    ax.text(0.5, 0.115, "compare the gaps, not the levels",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=7.0,
            color=PALETTE["greyd"])
    ax.set_xticks(tickx)
    ax.set_xticklabels(ticklab, fontsize=7.0)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(-0.45, 2.70)
    ax.set_ylim(0.0, 0.56)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_ylabel(r"per-parent $r(\log q_\theta,\ -E/kT)$")
    ax.set_title("(d)  the gap survives fixed RMSD", loc="left")


# ---------------------------------------------------------------------------
def main():
    set_style()
    ids = primary_ids()

    r_json = per_seed_r_from_scale_json()
    r_rec, per_parent = per_seed_r_from_records(ids)

    # --- cross-check: cached aggregate vs raw per-geometry records ----------
    for key in ARMS:
        for a, b in zip(r_json[key], r_rec[key]):
            assert abs(a - b) < 1e-4, (key, a, b)

    n_better = sum(1 for g in per_parent["fm"]
                   if per_parent["value"][g] > per_parent["fm"][g])

    t = welch_t(r_rec["value"], r_rec["fm"])
    _, _, pp = permutation_p(r_rec["value"], r_rec["fm"])
    stats = dict(delta=float(np.mean(r_rec["value"]) - np.mean(r_rec["fm"])),
                 t=t, perm_p=pp)
    retr = load_retrieval()

    # --- guard: everything drawn must reproduce the verified register ------
    assert abs(np.mean(r_rec["fm"]) - VERIFIED["fm_mean"]) < 5e-4
    assert abs(np.mean(r_rec["value"]) - VERIFIED["value_mean"]) < 5e-4
    assert abs(sem(r_rec["fm"]) - VERIFIED["fm_sem"]) < 5e-4
    assert abs(sem(r_rec["value"]) - VERIFIED["value_sem"]) < 5e-4
    assert abs(stats["delta"] - VERIFIED["delta"]) < 5e-4
    assert abs(stats["t"] - VERIFIED["t"]) < 5e-3
    assert abs(stats["perm_p"] - VERIFIED["perm_p"]) < 5e-4
    assert n_better == VERIFIED["n_better"]
    assert min(r_rec["value"]) > max(r_rec["fm"]), "seed separation claim"
    assert abs(retr["fm"]["top1_mean"] - VERIFIED["top1_fm"]) < 5e-4
    assert abs(retr["value"]["top1_mean"] - VERIFIED["top1_value"]) < 5e-4
    assert abs(retr["fm"]["top3_mean"] - VERIFIED["top3_fm"]) < 5e-4
    assert abs(retr["value"]["top3_mean"] - VERIFIED["top3_value"]) < 5e-4
    assert retr["chance"]["top1"] == VERIFIED["chance_top1"]
    assert retr["chance"]["top3"] == VERIFIED["chance_top3"]
    assert abs((FIXED_RMSD["value"] - FIXED_RMSD["fm"])
               - FIXED_RMSD["delta"]) < 1.5e-3
    assert len(r_rec["fm"]) == 5 and len(r_rec["value"]) == 4

    fig = plt.figure(figsize=(FULL_W, 4.00))
    gs = fig.add_gridspec(2, 2, hspace=0.70, wspace=0.36,
                          left=0.10, right=0.975, top=0.925, bottom=0.105)
    panel_a(fig.add_subplot(gs[0, 0]), r_rec, stats)
    panel_b(fig.add_subplot(gs[0, 1]), per_parent, r_rec, n_better)
    panel_c(fig.add_subplot(gs[1, 0]), retr)
    panel_d(fig.add_subplot(gs[1, 1]), r_rec, FIXED_RMSD)

    OUT.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        p = OUT / f"fig2_ordering.{fmt}"
        fig.savefig(p, format=fmt)
        print(f"  wrote {p}  ({p.stat().st_size/1024:.0f} kB)")
    plt.close(fig)

    # --- machine-readable record of everything drawn -----------------------
    rec = {
        "notation": ("Every drawn label reads log q_theta, the clamped "
                     "positional-flow density (endpoint discrete labels held "
                     "fixed, positional ODE reversed). The record files keep "
                     "the column name log_p_theta; that name is never drawn."),
        "population_abc": ("93 held-out parents disjoint from the energy "
                           "term's shard pool; data geometry dropped; scored "
                           "by GFN2-xTB, independent of the eSEN teacher"),
        "sources": {
            "panel_a": str(F_SCALE) + "  (cross-checked against "
                       + str(RUNS) + "/wide_*/boltz_records.csv)",
            "panel_b": str(RUNS) + "/wide_*/boltz_records.csv",
            "panel_c": str(F_RETR),
            "panel_d_gaussian": str(RUNS) + "/wide_*/boltz_records.csv",
            "panel_d_fixed_rmsd": FIXED_RMSD["source"],
            "ids": str(F_PRIM),
        },
        "per_seed_r_cached": r_json,
        "per_seed_r_recomputed": r_rec,
        "arm_mean_r": {k: float(np.mean(v)) for k, v in r_rec.items()},
        "arm_sem_r": {k: sem(v) for k, v in r_rec.items()},
        "n_parents_value_gt_fm": n_better,
        "median_per_parent_r": {k: float(np.median(list(v.values())))
                                for k, v in per_parent.items()},
        "primary_contrast": stats,
        "retrieval": {k: v for k, v in retr.items()},
        "retrieval_reference_note": (
            "'teacher_ref' is the ['oracle'] block of retrieval_utility.json: "
            "the independent GFN2-xTB evaluator scoring the eSEN training "
            "teacher itself on the same 93 parents. It is drawn as a small "
            "ensemble-specific reference marker and labelled as such; the "
            "words 'oracle', 'ceiling' and 'bound' describing it appear "
            "nowhere in the figure."),
        "fixed_rmsd_panel": dict(FIXED_RMSD),
        "fixed_rmsd_note": (
            "Panel (d) draws the verified fixed-RMSD arm means only. No "
            "artefact under runs/eval_p3 reproduces this aggregation: the "
            "__cpu round (40 parents, 4 seeds/arm) gives 0.3028/0.1885 and "
            "the __gpu60 round (120 parents, 4 seeds/arm) 0.3505/0.2482, so "
            "no per-seed cloud is drawn on that side and the panel states "
            "that the two families are separate rounds on different parent "
            "sets."),
        "panels_removed_this_revision": (
            "The standardised geometry-level scatter is gone: its fitted "
            "coefficient was algebraically the panel-(a) arm mean, so it "
            "restated a number panels (a) and (b) already carried. The "
            "scrambled-energy arm is gone from panel (a): only one of its two "
            "shards was scrambled, and cell C of the matched grid supersedes "
            "it."),
    }
    p = OUT / "fig2_ordering_data.json"
    p.write_text(json.dumps(rec, indent=1))
    print(f"  wrote {p}")

    print("\n  panel (a) arm means: "
          + ", ".join(f"{ARMS[k]['label']} {np.mean(r_rec[k]):.3f} "
                      f"+/- {sem(r_rec[k]):.3f}" for k in ARMS))
    print(f"  primary contrast: delta {stats['delta']:+.3f}, "
          f"t {stats['t']:.2f}, exact permutation p {stats['perm_p']:.3f}")
    print(f"  panel (b): value exceeds FM on {n_better} of 93 parents")
    print(f"  panel (c): top-1 {retr['fm']['top1_mean']:.3f} -> "
          f"{retr['value']['top1_mean']:.3f};  top-3 "
          f"{retr['fm']['top3_mean']:.3f} -> {retr['value']['top3_mean']:.3f}")
    print(f"  panel (d): fixed-RMSD {FIXED_RMSD['fm']:.3f} -> "
          f"{FIXED_RMSD['value']:.3f}, +{FIXED_RMSD['delta']:.3f}, "
          f"t {FIXED_RMSD['t']:.2f}, {FIXED_RMSD['n_seeds']} seeds per arm")


if __name__ == "__main__":
    main()
