#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recompute_disjoint_noreference.py -- the JOINT primary endpoint.

WHY THIS EXISTS
  The manuscript previously reported two robustness corrections SEPARATELY:
    (a) leaveout_contamination.py     -> restrict to the 93 evaluation parents
                                         outside the energy term's shard pool
    (b) recompute_without_reference.py-> drop the unperturbed reference geometry
                                         from every group (8 displaced only)
  A reader cannot compose two marginal corrections into the joint one.  This
  script applies BOTH at once and is the number the revised Sec. 5 reports as the
  primary endpoint.

  It is exactly leaveout_contamination.py's group->val-index mapping composed
  with recompute_without_reference.py's per-record re-scoring, so any
  discrepancy with either of those scripts is a bug in this one.

RUN (from the repo root on holylabs, in envs/flowmol)
  /n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/bin/python \
      paper/figures/recompute_disjoint_noreference.py
"""

from __future__ import annotations

import collections
import csv
import glob
import itertools
import json
import math
import os
import statistics
import sys

import torch

ROOT = "/n/holylabs/woo_lab/Lab/yulili/bgfm/processed_data/omol25_4m_processed"
RUNS = "/n/holylabs/woo_lab/Lab/yulili/bgfm/runs/eval_ours"
SHARD_PARENTS = 10000
MAP_ARMS = ("wide_a3_energy_only_s1", "wide_a1_fm_only_s2")


# --------------------------------------------------------------------------
# group_id -> validation-split index   (verbatim logic from leaveout_contamination.py)
# --------------------------------------------------------------------------
def sig(P: torch.Tensor):
    P = P - P.mean(0, keepdim=True)
    d = torch.sort(P.pow(2).sum(1).sqrt()).values
    return (P.shape[0], tuple((d * 1e4).round().long().tolist()))


def build_val_table():
    val = torch.load(f"{ROOT}/val_data_processed.pt", map_location="cpu",
                     weights_only=False)
    pos = val["positions"].double()
    nia = val["node_idx_array"].tolist()
    table = {}
    for i, (a, b) in enumerate(nia):
        table.setdefault(sig(pos[a:b]), []).append(i)
    return table


def map_arm(tag, table):
    out = {}
    for r in json.load(open(f"{RUNS}/{tag}/boltz1/boltzmann_samples.json")):
        if r["pert_id"]:
            continue
        hit = table.get(sig(torch.tensor(r["positions"], dtype=torch.float64)))
        if hit and len(hit) == 1:
            out[r["group_id"]] = hit[0]
    return out


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def pearson(x, y):
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx < 1e-12 or sy < 1e-12:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def mean(x):
    return sum(x) / len(x)


def sem(x):
    return statistics.stdev(x) / math.sqrt(len(x)) if len(x) > 1 else float("nan")


def welch(A, B):
    va, vb = statistics.variance(A), statistics.variance(B)
    na, nb = len(A), len(B)
    se = math.sqrt(va / na + vb / nb)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return mean(A) - mean(B), (mean(A) - mean(B)) / se, df


def permutation_p(A, B):
    pool, obs, ge, tot = A + B, abs(mean(A) - mean(B)), 0, 0
    for combo in itertools.combinations(range(len(pool)), len(A)):
        a = [pool[i] for i in combo]
        b = [pool[i] for i in range(len(pool)) if i not in combo]
        tot += 1
        if abs(mean(a) - mean(b)) >= obs - 1e-12:
            ge += 1
    return ge, tot, ge / tot


# --------------------------------------------------------------------------
def arm_cells(path, disjoint_gids):
    """Return the four cells: (all120,all9) (all120,noref) (disj,all9) (disj,noref)
    plus the per-group counts."""
    groups = collections.defaultdict(list)
    for row in csv.DictReader(open(path)):
        groups[int(row["group_id"])].append(
            (int(row["pert_id"]), float(row["log_p_theta"]), float(row["negE_kT"])))
    cells, counts = {}, {}
    for gsel, gname in ((None, "all120"), (disjoint_gids, "disjoint")):
        for keep, kname in ((lambda p: True, "all9"), (lambda p: p != 0, "noref")):
            rs = []
            for gid, recs in groups.items():
                if gsel is not None and gid not in gsel:
                    continue
                sel = [(l, e) for p, l, e in recs if keep(p)]
                r = pearson([a for a, _ in sel], [b for _, b in sel])
                if r is not None:
                    rs.append(r)
            cells[(gname, kname)] = mean(rs)
            counts[(gname, kname)] = len(rs)
    return cells, counts


def main():
    table = build_val_table()
    gid2val = {}
    for tag in MAP_ARMS:
        gid2val.update(map_arm(tag, table))
    inside = {g for g, v in gid2val.items() if v < SHARD_PARENTS}
    disjoint = {g for g in gid2val if g not in inside}
    print(f"eval parents mapped {len(gid2val)}/120 | in shard pool {len(inside)} | "
          f"disjoint {len(disjoint)}")

    res, cnt = {}, {}
    for path in sorted(glob.glob(f"{RUNS}/wide_*/boltz_records.csv")):
        tag = os.path.basename(os.path.dirname(path))
        res[tag], cnt[tag] = arm_cells(path, disjoint)
    print()
    hdr = f"{'arm':42s}" + "".join(f"{a}/{b:>6s}" for a, b in
                                   (("all120", "all9"), ("all120", "noref"),
                                    ("disj", "all9"), ("disj", "noref")))
    print(hdr)
    for tag in sorted(res):
        row = "".join(f"  {res[tag][k]:+.4f}   " for k in
                      (("all120", "all9"), ("all120", "noref"),
                       ("disjoint", "all9"), ("disjoint", "noref")))
        print(f"{tag:42s}{row}")

    print()
    for key, label in ((("all120", "all9"),  "ALL 120, all 9        "),
                       (("all120", "noref"), "ALL 120, no reference "),
                       (("disjoint", "all9"), "DISJOINT 93, all 9    "),
                       (("disjoint", "noref"), "DISJOINT 93, no ref   ")):
        E = [res[k][key] for k in res if "a3_energy" in k]
        C = [res[k][key] for k in res if "a1_fm" in k]
        S = [res[k][key] for k in res if "a6_" in k]
        n_groups = cnt[[k for k in res if "a3_energy" in k][0]][key]
        d, t, df = welch(E, C)
        ge, tot, p = permutation_p(E, C)
        line = (f"{label} (n_groups={n_groups:3d})  energy {mean(E):+.4f}+-{sem(E):.4f}"
                f"  control {mean(C):+.4f}+-{sem(C):.4f}"
                f"  Delta {d:+.4f}  t {t:.2f}  df {df:.2f}  perm {ge}/{tot}={p:.3f}")
        if len(S) >= 2:
            ds, ts, _ = welch(E, S)
            ges, tots, ps = permutation_p(E, S)
            dsc, tsc, _ = welch(S, C)
            gsc, tsc_tot, psc = permutation_p(S, C)
            line += (f"  |  scrambled {mean(S):+.4f}+-{sem(S):.4f}"
                     f"  E-S Delta {ds:+.4f} t {ts:.2f} perm {ges}/{tots}={ps:.3f}"
                     f"  |  S-C Delta {dsc:+.4f} t {tsc:.2f} perm p={psc:.3f}")
        print(line)

    # ---- machine-readable artefact consumed by make_experiment_figures.py ----
    out = {"disjoint_group_ids": sorted(disjoint),
           "n_mapped": len(gid2val), "n_in_pool": len(inside),
           "n_disjoint": len(disjoint),
           "cells": {tag: {f"{a}|{b}": res[tag][(a, b)]
                           for (a, b) in res[tag]} for tag in res}}
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "out", "primary_endpoint.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(out, open(dst, "w"), indent=1)
    print(f"\nwrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
