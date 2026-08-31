#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recompute_without_reference.py -- recompute every per-group Boltzmann r with the
unperturbed reference geometry excluded.

WHY THIS EXISTS
  scripts/eval_boltzmann_stage1.py builds each evaluation group as
      p = 0  -> the UNPERTURBED reference geometry (base_pos.clone())
      p > 0  -> n_perturb = 8 copies displaced by N(0, 0.15^2)
  so K = 9 is "reference + 8 displacements", not "9 perturbations".  The
  reference is systematically the lowest-energy member of its group and is a
  high-leverage point in both log q_theta and -E/kT, which means a density that merely
  separates clean geometries from noised ones can move r_m without carrying any
  Boltzmann information.  Dropping it is the cleanest robustness check available
  for the primary metric.

  scripts/eval_boltzmann_stage2.py aggregates to per-group r and discards the
  individual (log q_theta, E) pairs, so this check requires re-scoring.  Run
  paper/figures/dump_all_wide.sh first; it drives
  paper/figures/dump_boltz_records.py over every n = 120 arm and writes
  <tag>/boltz_records.csv.

WHAT IT FOUND (2026-08-03; recorded in Sec. 5.1, Table 1 and Sec. 5.7)
  arm                     all 9      excl. reference
  no physics (4 seeds)   +0.074         +0.218
  energy     (4 seeds)   +0.434         +0.397
  Delta                  +0.360         +0.178   (Welch t 11.3 -> 6.2)
  permutation p           0.029          0.029   (all 4 energy seeds still above
                                                  all 4 control seeds)
  The contrast halves and survives.  The asymmetry is interpretable: on the full
  group the control is PENALISED by the reference point, whose log-density it
  gets wrong relative to its energy, while the energy-supervised model gets it
  right.  Note the all-9 means here differ from the pipeline's by <= 0.004
  because this re-scoring loses 6 of 1080 xTB single points per arm rather than
  the pipeline's 1-2.

RUN
  /n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/bin/python \
      paper/figures/recompute_without_reference.py
"""

from __future__ import annotations

import collections
import csv
import glob
import itertools
import math
import os
import statistics
import sys

RUNS = "runs/eval_ours"


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


def arm_means(path):
    groups = collections.defaultdict(list)
    for row in csv.DictReader(open(path)):
        groups[int(row["group_id"])].append(
            (int(row["pert_id"]), float(row["log_p_theta"]), float(row["negE_kT"])))
    out = []
    for keep in (lambda p: True, lambda p: p != 0):
        rs = []
        for recs in groups.values():
            sel = [(l, e) for p, l, e in recs if keep(p)]
            r = pearson([a for a, _ in sel], [b for _, b in sel])
            if r is not None:
                rs.append(r)
        out.append(sum(rs) / len(rs))
    return out


def mean(x):
    return sum(x) / len(x)


def sem(x):
    return statistics.stdev(x) / math.sqrt(len(x))


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


def main():
    res = {}
    files = sorted(glob.glob(f"{RUNS}/wide_*/boltz_records.csv"))
    if not files:
        print("no boltz_records.csv found -- run paper/figures/dump_all_wide.sh first")
        return 1
    for path in files:
        tag = os.path.basename(os.path.dirname(path))
        res[tag] = arm_means(path)
        print(f"{tag:40s} all 9 {res[tag][0]:+.4f}   excl. reference {res[tag][1]:+.4f}")

    for idx, label in ((0, "ALL 9"), (1, "NO REFERENCE")):
        E = [res[k][idx] for k in res if "a3_energy" in k]
        C = [res[k][idx] for k in res if "a1_fm" in k]
        S = [res[k][idx] for k in res if "a6_" in k]
        if len(E) < 2 or len(C) < 2:
            continue
        d, t, df = welch(E, C)
        ge, tot, p = permutation_p(E, C)
        line = (f"{label:13s} energy {mean(E):+.4f}+-{sem(E):.4f} ({len(E)} seeds)  "
                f"control {mean(C):+.4f}+-{sem(C):.4f} ({len(C)} seeds)  "
                f"Delta {d:+.4f}  t {t:.2f}  df {df:.2f}  perm p {ge}/{tot} = {p:.3f}")
        if S:
            line += f"  |  scrambled {mean(S):+.4f} ({len(S)} seeds)"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
