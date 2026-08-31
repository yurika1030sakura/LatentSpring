#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
leaveout_contamination.py -- quantify, and correct for, the validation-split
leak in the energy-supervised arms.

WHY THIS EXISTS
  The energy arms (configs/sweep/a3_*.yaml and a6_*.yaml) list TWO perturbation
  shards as training input:

      perturbation_train_n30000_s0.pt      <- built from the TRAIN split
      perturbation_val_n10000_s0.pt        <- built from the VAL split

  and scripts/eval_boltzmann_stage1.py draws its held-out parents from
  MoleculeDataset("val", ...).  The val shard is the FIRST 10,000 molecules of
  val_data_processed.pt (scripts/precompute_energy_perturbations.py takes
  val[start_index : start_index + n_molecules] in file order, start_index = 0),
  and val has 39,415 molecules.  So roughly a quarter of the evaluation parents
  were inside the energy term's training pool, WITH their true teacher energies.
  The no-physics control (a1_*.yaml) sets bgfm.enabled = false and consumes no
  shard at all, so the leak is arm-specific and favours the winning arm.

WHAT THIS SCRIPT DOES
  1. Builds a translation/rotation-invariant signature for every val molecule
     (sorted distances from the centroid, rounded to 1e-4 A) and for every
     unperturbed reference geometry (pert_id == 0) stored in each arm's
     Stage-1 JSON.  This recovers, for each evaluation group_id, the val index
     of the parent -- no RNG replay needed, and the mapping is verifiable.
  2. Confirms that shard parent p is val molecule p (atom counts agree for all
     10,000 parents).
  3. Splits each arm's per-parent Pearson r (boltz_independent.csv) into the
     parents inside the shard pool (val index < 10000) and the parents outside
     it, and recomputes the arm means and the Welch contrasts on each subset.

WHAT IT FOUND (2026-08-03, recorded in the manuscript's Sec. 5.1 and App. B.5)
  118 of 120 evaluation parents map unambiguously; 25 of those 118 lie inside
  the shard pool.  On the 93 disjoint parents the primary contrast is
  UNCHANGED-to-slightly-larger: energy +0.415 vs control +0.046,
  Delta = +0.370 (Welch t = 10.57), against +0.355 on all 120.  The no-physics
  control -- which never consumed a shard -- also scores higher on the in-pool
  parents (+0.188 vs +0.046), so the in-pool/out-of-pool gap is a property of
  the molecules, not of memorisation.

RUN
  /n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/bin/python \
      paper/figures/leaveout_contamination.py
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import statistics
import sys

import torch

ROOT = "/n/holylabs/woo_lab/Lab/yulili/bgfm/processed_data/omol25_4m_processed"
RUNS = "runs/eval_ours"
SHARD_PARENTS = 10000


def sig(P: torch.Tensor):
    """Translation- and rotation-invariant molecular signature."""
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
    return table, nia


def verify_shard_identity(nia_val):
    """Shard parent p must be val molecule p."""
    sh = torch.load(f"{ROOT}/perturbation_val_n10000_s0.pt", map_location="cpu",
                    weights_only=False)
    K = int(sh["K"]) if not torch.is_tensor(sh["K"]) else int(sh["K"].item())
    sn = sh["node_idx_array"]
    vcnt = [b - a for a, b in nia_val]
    scnt = (sn[:, 1] - sn[:, 0]).tolist()
    bad = sum(1 for p in range(SHARD_PARENTS) if scnt[p * K] != vcnt[p])
    return K, bad


def map_arm(tag, table):
    path = f"{RUNS}/{tag}/boltz1/boltzmann_samples.json"
    out = {}
    for r in json.load(open(path)):
        if r["pert_id"]:
            continue
        hit = table.get(sig(torch.tensor(r["positions"], dtype=torch.float64)))
        if hit and len(hit) == 1:
            out[r["group_id"]] = hit[0]
    return out


def mean(x):
    return sum(x) / len(x)


def welch(A, B):
    va, vb = statistics.variance(A), statistics.variance(B)
    na, nb = len(A), len(B)
    se = math.sqrt(va / na + vb / nb)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return mean(A) - mean(B), (mean(A) - mean(B)) / se, df


def main():
    table, nia_val = build_val_table()
    K, bad = verify_shard_identity(nia_val)
    print(f"val molecules      : {len(nia_val)}")
    print(f"shard K per parent : {K}  (1 reference + 4 displaced)")
    print(f"shard parent p == val molecule p : {SHARD_PARENTS - bad}/{SHARD_PARENTS} atom-count matches")

    gid2val = {}
    for tag in ("wide_a3_energy_only_s1", "wide_a1_fm_only_s2"):
        gid2val.update(map_arm(tag, table))
    inside = {g for g, v in gid2val.items() if v < SHARD_PARENTS}
    print(f"eval parents mapped: {len(gid2val)}/120; inside shard pool: {len(inside)}")

    res = {}
    for path in sorted(glob.glob(f"{RUNS}/wide_*")):
        tag = os.path.basename(path)
        rows = list(csv.DictReader(open(f"{path}/boltz_independent.csv")))
        allr = [float(r["pearson_r"]) for r in rows]
        clean = [float(r["pearson_r"]) for r in rows
                 if int(r["group_id"]) in gid2val and int(r["group_id"]) not in inside]
        pool = [float(r["pearson_r"]) for r in rows if int(r["group_id"]) in inside]
        res[tag] = (mean(allr), mean(clean), mean(pool))
        print(f"{tag:40s} all {mean(allr):+.4f} | disjoint(n={len(clean)}) {mean(clean):+.4f} "
              f"| in-pool(n={len(pool)}) {mean(pool):+.4f}")

    for label, idx in (("ALL 120", 0), ("DISJOINT 93", 1), ("IN-POOL 25", 2)):
        E = [res[k][idx] for k in res if "a3_energy" in k]
        C = [res[k][idx] for k in res if "a1_fm" in k]
        S = [res[k][idx] for k in res if "a6_" in k]
        d, t, df = welch(E, C)
        d2, t2, _ = welch(E, S)
        print(f"{label:12s} energy {mean(E):+.4f}  control {mean(C):+.4f}  scrambled {mean(S):+.4f}"
              f" | E-C Delta {d:+.4f} t {t:.2f} df {df:.2f} | E-S Delta {d2:+.4f} t {t2:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
