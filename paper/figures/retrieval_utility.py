#!/usr/bin/env python
"""Does the improved ordering buy anything? A within-group retrieval read-out.

For each held-out parent we rank its geometries by log q_theta and ask whether the
top-ranked one is the geometry an independent potential (GFN2-xTB) scores lowest.
This is a decision a practitioner would actually make with a density, it is
invariant to both the per-group constant and the scale, and it is computed from
exactly the per-record source behind Tables 1 and 2.

Reported on the primary population (93 parents disjoint from the energy term's
shard pool, reference geometry dropped) and, for comparison, on all 120.
Chance level is 1/8 = 12.5% for top-1 and 3/8 = 37.5% for top-3.

Output: paper/figures/out/retrieval_utility.json
"""
from __future__ import annotations

import csv
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EV = os.path.join(ROOT, "runs", "eval_ours")
PRIMARY = os.path.join(HERE, "out", "primary_endpoint.json")
OUT = os.path.join(HERE, "out", "retrieval_utility.json")

ARMS = {
    # "wide_a1_fm_only" (no suffix) is the fifth control seed, recovered and scored
    # under the identical protocol after the first four.
    "a1_fm_only": ["wide_a1_fm_only_s2", "wide_a1_fm_only_s3",
                   "wide_a1_fm_only_s4", "wide_a1_fm_only_s5",
                   "wide_a1_fm_only"],
    "a3_energy": ["wide_a3_energy_only_s1", "wide_a3_energy_only_s2",
                  "wide_a3_energy_only_s3", "wide_a3_energy_only_s5"],
    "a6_scrambled": ["wide_a6_energy_only_shuffled_s2",
                     "wide_a6_energy_only_shuffled_s3",
                     "wide_a6_energy_only_shuffled_stab_s5"],
}


def load(path, drop_reference=True):
    g = {}
    for row in csv.DictReader(open(path)):
        if drop_reference and int(row["pert_id"]) == 0:
            continue
        g.setdefault(int(row["group_id"]), []).append(
            (float(row["log_p_theta"]), float(row["E_eV"])))
    return g


def load_oracle(path, drop_reference=True):
    """The teacher's own energies in place of log p (the oracle ceiling)."""
    g = {}
    for row in csv.DictReader(open(path)):
        if drop_reference and int(row["pert_id"]) == 0:
            continue
        g.setdefault(int(row["group_id"]), []).append(
            (-float(row["E_esen_eV"]), float(row["E_eV"])))
    return g


def scores(groups, gids):
    top1, top3, pct = [], [], []
    for gid in gids:
        rows = groups.get(gid)
        if not rows or len(rows) < 4:
            continue
        by_density = sorted(range(len(rows)), key=lambda i: -rows[i][0])
        best_e = min(range(len(rows)), key=lambda i: rows[i][1])
        top1.append(1.0 if by_density[0] == best_e else 0.0)
        top3.append(1.0 if best_e in by_density[:3] else 0.0)
        # percentile of the true best geometry in the density ranking (1 = first)
        pct.append(1.0 - by_density.index(best_e) / (len(rows) - 1))
    return {"n": len(top1), "top1": st.fmean(top1), "top3": st.fmean(top3),
            "rank_pct": st.fmean(pct)}


def main():
    gids = json.load(open(PRIMARY))["disjoint_group_ids"]
    pops = {"primary_93": gids, "all_120": list(range(120))}
    res = {"chance_top1": 0.125, "chance_top3": 0.375, "arms": {}}
    for arm, runs in ARMS.items():
        per_seed = {}
        for r in runs:
            p = os.path.join(EV, r, "boltz_records.csv")
            if not os.path.exists(p):
                continue
            g = load(p)
            per_seed[r] = {k: scores(g, v) for k, v in pops.items()}
        agg = {}
        for pop in pops:
            for q in ("top1", "top3", "rank_pct"):
                vals = [per_seed[s][pop][q] for s in per_seed]
                agg.setdefault(pop, {})[q] = {
                    "mean": st.fmean(vals),
                    "sd": st.stdev(vals) if len(vals) > 1 else 0.0,
                    "sem": (st.stdev(vals) / len(vals) ** 0.5) if len(vals) > 1 else 0.0,
                    "per_seed": vals}
        res["arms"][arm] = {"per_seed": per_seed, "aggregate": agg}

    # oracle ceiling: the teacher's own energies in place of log q_theta, on the same
    # geometries, paired with the same independent xTB energies.
    esen = {}
    for line in open(os.path.join(EV, "ceiling_full", "esen_energies.jsonl")):
        d = json.loads(line)
        esen[(d["group_id"], d["pert_id"])] = d["E_eV"]
    xtb = {}
    ref = os.path.join(EV, ARMS["a1_fm_only"][0], "boltz_records.csv")
    for row in csv.DictReader(open(ref)):
        xtb[(int(row["group_id"]), int(row["pert_id"]))] = float(row["E_eV"])
    g = {}
    for key, e_esen in esen.items():
        if key[1] == 0 or key not in xtb:
            continue
        g.setdefault(key[0], []).append((-e_esen, xtb[key]))
    res["oracle"] = {k: scores(g, v) for k, v in pops.items()}

    json.dump(res, open(OUT, "w"), indent=1)
    for arm in res["arms"]:
        for pop in pops:
            a = res["arms"][arm]["aggregate"][pop]
            print(f"{arm:14s} {pop:11s} top1={a['top1']['mean']*100:5.1f}+-{a['top1']['sd']*100:4.1f} "
                  f"top3={a['top3']['mean']*100:5.1f}+-{a['top3']['sd']*100:4.1f} "
                  f"rankpct={a['rank_pct']['mean']*100:5.1f}")
    if "oracle" in res:
        for pop in pops:
            o = res["oracle"][pop]
            print(f"{'oracle':14s} {pop:11s} top1={o['top1']*100:5.1f} top3={o['top3']*100:5.1f} "
                  f"rankpct={o['rank_pct']*100:5.1f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
