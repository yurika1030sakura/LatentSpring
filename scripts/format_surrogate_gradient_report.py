"""Aggregate the P1 full-gradient vs surrogate-gradient JSONs into one table.

Reads every *.json written by scripts/validate_surrogate_gradient.py under a
results directory and prints, per (arm, cell, objective):
  cos(g_full, g_surr) mean/sd/min, fraction of reps with cos > 0,
  ||g_surr||/||g_full||, the exact decomposition norms
  ||g_prior||/||g_full|| and ||g_traj||/||g_full||, and the per-parameter-tensor
  cosine quantiles pooled over reps.

Usage:
    python scripts/format_surrogate_gradient_report.py \
        --dir /n/holylabs/woo_lab/Lab/yulili/bgfm/runs/p1_surrogate_gradient
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st


def cell_key(path, r):
    base = os.path.basename(path)
    if r["arm"] == "synthetic":
        return f"synthetic n={r['n_ode_steps']} " \
               f"h={'EXACT' if r['n_hutchinson'] == 0 else r['n_hutchinson']}"
    tag = base.replace("real_trainmatched_", "").replace("real_v2_", "") \
              .replace("real", "").replace(".json", "")
    return f"real {tag} n={r['n_ode_steps']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--glob", default="*.json")
    args = ap.parse_args()

    cells = {}
    for f in sorted(glob.glob(os.path.join(args.dir, args.glob))):
        d = json.load(open(f))
        for r in d["results"]:
            k = cell_key(f, r)
            for o, v in r["objectives"].items():
                cells.setdefault((k, o), []).append(v)

    hdr = (f"{'cell':<38}{'obj':>7}{'reps':>5}{'cos mean':>10}{'sd':>8}"
           f"{'min':>8}{'f>0':>6}{'|s|/|f|':>9}{'|prior|':>9}{'|traj|':>8}"
           f"{'plcos med':>10}{'plcos p10':>10}{'neg tens':>9}")
    print(hdr)
    print("-" * len(hdr))
    for (k, o), vs in sorted(cells.items()):
        c = [v["cos_full_surr"] for v in vs]
        rn = [v["rel_norm_surr_over_full"] for v in vs]
        pr = [v["norm_prior_path"] / v["norm_full"] for v in vs]
        tr = [v["norm_traj_path"] / v["norm_full"] for v in vs]
        med = [v["per_layer_cos"].get("median") for v in vs
               if v.get("per_layer_cos")]
        p10 = [v["per_layer_cos"].get("p10") for v in vs
               if v.get("per_layer_cos")]
        neg = [v.get("n_tensors_negative_cos") for v in vs
               if v.get("n_tensors_negative_cos") is not None]
        sd = st.stdev(c) if len(c) > 1 else 0.0
        print(f"{k:<38}{o:>7}{len(c):>5}{st.mean(c):>10.4f}{sd:>8.4f}"
              f"{min(c):>8.4f}{sum(x > 0 for x in c) / len(c):>6.2f}"
              f"{st.mean(rn):>9.4f}{st.mean(pr):>9.4f}{st.mean(tr):>8.4f}"
              f"{(st.mean(med) if med else float('nan')):>10.3f}"
              f"{(st.mean(p10) if p10 else float('nan')):>10.3f}"
              f"{(st.mean(neg) if neg else float('nan')):>9.1f}")

    # descent check across everything
    all_v = [v for vs in cells.values() for v in vs]
    n_desc = sum(1 for v in all_v if v["descent_first_order"] == "descent")
    print(f"\nfirst-order descent test: {n_desc}/{len(all_v)} measurements have "
          f"<g_full, g_surr> > 0")
    print("random-direction cosine reference (D params): "
          f"{all_v[0]['cos_random_direction_reference']:.2e}")


if __name__ == "__main__":
    raise SystemExit(main())
