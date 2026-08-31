#!/usr/bin/env python3
"""Seed-selection sensitivity for the primary endpoint.

Reviewer request (2026-08-04): the reported means are conditioned on completion,
so report whether the primary contrast survives an adversarial imputation of the
runs that are not reported.

REVISED 2026-08-17.  The flow-matching control arm is now complete: its fifth
prepared seed trained to the full budget with finite weights but had never been
scored under this protocol, and has since been scored under it (tag
wide_a1_fm_only, 120 parents, n_ode = 12, same evaluation seed).  The control arm
therefore has NO unreported run, and the imputation charges the value arm alone --
5 of 5, 4 of 5 and 3 of 7 prepared seed configurations for the no-physics,
true-label and scrambled arms.

Reads paper/figures/out/primary_endpoint.json (per-seed per-group correlations
already aggregated by recompute_disjoint_noreference.py) and writes
paper/figures/out/seed_sensitivity.json.

No GPU, no re-evaluation: this is arithmetic on values already on disk.
"""
import itertools
import json
import os
import statistics as st
from math import sqrt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "out", "primary_endpoint.json")
DST = os.path.join(HERE, "out", "seed_sensitivity.json")
KEY = "disjoint|noref"          # the primary endpoint

# Prepared seed configurations per arm (app:hyper); the difference between
# "prepared" and "reported" is what this script stresses.
PREPARED = {"fm_only": 5, "energy": 5, "scrambled": 7}


def welch(a, b):
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.variance(a), st.variance(b)
    se = sqrt(va / len(a) + vb / len(b))
    t = (ma - mb) / se
    df = se ** 4 / ((va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
    return ma - mb, t, df


def perm_p(a, b):
    """Exact two-sided permutation p over seed relabellings."""
    pool = list(a) + list(b)
    n = len(a)
    obs = abs(st.mean(a) - st.mean(b))
    hits = tot = 0
    for idx in itertools.combinations(range(len(pool)), n):
        g1 = [pool[i] for i in idx]
        g2 = [pool[i] for i in range(len(pool)) if i not in idx]
        tot += 1
        if abs(st.mean(g1) - st.mean(g2)) >= obs - 1e-12:
            hits += 1
    return hits, tot, hits / tot


def block(name, en, fm):
    d, t, df = welch(en, fm)
    hits, tot, p = perm_p(en, fm)
    return {
        "label": name,
        "energy_seeds": [round(x, 6) for x in en],
        "control_seeds": [round(x, 6) for x in fm],
        "energy_mean": round(st.mean(en), 6),
        "control_mean": round(st.mean(fm), 6),
        "delta": round(d, 6),
        "welch_t": round(t, 4),
        "welch_df": round(df, 3),
        "perm_hits": hits,
        "perm_total": tot,
        "perm_p": round(p, 6),
    }


def main():
    cells = json.load(open(SRC))["cells"]
    fm = [v[KEY] for k, v in cells.items() if "a1_fm" in k]
    en = [v[KEY] for k, v in cells.items() if "a3_energy" in k]
    sc = [v[KEY] for k, v in cells.items() if "a6" in k]
    worst_overall = min(fm + en + sc)

    out = {
        "endpoint": KEY,
        "prepared_seed_configurations": PREPARED,
        "reported": {"fm_only": len(fm), "energy": len(en), "scrambled": len(sc)},
        "blocks": [
            block("as reported (4 vs 5)", en, fm),
            # The control arm is complete, so only the value arm's one diverged run
            # is imputed.  Mean imputation: it behaves like the control arm's mean.
            block("value run imputed at control mean (5 vs 5)",
                  en + [st.mean(fm)], fm),
            # Extreme imputation: it takes the worst value observed in any arm.
            block("value run imputed at worst observed (5 vs 5)",
                  en + [worst_overall], fm),
        ],
        "scramble_worst_case": {
            "note": "4 unreported scrambled seeds imputed at the true-label mean / maximum",
            "at_energy_mean": {
                "scrambled_mean": round(st.mean(sc + [st.mean(en)] * 4), 6),
                "energy_minus_scrambled": round(st.mean(en) - st.mean(sc + [st.mean(en)] * 4), 6),
                "welch_t": round(welch(en, sc + [st.mean(en)] * 4)[1], 4),
            },
            "at_energy_max": {
                "scrambled_mean": round(st.mean(sc + [max(en)] * 4), 6),
                "energy_minus_scrambled": round(st.mean(en) - st.mean(sc + [max(en)] * 4), 6),
                "welch_t": round(welch(en, sc + [max(en)] * 4)[1], 4),
            },
        },
    }
    with open(DST, "w") as fh:
        json.dump(out, fh, indent=1)
    for b in out["blocks"]:
        print("%-24s Delta = %+.4f  t = %5.2f  perm p = %.4f (%d/%d)"
              % (b["label"], b["delta"], b["welch_t"], b["perm_p"],
                 b["perm_hits"], b["perm_total"]))
    print("wrote", DST)


if __name__ == "__main__":
    main()
