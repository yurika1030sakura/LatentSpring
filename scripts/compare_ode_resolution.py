"""Is the FFJORD log-density converged enough for BETWEEN-basin comparisons?

The global-ensemble metrics are built from log p DIFFERENCES between basins of
the same molecule. Those differences are only meaningful if they are stable
under the ODE discretisation. This script compares two (or more) `basin_logp`
files produced by scripts/eval_global_ensemble.py at different `--n_ode_steps`
and reports, per system:

  * the correlation of the mean-centred basin log p vectors between resolutions
    (1.0 = the ranking and spacing of basins is identical)
  * the spread (std across basins) at each resolution
  * the Boltzmann-allowed spread beta * std_b(E_b), i.e. how large the spread
    would be if the model were exactly Boltzmann at kT

The last column is the one that matters: if the measured spread exceeds the
Boltzmann-allowed spread by orders of magnitude at EVERY resolution, the
qualitative conclusion survives the discretisation uncertainty even though the
precise number does not.

Usage:
  python scripts/compare_ode_resolution.py --kT_eV 1.0 \
      --label ode6  <dir>/basin_logp_partial.json \
      --label ode12 <dir>/basin_logp_partial.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HARTREE_TO_EV = 27.211386245988


def load_minima(path):
    d = json.load(open(path))
    out = {}
    for r in d["records"]:
        if r["pert_id"] != 0:
            continue
        out.setdefault(r["system_id"], []).append(
            (r["basin_id"], r["log_p_theta"], r.get("ref_energy_eV_basin_min")))
    res = {}
    for sid, rows in out.items():
        rows.sort()
        res[sid] = (np.array([x[1] for x in rows], float),
                    np.array([x[2] for x in rows], float))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="append", required=True)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--kT_eV", type=float, default=1.0)
    a = ap.parse_args()
    if len(a.label) != len(a.files):
        raise SystemExit("need one --label per file")

    sets = [load_minima(f) for f in a.files]
    common = sorted(set.intersection(*[set(s) for s in sets]))
    if not common:
        print("no systems in common")
        return 1

    beta = 1.0 / a.kT_eV
    print(f"{'system':>8} {'nb':>3} " +
          " ".join(f"{'spread_' + l:>12}" for l in a.label) +
          f" {'boltz_spread':>13} " +
          " ".join(f"{'corr_' + a.label[0] + '_' + l:>16}" for l in a.label[1:]))
    rows = []
    for sid in common:
        lps = [s[sid][0] for s in sets]
        E = sets[0][sid][1]
        nb = len(lps[0])
        boltz = float(np.std(beta * E))
        spreads = [float(np.std(v)) for v in lps]
        base = lps[0] - lps[0].mean()
        corrs = []
        for v in lps[1:]:
            v = v - v.mean()
            corrs.append(float(np.corrcoef(base, v)[0, 1])
                         if min(base.std(), v.std()) > 1e-12 else float("nan"))
        rows.append((sid, nb, spreads, boltz, corrs))
        print(f"{sid:>8} {nb:>3} " + " ".join(f"{s:>12.3f}" for s in spreads) +
              f" {boltz:>13.4f} " + " ".join(f"{c:>16.3f}" for c in corrs))

    print()
    for i, l in enumerate(a.label):
        sp = np.array([r[2][i] for r in rows])
        bo = np.array([r[3] for r in rows])
        print(f"{l:>8}: mean basin log p spread {sp.mean():.3f} nats; "
              f"mean Boltzmann-allowed spread {bo.mean():.4f} nats; "
              f"ratio {np.mean(sp / np.maximum(bo, 1e-9)):.1f}x")
    for j, l in enumerate(a.label[1:]):
        c = np.array([r[4][j] for r in rows])
        c = c[np.isfinite(c)]
        print(f"corr({a.label[0]}, {l}) over systems: mean {c.mean():.3f}, "
              f"min {c.min():.3f}, n={c.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
