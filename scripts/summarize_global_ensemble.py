"""Collate global-ensemble results across arms into a markdown table.

Reads <root>/<arm>/global_ensemble_results.json for each arm directory and emits
a comparison table plus a PAIRED (per-system) energy-arm vs control-arm test,
which is the statistically correct comparison because both arms are evaluated on
the identical set of systems and basins.

Usage:
  python scripts/summarize_global_ensemble.py --root <dir> --arms energy_s2 control_s2
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    from scipy import stats as _st
except Exception:
    _st = None

HEADLINE = [
    ("n_systems", "systems", None),
    ("mean_within_basin_r", "within-basin r (old metric)", "mean"),
    ("r_basin_mass", "between-basin r (basin mass)", "mean"),
    ("slope_basin_mass", "between-basin slope (ideal 1)", "mean"),
    ("nrv_basin_mass", "NRV basin mass (ideal 0)", "mean"),
    ("nrv_within", "NRV within basins (ideal 0)", "mean"),
    ("kT_eff_basin_mass_eV", "kT_eff between basins (eV)", "median"),
    ("dF_err_mass_eV", "basin dF error, mass (eV)", "mean"),
    ("dF_true_mass_eV", "true basin dF spread, mass (eV)", "mean"),
    ("dF_err_mass_ratio", "dF err / true (1 = flat, 0 = ideal)", "mean"),
    ("dF_err_min_eV", "basin dF error, minima (eV)", "mean"),
    ("dF_true_min_eV", "true basin dF spread, minima (eV)", "mean"),
    ("r_pointwise", "pointwise r at minima", "mean"),
    ("r_pointwise_partial_Rg", "pointwise r | Rg (confound ctrl)", "mean"),
    ("kl_ref_model", "KL(P_ref || P_model)", "mean"),
    ("tv", "total variation", "mean"),
    ("w1_energy_eV", "W1 on basin energy (eV)", "mean"),
    ("ess_frac_basins", "basin-weight ESS frac", "mean"),
    ("is_ess_frac_ref", "IS ESS frac, ref (est. quality)", "mean"),
    ("is_ess_frac_model", "IS ESS frac, model (est. quality)", "mean"),
    ("frac_top1_match", "top-1 basin match", None),
]


def load(root: Path, arm: str):
    p = root / arm / "global_ensemble_results.json"
    if not p.exists():
        return None
    return json.load(open(p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--paired", nargs=2, default=None,
                    metavar=("ENERGY_ARM", "CONTROL_ARM"))
    ap.add_argument("--out_md", type=Path, default=None)
    a = ap.parse_args()

    data = {arm: load(a.root, arm) for arm in a.arms}
    have = [arm for arm in a.arms if data[arm]]
    if not have:
        print("no results found")
        return 1

    lines = []
    lines.append("| metric | " + " | ".join(have) + " |")
    lines.append("|---|" + "---|" * len(have))
    for key, label, field in HEADLINE:
        cells = []
        for arm in have:
            s = data[arm]["summary"]
            v = s.get(key)
            if isinstance(v, dict):
                v = v.get(field or "mean")
            if v is None:
                cells.append("--")
            elif isinstance(v, float):
                cells.append(f"{v:.4g}")
            else:
                cells.append(str(v))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print(table)

    paired_txt = ""
    if a.paired and all(data.get(x) for x in a.paired):
        e_arm, c_arm = a.paired
        E = {r["system_id"]: r for r in data[e_arm]["per_system"]}
        C = {r["system_id"]: r for r in data[c_arm]["per_system"]}
        common = sorted(set(E) & set(C))
        rows = []
        for key in ["r_basin_mass", "slope_basin_mass", "nrv_basin_mass",
                    "dF_err_mass_eV", "dF_err_mass_ratio", "dF_err_min_eV",
                    "kl_ref_model", "tv", "mean_within_basin_r", "nrv_within",
                    "r_pointwise", "ess_frac_basins"]:
            de = np.array([E[s][key] for s in common], float)
            dc = np.array([C[s][key] for s in common], float)
            m = np.isfinite(de) & np.isfinite(dc)
            de, dc = de[m], dc[m]
            if de.size < 2:
                continue
            d = de - dc
            t = d.mean() / (d.std(ddof=1) / math.sqrt(d.size)) if d.std(ddof=1) > 0 else float("nan")
            p = (float(2 * (1 - _st.t.cdf(abs(t), d.size - 1)))
                 if (_st is not None and np.isfinite(t)) else float("nan"))
            rows.append((key, de.mean(), dc.mean(), d.mean(), t, p, d.size))
        paired_txt = ("\n\n**Paired per-system comparison "
                      f"({e_arm} - {c_arm}, n={len(common)} shared systems)**\n\n"
                      "| metric | energy | control | Δ | t | p |\n|---|---|---|---|---|---|\n")
        for k, me, mc, md, t, p, n in rows:
            paired_txt += f"| {k} | {me:.4g} | {mc:.4g} | {md:+.4g} | {t:.2f} | {p:.3g} |\n"
        print(paired_txt)

    if a.out_md:
        a.out_md.write_text(table + paired_txt + "\n")
        print(f"wrote {a.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
