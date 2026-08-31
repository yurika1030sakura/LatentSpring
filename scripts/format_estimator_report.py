"""Render the P6 estimator-validation report.json into markdown tables.

Usage:
  python scripts/format_estimator_report.py <out_dir> [<out_dir> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _f(x, n=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:
        return "n/a"
    return f"{v:.{n}f}"


def _key(k):
    s, p = k.replace("steps", "").split("_probes")
    return int(s), int(p)


def render(out_dir: Path, name: str, report_file: str = "report.json") -> str:
    rp = out_dir / report_file
    if not rp.exists():
        return f"\n_(no {report_file} in {out_dir})_\n"
    r = json.load(open(rp))
    L = []
    L.append(f"\n### {name}  (`{out_dir}`)\n")
    L.append(f"kT = {r.get('kT_eV')} eV, "
             f"reference conformer {'DROPPED' if r.get('drop_reference_conformer') else 'included'}, "
             f"reference (highest-precision) cell = `{r.get('reference_cell','?')}`\n")

    L.append("\n#### A/E. log p estimator precision vs. the headline metric\n")
    L.append("| ODE steps | probes | mean log p | estimator SD (marginal) | estimator SD (within-group centred) | within-group signal SD | SNR | mean per-group r | SEM | r (repeat-avg log p) | attenuation-corrected r | median slope | median NRV | s/rep |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    L.append("<!-- probes < 0 = common random numbers (probe pattern tiled across the geometries of a group) -->")
    for k in sorted(r["cells"], key=_key):
        c = r["cells"][k]
        s, p = _key(k)
        L.append("| {} | {} | {} | {} | {} | {} | {} | **{}** | {} | {} | {} | {} | {} | {} |".format(
            s, p, _f(c["logp_mean"], 2), _f(c["estimator_sd_mean"], 3),
            _f(c.get("estimator_sd_centred_mean"), 3),
            _f(c["within_group_signal_sd_mean"], 3), _f(c["snr_signal_over_noise"], 2),
            _f(c["mean_per_group_r"]), _f(c["sem_per_group_r"]),
            _f(c["mean_per_group_r_repavg"]), _f(c.get("attenuation_corrected_r")),
            _f(c["median_per_group_slope"], 3), _f(c["median_per_group_nrv"], 2),
            _f(c["seconds_per_rep"], 0)))

    L.append("\n#### B. Is the estimator NOISE geometry-correlated?\n")
    L.append("| ODE steps | probes | corr(SD_c, n_atoms) | corr(SD_c, RMSD) | corr(SD_c, -E) pooled | mean within-group corr(SD_c, -E) | NULL r (shuffled-group energies) | NULL mean abs r |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k in sorted(r["cells"], key=_key):
        c = r["cells"][k]
        s, p = _key(k)
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            s, p, _f(c["corr_noise_vs_n_atoms"]), _f(c["corr_noise_vs_rmsd"]),
            _f(c["corr_noise_vs_negE_pooled"]),
            _f(c["mean_within_group_corr_noise_vs_negE"]),
            _f(c.get("null_r_shuffled_group_energies")),
            _f(c.get("null_r_abs_mean"))))

    if r.get("bias_vs_reference"):
        L.append("\n#### C. Systematic BIAS vs. the highest-precision cell\n")
        L.append("| ODE steps | probes | mean bias | SD bias | mean abs bias | corr(bias, n_atoms) | corr(bias, RMSD) | mean within-group corr(bias, -E) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for k in sorted(r["bias_vs_reference"], key=_key):
            b = r["bias_vs_reference"][k]
            s, p = _key(k)
            L.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                s, p, _f(b["mean_bias"], 2), _f(b["sd_bias"], 2),
                _f(b["mean_abs_bias"], 2), _f(b["corr_bias_vs_n_atoms"]),
                _f(b["corr_bias_vs_rmsd"]),
                _f(b["mean_within_group_corr_bias_vs_negE"])))

    if r.get("headline_cell_pooled"):
        h = r["headline_cell_pooled"]
        L.append("\n#### F. Headline cell (12 steps / 2 probes), pooled within-group-centred\n")
        L.append(f"- pooled r(log p, -E) = **{_f(h['pooled_r'])}**  (n = {h['n']})")
        L.append(f"- partial r(log p, -E | RMSD) = **{_f(h['pooled_r_partial_given_rmsd'])}**")
        L.append(f"- r(log p, RMSD) = {_f(h['corr_logp_vs_rmsd'])}, "
                 f"r(-E, RMSD) = {_f(h['corr_negE_vs_rmsd'])}")

    ex = r.get("exact_vs_hutchinson")
    if ex and ex.get("molecules"):
        L.append("\n#### D. Hutchinson vs. EXACT divergence (small molecules)\n")
        L.append("| group | n_atoms | ODE steps | probes | mean bias vs exact | SD over repeats | |exact| spread within group |")
        L.append("|---|---|---|---|---|---|---|")
        for m in ex["molecules"]:
            le = m["logp_exact"]
            spread = (max(le) - min(le)) if len(le) > 1 else float("nan")
            for nh, d in sorted(m["hutchinson"].items(), key=lambda kv: int(kv[0])):
                bias = d["bias_vs_exact"]
                sd = d["sd"]
                mb = sum(bias) / len(bias)
                ms = sum(sd) / len(sd)
                L.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                    m["group_id"], m["n_atoms"], m["n_ode_steps"], nh,
                    _f(mb, 3), _f(ms, 3), _f(spread, 2)))
    return "\n".join(L) + "\n"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    for d in sys.argv[1:]:
        p = Path(d)
        print(render(p, p.name))
        alt = p / "report_dropref.json"
        if alt.exists():
            print(render(p, p.name + " [reference conformer dropped]",
                         "report_dropref.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
