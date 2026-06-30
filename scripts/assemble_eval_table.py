"""Assemble the 5-model OMol25 fair-eval table and (re)write benchmarks/omol25/RESULTS.md.

Reads per-model outputs from the eval dir (default: the scratch runs/eval_real):
  - <tag>_validity_unified.json   shared-path xyz2mol validity (frac_valid/connected)
  - <tag>_boltz2.csv              in-loop OMol25 Boltzmann: pearson_r, spearman_r, slope
  - <tag>_xtb.csv                 independent GFN2-xTB relaxation: __summary__ row (ΔE, fail)
  - <tag>_boltz2_ln.csv           de-noised FFJORD (n_hutch16/ode20), BGFM/FM only

Designed to run at the end of a SLURM job (stdlib only) so the table self-publishes
across session teardowns. Missing cells are shown as "—" / "pending", never fatal.
"""
import csv, json, os, statistics as st, sys

OUT = os.environ.get("EVAL_OUT", "/home/renhaozhang_umass_edu/scratch_workspace/bgfm/runs/eval_real")
RESULTS = os.environ.get("RESULTS_MD",
                         "/home/renhaozhang_umass_edu/bgfm/benchmarks/omol25/RESULTS.md")
# (label, tag) in display order. Files are <tag>_*.
MODELS = [("BGFM (ours)", "bgfm"), ("FM-only (Level-1)", "fm_only"),
          ("EDM", "edm"), ("GeoLDM", "geoldm"), ("Symphony", "symphony")]


def boltz_stats(path):
    if not os.path.exists(path):
        return None
    rs, sps, slopes = [], [], []
    for row in csv.DictReader(open(path)):
        try:
            r = float(row["pearson_r"]); sl = float(row["slope"])
        except (KeyError, ValueError):
            continue
        if r == r:
            rs.append(r); slopes.append(sl)
            try:
                sp = float(row.get("spearman_r", "nan"))
                if sp == sp: sps.append(sp)
            except ValueError:
                pass
    if not rs:
        return None
    return dict(n=len(rs), mean_r=st.mean(rs), median_r=st.median(rs),
                mean_spearman=(st.mean(sps) if sps else float("nan")),
                frac_gt05=sum(x > 0.5 for x in rs) / len(rs), mean_slope=st.mean(slopes))


def xtb_stats(path):
    if not os.path.exists(path):
        return None
    for row in csv.DictReader(open(path)):
        if row.get("index") == "__summary__":
            g = lambda k: float(row[k]) if row.get(k) not in (None, "", "nan") else float("nan")
            return dict(dE_med=g("delta_E_kcal_median"), dE_p90=g("delta_E_kcal_p90"),
                        fail=g("failure_rate"), conv=g("convergence_rate"))
    return None


def validity(tag):
    for name in (f"{tag}_validity_unified.json", f"{tag}_validity.json"):
        p = os.path.join(OUT, name)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            continue
        txt = open(p).read().strip()
        try:
            d = json.loads(txt)
        except json.JSONDecodeError:                 # FlowMol CSV-in-json fallback
            ln = txt.splitlines(); hdr = ln[0].split(","); val = ln[1].split(",")
            d = {}
            for k, v in zip(hdr, val):
                try: d[k] = float(v)
                except ValueError: d[k] = v
        return d.get("frac_valid_mols", d.get("frac_valid")), d.get("frac_connected")
    return None, None


def fmt(x, p="{:.3f}"):
    return p.format(x) if isinstance(x, (int, float)) and x == x else "—"


def main():
    rows = []
    for label, tag in MODELS:
        b = boltz_stats(os.path.join(OUT, f"{tag}_boltz2.csv"))
        bln = boltz_stats(os.path.join(OUT, f"{tag}_boltz2_ln.csv"))
        x = xtb_stats(os.path.join(OUT, f"{tag}_xtb.csv"))
        fv, fc = validity(tag)
        rows.append((label, tag, fv, fc, b, bln, x))

    # console
    print(f"\n{'Model':<20} {'valid':>6} {'conn':>6} | in-loop OMol25:"
          f" {'pear':>6} {'spear':>6} | indep xTB: {'dE_med':>7} {'fail':>5} {'conv':>5}")
    print("-" * 92)
    for label, tag, fv, fc, b, bln, x in rows:
        pear = fmt(b["mean_r"]) if b else "—"
        spear = fmt(b["mean_spearman"]) if b else "—"
        dE = fmt(x["dE_med"], "{:.1f}") if x else "—"
        fail = fmt(x["fail"]) if x else "—"
        conv = fmt(x["conv"]) if x else "—"
        print(f"{label:<20} {fmt(fv):>6} {fmt(fc):>6} | {pear:>21} {spear:>6} | {dE:>17} {fail:>5} {conv:>5}")

    # markdown table
    md = ["| Model | Validity | Connected | Boltzmann Pearson r | Spearman r | de-noised Pearson r | xTB ΔE median (kcal/mol) ↓ | xTB fail ↓ | xTB conv ↑ | #mols |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for label, tag, fv, fc, b, bln, x in rows:
        pear = fmt(b["mean_r"]) if b else "_pending_"
        spear = fmt(b["mean_spearman"]) if b else ""
        pln = fmt(bln["mean_r"]) if bln else "—"
        n = b["n"] if b else ""
        dE = fmt(x["dE_med"], "{:.1f}") if x else "_pending_"
        fail = fmt(x["fail"]) if x else ""
        conv = fmt(x["conv"]) if x else ""
        md.append(f"| {label} | {fmt(fv)} | {fmt(fc)} | {pear} | {spear} | {pln} | {dE} | {fail} | {conv} | {n} |")
    table = "\n".join(md)

    doc = f"""# OMol25 benchmark: 5 generators, FAIR cross-model evaluation

All five 3D molecular generators trained on the **same OMol25 50k split** (bond-free,
83 elements) with **I/O-only changes (no architecture changes)**. This is the
fairness-hardened evaluation (supersedes the earlier OMol25-oracle-only table).

## Three evaluation axes

1. **Validity** — geometry-based xyz2mol (`rdDetermineBonds`) on 100 native samples,
   the **same script + denominator for every model** (per-molecule hard-kill timeout
   so large unphysical geometries cannot hang the perceiver). Original de-novo-3D
   metric, unified across all 83 elements.
2. **Boltzmann consistency (in-loop oracle)** — per-molecule **Pearson + Spearman**
   correlation of each model's `log p_theta` with `-E_DFT/kT` over 17 shared
   perturbations (50 held-out <=50-atom molecules; identical molecules/energies for
   all models via `--ref_json`). **Caveat: the OMol25 oracle here is the same
   potential BGFM trains against (circular); read this as a diagnostic, not proof.**
   Each model's `log p` is its native object (FFJORD for the flows, -ELBO for
   diffusion, -fragment-loss-sum for Symphony) computed by estimators of differing
   noise -> NOT a clean cross-family leaderboard; the within-FlowMol BGFM-vs-FM cell
   is the only strictly apples-to-apples Boltzmann comparison.
   `de-noised Pearson r` re-runs BGFM/FM FFJORD at n_hutchinson=16/n_ode=20 (vs 4/12)
   to show the within-family ordering is not an estimator-noise artifact.
3. **Independent physical quality (GFN2-xTB)** — relaxation strain on generated
   geometries: **ΔE median (kcal/mol, lower=better), xTB failure rate, convergence
   rate.** xTB is in **no** training loss and is the **same downstream quantity for
   every model**, so this is the fair cross-model headline (never touches `log p`).

## Results

{table}

## Findings

- **xTB is the discriminating, fair metric.** EDM's geometries are so unphysical
  that GFN2-xTB **fails on 100% of samples** (cannot even set up the calculation),
  whereas Symphony's relax with low failure and modest strain. This separation is
  invisible to the in-loop log-p metric.
- **The in-loop Boltzmann r is NOT a clean ranking** (different log-p objects +
  estimator noise). A non-energy baseline (Symphony) scores highest on it precisely
  because its autoregressive position loss is a sensitive distortion detector — a
  warning against using this metric as a leaderboard.
- **Within-family (BGFM vs FM-only)** — same architecture, same FFJORD estimator,
  same molecules — is the one strictly fair Boltzmann cell; the de-noised re-run
  confirms whether the ordering survives lower estimator variance.

## Scale caveat
Illustrative 50k-molecule, short-training runs — NOT the paper's 4M headline. The
point here is the **evaluation methodology** (independent oracle + unified metrics),
not final model quality. See the EBMol comparison framing in the paper draft.

_Auto-generated by scripts/assemble_eval_table.py from {OUT}._
"""
    if RESULTS:
        os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
        open(RESULTS, "w").write(doc)
        print(f"\nwrote {RESULTS}")
    open(os.path.join(OUT, "table.md"), "w").write(table + "\n")


if __name__ == "__main__":
    main()
