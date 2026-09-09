#!/usr/bin/env python3
"""Recompute the archived evidence without GPUs, private paths, or ML packages.

Resolution comparisons use the intersection of checkpoints at every requested
resolution. Every re-score is joined by (group_id, pert_id), with the archived
likelihood checked before attaching its energy. Seed means, not geometries,
are the units of replication. The 93-parent membership is an archived input;
this script does not independently certify molecular-identity disjointness.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st


def summary(values):
    values = list(values)
    return {"n_seeds": len(values), "mean": st.mean(values) if values else None,
            "sem": st.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None}


def pearson(x, y):
    if len(x) < 3 or len(x) != len(y):
        return None
    dx = [v - st.mean(x) for v in x]
    dy = [v - st.mean(y) for v in y]
    den = math.sqrt(sum(v*v for v in dx) * sum(v*v for v in dy))
    return sum(a*b for a, b in zip(dx, dy)) / den if den > 1e-12 else None


def load_records(path):
    records = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            key = int(row["group_id"]), int(row["pert_id"])
            if key in records:
                raise ValueError(f"Duplicate record {key} in {path}")
            records[key] = row
    return records


def group_scores(records, groups, replacement=None):
    by_group = {}
    for (gid, pid), row in records.items():
        if gid not in groups or pid == 0 or row.get("xtb_ok", "true").lower() == "false":
            continue
        lp = float(row["log_p_theta"]) if replacement is None else replacement[(gid, pid)]
        energy = float(row["negE_kT"])
        if math.isfinite(lp) and math.isfinite(energy):
            by_group.setdefault(gid, []).append((pid, lp, energy))
    rs, nrvs, top1, top3 = [], [], [], []
    for rows in by_group.values():
        rows.sort()
        x, y = [r[1] for r in rows], [r[2] for r in rows]
        r = pearson(x, y)
        if r is None:
            continue
        rs.append(r)
        nrvs.append(st.pvariance([a-b for a, b in zip(x, y)]) / st.pvariance(y))
        # Deterministic average credit under ties, avoiding CSV-order advantage.
        best = [i for i, v in enumerate(y) if v == max(y)]
        for k, output in ((1, top1), (3, top3)):
            credit = []
            for i in best:
                above = sum(v > x[i] for v in x)
                tied = sum(v == x[i] for v in x)
                credit.append(max(0.0, min(1.0, (k-above)/tied)))
            output.append(st.mean(credit))
    return {"n_groups": len(rs), "r": st.mean(rs) if rs else None,
            "median_nrv": st.median(nrvs) if nrvs else None,
            "top1": st.mean(top1) if top1 else None,
            "top3": st.mean(top3) if top3 else None}


def arm_of(tag):
    for prefix, arm in (("a1_fm_only", "fm"), ("a3_energy_only", "value"),
                        ("a6_energy_only_shuffled", "scrambled")):
        if tag.startswith(prefix):
            return arm
    return None


def audit(root, resolutions=(12, 24, 48)):
    files = set()
    membership = root / "paper/figures/out/primary_endpoint.json"
    files.add(membership)
    ids = set(json.loads(membership.read_text())["disjoint_group_ids"])
    result = {"membership_source": str(membership.relative_to(root)),
              "n_disjoint_parents": len(ids),
              "membership_caveat": "Archived group membership, not an independent identity/split audit.",
              "likelihood_join_tolerance": {"absolute": 1e-5, "relative": 2e-7},
              "primary": {}, "resolution": {}, "generation": {}}
    original = {}
    for path in sorted((root / "results/eval_ours").glob("wide_*/boltz_records.csv")):
        tag = path.parent.name.removeprefix("wide_")
        if arm_of(tag) is None:
            continue
        files.add(path)
        original[tag] = load_records(path)
        result["primary"][tag] = group_scores(original[tag], ids)

    rescored = {}
    for path in sorted((root / "results/rescore").glob("*.json")):
        tag, suffix = path.stem.rsplit("__n", 1)
        n = int(suffix)
        if n not in resolutions or arm_of(tag) is None:
            continue
        files.add(path)
        data = json.loads(path.read_text())
        if data["n_ode_steps"] != n:
            raise ValueError(f"Resolution metadata mismatch in {path}")
        replacement = {}
        for row in data["records"]:
            key = int(row["group_id"]), int(row["pert_id"])
            if key in replacement:
                raise ValueError(f"Duplicate rescore key in {path}: {key}")
            old_row = original[tag].get(key)
            # The energy CSV omits failed oracle evaluations; a rescore still
            # contains those geometries. They cannot enter an energy metric.
            if old_row is not None and not math.isclose(float(old_row["log_p_theta"]), row["log_p_theta_published"], abs_tol=1e-5, rel_tol=2e-7):
                raise ValueError(f"Published likelihood mismatch in {path}: {key}")
            replacement[key] = row["log_p_theta_new"]
        if not original[tag].keys() <= replacement.keys():
            raise ValueError(f"Incomplete geometry set in {path}")
        rescored[(tag, n)] = group_scores(original[tag], ids, replacement)
    matched = sorted({tag for tag, n in rescored
                      if all((tag, step) in rescored for step in resolutions)})
    result["matched_resolution_checkpoints"] = matched
    for n in resolutions:
        per_seed = {tag: rescored[tag, n] for tag in matched}
        arms = {arm: summary(v["r"] for tag, v in per_seed.items() if arm_of(tag) == arm)
                for arm in ("fm", "value", "scrambled")}
        result["resolution"][str(n)] = {"per_seed": per_seed, "arms": arms,
            "value_minus_fm": arms["value"]["mean"]-arms["fm"]["mean"],
            "value_minus_scrambled": arms["value"]["mean"]-arms["scrambled"]["mean"]}

    for path in sorted((root / "results/genstrain").glob("*/xtb_relax.csv")):
        files.add(path)
        with path.open() as handle:
            rows = [r for r in csv.DictReader(handle) if r["index"] != "__summary__"]
        success = [r for r in rows if not r["failure_reason"]
                   and r["delta_E_kcal_per_atom"]
                   and math.isfinite(float(r["delta_E_kcal_per_atom"]))]
        converged = [r for r in success if r["converged"].lower() == "true"]
        result["generation"][path.parent.name] = {
            "n_generated": len(rows), "n_finite_strain": len(success),
            "n_converged": len(converged), "failure_rate": 1-len(success)/len(rows),
            "unconverged_fraction": 1-len(converged)/len(rows),
            "mean_strain_kcal_per_atom": st.mean(float(r["delta_E_kcal_per_atom"]) for r in success),
            "median_strain_kcal_per_atom": st.median(float(r["delta_E_kcal_per_atom"]) for r in success),
            "converged_mean_strain_kcal_per_atom": st.mean(float(r["delta_E_kcal_per_atom"]) for r in converged) if converged else None}
    result["generation_arms"] = {arm: {
        metric: summary(v[metric] for tag, v in result["generation"].items() if arm_of(tag) == arm)
        for metric in ("mean_strain_kcal_per_atom", "median_strain_kcal_per_atom", "failure_rate", "unconverged_fraction")}
        for arm in ("fm", "value", "scrambled")}
    result["primary_arms"] = {arm: {
        metric: summary(v[metric] for tag, v in result["primary"].items() if arm_of(tag) == arm)
        for metric in ("r", "top1", "top3", "median_nrv")}
        for arm in ("fm", "value", "scrambled")}
    result["input_sha256"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(files)}
    return result


def render(data):
    lines = ["# Archived evidence audit", "", "Generated from checked-in records; uncertainty is SEM across completed training seeds.",
             "The legacy scalar readout is not a validated generator likelihood.", "",
             "## Matched checkpoint resolution sweep", "",
             "Same record keys and archived likelihoods verified before joining energies. Reference removed; archived 93-parent filter applied.", "",
             "| Steps | FM r | Value r | Scrambled r | Value - FM | Value - scrambled |",
             "|---|---|---|---|---|---|"]
    for n, row in data["resolution"].items():
        vals = [f"{row['arms'][a]['mean']:.4f} ± {row['arms'][a]['sem']:.4f} ({row['arms'][a]['n_seeds']})" for a in ("fm", "value", "scrambled")]
        lines.append(f"| {n} | " + " | ".join(vals) + f" | {row['value_minus_fm']:+.4f} | {row['value_minus_scrambled']:+.4f} |")
    lines += ["", "## Generation strain (lower is better)", "",
              "| Arm | Seeds | Mean strain, kcal/mol/atom | Failure rate | Not converged (all samples) |",
              "|---|---|---|---|---|"]
    for arm, row in data["generation_arms"].items():
        v = row["mean_strain_kcal_per_atom"]
        lines.append(f"| {arm} | {v['n_seeds']} | {v['mean']:.4f} ± {v['sem']:.4f} | {row['failure_rate']['mean']:.3f} | {row['unconverged_fraction']['mean']:.3f} |")
    lines += ["", "Equal failure rates do not establish absence of selection bias. Failed and unconverged cases remain distinct outcomes.",
              "These observations do not establish improved generation quality or calibrated Boltzmann sampling.", "",
              "Full per-seed results and input hashes are in evidence.json.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = audit(args.root.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "evidence.json").write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    report = render(data)
    (args.out / "evidence.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
