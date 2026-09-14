#!/usr/bin/env python3
"""Summarize the prospective reference-qualified monomer test without dropping arms."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    assert audit['complete'] and audit['sources_and_structural_outcomes_replayed'] == 10752
    order = ['gaussian', 'fixed', 'harmonic_tree', 'node', 'pair']
    labels = {'gaussian': 'Gaussian', 'fixed': 'Shell', 'harmonic_tree': 'Harmonic', 'node': 'Node', 'pair': 'Pair'}
    colors = {'gaussian': '#4877A8', 'fixed': '#D28A30', 'harmonic_tree': '#8C70A9', 'node': '#328866', 'pair': '#B26076'}
    rows = [dict(seed=study['seed_index'], method=method, **study['methods'][method]) for study in audit['studies']
            for method in order if method in study['methods']]
    output = args.project/'research/evidence/monomer_results_v1.csv'
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for row, study in enumerate(audit['studies']):
        methods = [m for m in order if m in study['methods']]
        for col, (key, title) in enumerate([('graph_supported', 'Graph check passed'), ('geometrically_supported', 'Geometry check passed')]):
            ax = axes[row, col]
            values = [study['methods'][m][key] for m in methods]
            bars = ax.bar(range(len(methods)), values, color=[colors[m] for m in methods], width=.65)
            ax.bar_label(bars, padding=3)
            ax.set_xticks(range(len(methods)), [labels[m] for m in methods])
            ax.set_ylim(0, max(1100, 1.15*max(values)))
            ax.set_ylabel('Count / 1,344')
            ax.set_title(f'Continuation {row+1}: {title}')
    figure.suptitle('Reference-qualified neutral organic monomers: all frozen models retained', fontsize=13)
    figure.text(.5, .012, '21 compositions; 64 outputs per composition and model. All 21 original references passed before generation.\n'
                'Node/pair source learning is evaluated only in the first continuation. No new fitting or oracle queries.', ha='center', fontsize=9)
    figure.tight_layout(rect=(0, .075, 1, .95))
    directory = args.project/'research/figures/monomer_v1'
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ['pdf', 'png']:
        figure.savefig(directory/f'monomer_results_v1.{extension}', dpi=180)
    plt.close(figure)
    lines = [r'\begin{table}[t]', r'\centering',
        r'\caption{Prospective monomer evaluation: graph-check passes /1,344. Node/pair source models exist only for the first matched continuation. No model is fitted on the evaluation references.}',
        r'\begin{tabular}{lrrrrr}', r'\toprule',
        r'Continuation & Gaussian & Shell & Harmonic & Node & Pair\\', r'\midrule']
    for study in audit['studies']:
        values = [str(study['methods'][m]['graph_supported']) if m in study['methods'] else '---' for m in order]
        lines.append(str(study['seed_index']+1)+' & '+' & '.join(values)+r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\label{tab:monomer}', r'\end{table}']
    for study in audit['studies']:
        comparisons = ['fixed minus gaussian']
        if study['seed_index'] == 0:
            comparisons += ['node minus fixed', 'pair minus fixed']
        for key in comparisons:
            result = study['comparisons'][key]['graph']
            low, high = result['within_composition_paired95']
            clow, chigh = result['descriptive_size_stratified95']
            lines.append(f"In continuation{study['seed_index']+1}, {key.replace('fixed','shell')} is ${100*result['mean']:+.2f}$ percentage points, "
                f"with paired interval $[{100*low:.2f},{100*high:.2f}]$ and descriptive size-stratified composition interval $[{100*clow:.2f},{100*chigh:.2f}]$.")
    lines.append('Intervals are marginal and not multiplicity-adjusted. Any node/pair result is one-continuation evidence and does not establish a replicated learned-source increment.')
    (args.project/'paper/sections/monomer_results.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps({f's{s["seed_index"]}': s['methods'] for s in audit['studies']}))


if __name__ == '__main__':
    main()
