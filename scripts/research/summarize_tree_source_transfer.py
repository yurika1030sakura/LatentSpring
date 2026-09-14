#!/usr/bin/env python3
"""Summarize all additional-composition results without source selection by seed."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    assert audit['complete'] and audit['source_and_structural_outputs_replayed'] == 6144
    methods = ['gaussian', 'fixed', 'harmonic_tree']
    records = [dict(seed=study['seed_index'], method=method, **study['methods'][method])
               for study in audit['studies'] for method in methods]
    output = args.project/'research/evidence/tree_source_transfer_results_v1.csv'
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for row, study in enumerate(audit['studies']):
        ax = axes[row, 0]
        values = [study['methods'][method]['graph_supported'] for method in methods]
        bars = ax.bar(range(3), values, color=['#4877A8', '#D28A30', '#8C70A9'], width=.65)
        ax.bar_label(bars, padding=3)
        ax.set_xticks(range(3), ['Gaussian', 'Shell tree', 'Harmonic tree'])
        ax.set_ylim(0, max(300, 1.2*max(values)))
        ax.set_ylabel('Graph-check passes / 1,024')
        ax.set_title(f'Continuation {row+1}: all 32 compositions')
        ax = axes[row, 1]
        for position, comparison in enumerate(['fixed minus gaussian', 'harmonic_tree minus gaussian', 'fixed minus harmonic_tree']):
            values = study['comparisons'][comparison]['graph']
            mean = values['mean']*100
            for offset, key, color in [(-.09, 'within_composition_paired95', '#4877A8'),
                                       (.09, 'descriptive_stratified_composition95', '#C26F39')]:
                low, high = np.array(values[key])*100
                ax.errorbar(mean, position+offset, xerr=np.array([[mean-low], [high-mean]]), fmt='o', color=color, capsize=3,
                    label=('Draw bootstrap' if offset < 0 else 'Composition bootstrap') if position == 0 else None)
        ax.axvline(0, color='black', lw=.8)
        ax.set_yticks(range(3), ['Shell - Gaussian', 'Harmonic - Gaussian', 'Shell - Harmonic'])
        ax.invert_yaxis()
        ax.set_xlabel('Graph pass-rate difference (percentage points)')
        ax.set_title('95% intervals\nComposition resampling is descriptive', fontsize=10)
    fig.suptitle('Frozen source models on prospectively selected additional compositions', fontsize=13)
    fig.text(.5, .012, 'All 6,144 outputs retained. No new model fitting or physical queries.\n'
             'Additional development panel; composition-disjoint warm pretraining is not certified. Sources are not chosen separately by seed.', ha='center', fontsize=9)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=2, loc='lower center', bbox_to_anchor=(.5, .065))
    fig.tight_layout(rect=(0, .12, 1, .95))
    directory = args.project/'research/figures/tree_source_transfer_v1'
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ['pdf', 'png']:
        fig.savefig(directory/f'tree_source_transfer_v1.{extension}', dpi=180)
    plt.close(fig)
    lines = [r'\begin{table}[t]', r'\centering',
        r'\caption{Additional32-composition panel: graph-check passes /1,024. Source checkpoints were fixed before generation; no new fitting is used.}',
        r'\begin{tabular}{lrrr}', r'\toprule',
        r'Continuation & Gaussian & Shell tree & Harmonic tree\\', r'\midrule']
    for study in audit['studies']:
        lines.append(str(study['seed_index']+1)+' & '+' & '.join(str(study['methods'][m]['graph_supported']) for m in methods)+r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\label{tab:source-transfer}', r'\end{table}']
    for study in audit['studies']:
        metric = study['comparisons']['fixed minus gaussian']['graph']
        low, high = metric['within_composition_paired95']
        clow, chigh = metric['descriptive_stratified_composition95']
        lines.append(f"For continuation{study['seed_index']+1}, shell minus Gaussian is ${100*metric['mean']:+.2f}$ percentage points, "
            f"with paired draw interval $[{100*low:.2f},{100*high:.2f}]$ and descriptive stratified composition interval $[{100*clow:.2f},{100*chigh:.2f}]$.")
    lines.append('The panel was selected without output or energy ranking. Composition overlap with the full warm-pretraining corpus is not certified; these are additional development results, not a final unseen-composition benchmark.')
    (args.project/'paper/sections/tree_source_transfer_results.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps({f's{s["seed_index"]}': s['methods'] for s in audit['studies']}))


if __name__ == '__main__':
    main()
