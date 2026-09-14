#!/usr/bin/env python3
"""Write the two-seed moment-control table, figure and manuscript result block."""
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
    assert audit['complete'] and audit['source_and_structural_outputs_replayed'] == 4096
    names = {'gaussian': 'Unit Gaussian', 'covariance_gaussian': 'Covariance Gaussian',
             'harmonic_tree': 'Harmonic tree', 'fixed': 'Shell tree'}
    methods = list(names)
    colors = ['#4877A8', '#619CA7', '#8C70A9', '#D28A30']
    output = args.project/'research/evidence/tree_moment_results_v1.csv'
    records = []
    for study in audit['studies']:
        for method in methods:
            records.append(dict(seed=study['seed_index'], method=method, **study['methods'][method]))
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
    for row, study in enumerate(audit['studies']):
        for col, (key, label, limit) in enumerate([
                ('graph_supported', 'Graph check passed', 170),
                ('geometrically_supported', 'Geometry check passed', 400),
                ('disconnected', 'Disconnected outputs', 310)]):
            ax = axes[row, col]
            bars = ax.bar(range(4), [study['methods'][method][key] for method in methods], color=colors, width=.65)
            ax.bar_label(bars, padding=3)
            ax.set_xticks(range(4), ['Unit\nGaussian', 'Covariance\nGaussian', 'Harmonic\ntree', 'Shell\ntree'])
            ax.set_title(f'Continuation {row+1}: {label}', fontsize=10)
            ax.set_ylim(0, limit)
            ax.set_ylabel('Count / 512')
    fig.suptitle('Spatial source controls with matched second moments', fontsize=14)
    fig.text(.5, .012, 'Each continuation: matched 3,000 training examples; 64 fresh draws on each of 8 fixed development conditions.\n'
             'Harmonic and shell trees match covariance analytically. The single Gaussian uses a fixed covariance estimate.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .95))
    directory = args.project/'research/figures/tree_moment_v1'
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ['pdf', 'png']:
        fig.savefig(directory/f'tree_moment_results_v1.{extension}', dpi=180)
    plt.close(fig)
    lines = [r'\begin{table}[t]', r'\centering',
        r'\caption{Fresh moment-control outputs: graph check passed /512. All methods share training data and update budgets within a continuation.}',
        r'\begin{tabular}{lrrrr}', r'\toprule',
        r'Continuation & Unit Gaussian & Covariance Gaussian & Harmonic tree & Shell tree\\', r'\midrule']
    for study in audit['studies']:
        lines.append(str(study['seed_index']+1) + ' & ' + ' & '.join(str(study['methods'][m]['graph_supported']) for m in methods) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\label{tab:moment}', r'\end{table}',
              r'Table~\ref{tab:moment} reports the complete fresh comparison.']
    for comparison, name in [('fixed minus covariance_gaussian', 'shell minus covariance Gaussian'),
                             ('fixed minus harmonic_tree', 'shell minus harmonic tree')]:
        values = []
        for study in audit['studies']:
            metric = study['comparisons'][comparison]['graph']
            low, high = metric['conditional_paired_bootstrap95']
            values.append(f"${100*metric['mean']:+.2f}$ percentage points $[{100*low:.2f},{100*high:.2f}]$")
        lines.append('The ' + name + ' graph pass-rate differences are ' + ' and '.join(values) + ', respectively.')
    lines.append('These intervals condition on fitted models and the same development compositions. The controls do not test an additional learned-affinity increment.')
    (args.project/'paper/sections/tree_moment_results.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps({f's{study["seed_index"]}': study['methods'] for study in audit['studies']}))


if __name__ == '__main__':
    main()
