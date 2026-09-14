#!/usr/bin/env python3
"""Create the context ablation table, figure and auditable manuscript block."""
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
    assert audit['complete'] and audit['source_and_structural_outputs_replayed'] == 3072
    methods = ['fixed', 'tree_actual', 'tree_sham']
    labels = ['No tree\ncontext', 'Actual source\ntree', 'Independent\ntree']
    colors = ['#D28A30', '#328866', '#8B6CAC']
    records = [dict(seed=study['seed_index'], method=method, **study['methods'][method])
               for study in audit['studies'] for method in methods]
    output = args.project/'research/evidence/tree_context_results_v1.csv'
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
    for row, study in enumerate(audit['studies']):
        for col, (key, title, limit) in enumerate([
                ('graph_supported', 'Graph check passed', 175),
                ('geometrically_supported', 'Geometry check passed', 400),
                ('disconnected', 'Disconnected outputs', 310)]):
            ax = axes[row, col]
            values = [study['methods'][method][key] for method in methods]
            bars = ax.bar(range(3), values, color=colors, width=.65)
            ax.bar_label(bars, padding=3)
            ax.set_xticks(range(3), labels)
            ax.set_title(f'Continuation {row+1}: {title}', fontsize=10)
            ax.set_ylim(0, max(limit, 1.2*max(values)))
            ax.set_ylabel('Count / 512')
    fig.suptitle('Does the flow benefit from knowing its sampled source tree?', fontsize=14)
    fig.text(.5, .012, 'Same shell prior and paired source coordinates; actual/independent adapters have identical parameter counts and initialization.\n'
             'Two continuations, 3,000 matched training examples each, eight fixed development compositions. Every output retained.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .95))
    directory = args.project/'research/figures/tree_context_v1'
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ['pdf', 'png']:
        fig.savefig(directory/f'tree_context_results_v1.{extension}', dpi=180)
    plt.close(fig)
    lines = [r'\begin{table}[t]', r'\centering',
        r'\caption{Latent-tree context ablation: graph-check passes /512. The actual and independent-tree adapters have the same parameter count, initialization and learning rates.}',
        r'\begin{tabular}{lrrr}', r'\toprule',
        r'Continuation & No tree context & Actual source tree & Independent tree\\', r'\midrule']
    for study in audit['studies']:
        lines.append(str(study['seed_index']+1) + ' & ' + ' & '.join(str(study['methods'][m]['graph_supported']) for m in methods) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\label{tab:context}', r'\end{table}',
              r'Table~\ref{tab:context} reports all3,072 fresh outputs.']
    for comparison, name in [('tree_actual minus tree_sham', 'actual minus independent-tree'),
                             ('tree_actual minus fixed', 'actual minus no-context')]:
        values = []
        for study in audit['studies']:
            metric = study['comparisons'][comparison]['graph']
            low, high = metric['conditional_paired_bootstrap95']
            values.append(f"${100*metric['mean']:+.2f}$ percentage points $[{100*low:.2f},{100*high:.2f}]$")
        lines.append('The '+name+' graph pass-rate differences are '+' and '.join(values)+', respectively.')
    lines.append('Intervals condition on the fitted models and the same eight development compositions. These are marginal intervals without multiplicity adjustment.')
    (args.project/'paper/sections/tree_context_results.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps({f's{study["seed_index"]}': study['methods'] for study in audit['studies']}))


if __name__ == '__main__':
    main()
