#!/usr/bin/env python3
"""Export the fixed-budget calibration comparison, including simple controls."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    data = json.loads(a.summary.read_text())
    assert data['complete'] and data['every_weight_and_prefix_replayed']
    a.out.mkdir(parents=True, exist_ok=True)
    labels = {'independent': 'Independent', 'identity': 'Shared latent',
              'constant': 'Constant rotation', 'gaussian': 'Gaussian coupling',
              'nonlinear': 'Neural twist', 'binned': 'Eight radial bins'}
    colors = ['#777777', '#b59b00', '#56a0c7', '#005a8d', '#a83665', '#278656']
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.6), gridspec_kw={'height_ratios': [1.3, 1]})
    axes[0, 1].sharey(axes[0, 0])
    axes[1, 1].sharex(axes[1, 0])
    for col, scenario in enumerate(['constant', 'nonlinear']):
        ax = axes[0, col]
        rows = sorted([r for r in data['rows'] if r['scenario'] == scenario], key=lambda r: r['pairs'])
        for (method, label), color in zip(labels.items(), colors):
            ax.plot([2*r['pairs'] for r in rows], [100*r['methods'][method]['mass_RMSE'] for r in rows],
                    label=label, color=color, marker='o', markersize=3,
                    linestyle='-' if method in ['nonlinear', 'binned'] else '--', linewidth=1.5)
        ax.set_title({'constant': 'Constant target twist', 'nonlinear': 'Radius-dependent target twist'}[scenario], fontsize=10)
        ax.set_xlabel('Total target calls, including learning')
        ax.set_xticks([128, 256, 384, 512])
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        lower = axes[1, col]
        controls = ['independent', 'identity', 'constant', 'gaussian', 'binned']
        for i, method in enumerate(controls):
            comparison = rows[-1]['comparisons']['nonlinear minus ' + method]
            mean = comparison['paired_MSE_difference'] * 1e4
            lo, hi = [v * 1e4 for v in comparison['paired_history_bootstrap95']]
            lower.plot([lo, hi], [i, i], color='#555555', linewidth=1.5)
            lower.plot(mean, i, 'o', color='#a83665', markersize=4)
        lower.axvline(0, color='#888888', linestyle='--', linewidth=1)
        lower.set_yticks(range(len(controls)), [labels[m] for m in controls], fontsize=8)
        lower.invert_yaxis()
        lower.set_xlabel('Neural minus control MSE (x 10,000)\nNegative favors neural; paired 95% bootstrap', fontsize=8)
        lower.spines[['top', 'right']].set_visible(False)
        lower.grid(axis='x', alpha=.2)
    axes[0, 0].set_ylabel('Mass RMSE (percentage points)')
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(.5, .01))
    fig.suptitle('Known-target diagnostic: 32 adaptive histories per case', fontsize=11)
    fig.tight_layout(rect=(0, .11, 1, .96), h_pad=2)
    for suffix in ['pdf', 'png']:
        path = a.out / ('online_latent_mass_v2.' + suffix)
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
