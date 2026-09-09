"""Plot archived solver sensitivity and strain; no fitted or synthetic results."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.evidence.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    colors = {'fm': '#4063a5', 'value': '#d6604d', 'scrambled': '#6a9f58'}
    labels = {'fm': 'FM', 'value': 'Energy values', 'scrambled': 'Scrambled energies'}
    steps = sorted(map(int, data['resolution']))
    for arm in colors:
        rows = [data['resolution'][str(s)]['arms'][arm] for s in steps]
        axes[0].errorbar(steps, [r['mean'] for r in rows], yerr=[r['sem'] for r in rows],
                         label=labels[arm], color=colors[arm], marker='o', capsize=3)
    axes[0].set(xlabel='Legacy integration steps', ylabel='Within-parent Pearson r',
                title='Matched checkpoints and geometries', xticks=steps)
    axes[0].legend(frameon=False, fontsize=9)
    for i, arm in enumerate(colors):
        prefixes = {'fm': 'a1_', 'value': 'a3_', 'scrambled': 'a6_'}
        values = [v['median_strain_kcal_per_atom'] for tag, v in data['generation'].items()
                  if tag.startswith(prefixes[arm])]
        row = data['generation_arms'][arm]['median_strain_kcal_per_atom']
        axes[1].errorbar(i, row['mean'], yerr=row['sem'], fmt='s', color=colors[arm], capsize=6)
        offsets = [(j-(len(values)-1)/2)*0.055 for j in range(len(values))]
        axes[1].scatter([i+o for o in offsets], values, color=colors[arm], alpha=.65, s=25)
    axes[1].set(xticks=range(3), xticklabels=['FM', 'Value', 'Scrambled'],
                ylabel='Per-seed median decrease (kcal/mol/atom)', title='Generated-structure energy decrease')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
    fig.supxlabel('Completed-seed means ± SEM. Legacy readout is not a validated likelihood; finite relaxations include unconverged cases.', fontsize=8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'pdf']:
        fig.savefig(args.out.with_suffix('.'+suffix), dpi=180)


if __name__ == '__main__':
    main()
