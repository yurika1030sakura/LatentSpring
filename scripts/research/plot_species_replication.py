#!/usr/bin/env python3
"""Plot the paired refinement signal alongside its retained weight degeneracy."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--evidence', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
args = p.parse_args()
report = json.loads(args.evidence.read_text())
if not report['complete'] or not report['replay_verified']:
    raise ValueError('Require completed replay audit')
if args.out.exists():
    raise FileExistsError(args.out)
args.out.mkdir(parents=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), gridspec_kw={'width_ratios': [1.2, 1]})
colors = {'linear': '#757575', 'nonlinear': '#1677a5'}
pairs = list(report['pairs'].values())
for arm, offset in [('linear', -.09), ('nonlinear', .09)]:
    values = [pair['arms'][arm]['confirmation_delta_kl'] for pair in pairs]
    axes[0].errorbar(np.arange(2)+offset, [v['mean'] for v in values], yerr=[v['sem'] for v in values],
        fmt='o', capsize=4, color=colors[arm], label='Typed linear' if arm == 'linear' else 'Neural convex')
axes[0].axhline(0, color='black', linewidth=.8)
axes[0].set(xticks=[0, 1], xticklabels=['Training pair 1', 'Training pair 2'],
    ylabel='Change in endpoint KL (nat)', title='Small relative-KL improvement', ylim=(-.23, .025))
axes[0].legend(frameon=False, loc='upper right')
labels = ['Base', 'Linear 1', 'Convex 1', 'Linear 2', 'Convex 2']
ess = [report['fresh_base_weights']['ess']]
for pair in pairs:
    ess += [pair['arms'][arm]['confirmation_weights']['ess'] for arm in ['linear', 'nonlinear']]
axes[1].scatter(range(5), ess, color=['#333333', colors['linear'], colors['nonlinear'], colors['linear'], colors['nonlinear']])
axes[1].axhline(2048, color='#b05040', linestyle='--', linewidth=1)
axes[1].text(2, 1300, '2,048 evaluated paths', ha='center', color='#b05040')
axes[1].set(yscale='log', ylim=(.7, 3500), ylabel='Effective sample size',
    xticks=range(5), xticklabels=labels, title='Weight degeneracy remains')
axes[1].set_yticks([1, 10, 100, 1000], labels=['1', '10', '100', '1,000'])
axes[1].tick_params(axis='x', rotation=30)
fig.text(.5, .02, 'One molecular condition; shared 2,048-row evaluation. Error bars: row SEM, conditional on each trained model.',
    ha='center', fontsize=9)
fig.tight_layout(rect=(0, .07, 1, 1))
for extension in ['pdf', 'svg', 'png']:
    fig.savefig(args.out/f'replication.{extension}', dpi=180)
