#!/usr/bin/env python3
"""Preserve complete frozen-score outcomes and the failed confirmation decision."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--figure-prefix', type=Path)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    names = [('raw500', 'proposal_score_5846_v1', 'screen')]
    names += [(arm+'500', f'proposal_score_{arm}_v1', 'screen') for arm in ['scaled_iid', 'grouped_iid', 'antithetic', 'mean_cv']]
    names += [('anti500 seed9101', 'proposal_score_validation_v1', 'confirmation'),
              ('anti500 seed9103', 'proposal_score_replica_v1', 'confirmation')]
    names += [(f'{arm}3000 seed{seed}', f'proposal_score_{arm}_3000_s{seed}', 'longer_training')
              for arm in ['antithetic', 'grouped'] for seed in [9101, 9103]]
    rows = []; common = None
    for label, name, stage in names:
        path = args.runs_root/name/'results.json'
        result = json.loads(path.read_text())
        if not result['complete'] or result['forward_model_updates'] or result['new_oracle_evaluations']:
            raise ValueError('Require complete frozen, oracle-free score results')
        properties = {key: result[key] for key in ['condition', 'source_checkpoint_sha256', 'training_pool_sha256', 'terminal_noise_std']}
        if common is None:
            common = properties
        elif properties != common:
            raise ValueError('Mismatched frozen proposal or training pool')
        for file, expected in result['artifacts'].items():
            if hashlib.sha256((path.parent/file).read_bytes()).hexdigest() != expected:
                raise ValueError('Source artifact changed')
        stein = result['stein']['learned']
        rows.append({'label': label, 'stage': stage, 'source': str(path.resolve()),
            'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'configuration': result['configuration'], 'updates_in_this_run': result['configuration']['steps'],
            'training_parents': result['train_parents'], 'assessment_parents': result['heldout_parents'],
            'assessment_parent_seed': result['heldout_parent_seed'], 'seconds': result['seconds'],
            'paired_dsm_difference': result['paired_dsm_difference'], 'stein': stein,
            'absolute_stein_z': {key: abs(row['mean'])/row['sem'] if row['sem'] else None for key, row in stein.items()},
            'qualification_gate': result['qualification_gate'], 'clipped_updates': result.get('clipped_updates')})
    confirmations = [row for row in rows if row['stage'] == 'confirmation']
    long_rows = [row for row in rows if row['stage'] == 'longer_training']
    result = {'complete': True, 'scope': __doc__, 'common': common, 'rows': rows,
        'both_confirmations_pass': all(row['qualification_gate']['passed'] for row in confirmations),
        'any_longer_training_pass': any(row['qualification_gate']['passed'] for row in long_rows),
        'molecular_actor_updated': False, 'molecular_sampling_advantage_established': False,
        'scientific_submission_ready': False,
        'limitations': ['One molecular condition; no distributional actor result.',
            'The8192-parent panel was an independent confirmation, then became development data for the longer-training comparison.',
            'Finite Stein moments and denoising loss are necessary checks, not a global score certificate.',
            'Initial screening passes must not be selected in place of the failed larger confirmations.',
            'All recipes use established score/variance-reduction components; AI novelty remains unestablished.']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    if args.figure_prefix is not None:
        import numpy as np
        import matplotlib.pyplot as plt
        toy = json.loads((args.runs_root/'endpoint_entropy_toy_v1/results.json').read_text())
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), gridspec_kw={'width_ratios': [1, 1.3]})
        for arm, label, color, style in [
            ('independent_reverse_joint_kl', 'Restricted reverse', '#b34a4a', '-'),
            ('exact_endpoint_score', 'Exact score', '#25835d', '-'),
            ('fitted_endpoint_score', 'Fitted linear score', '#286fba', '--'),
            ('stale_endpoint_score', 'Stale score', '#888888', ':')]:
            selected = [row for row in toy['rows'] if row['arm'] == arm]
            values = np.array([[toy['configuration']['initial_a']**2+toy['configuration']['sigma']**2]+
                [entry['a']**2+toy['configuration']['sigma']**2 for entry in row['history']] for row in selected])
            steps = [0]+[entry['step'] for entry in selected[0]['history']]
            axes[0].plot(steps, values.mean(0), label=label, color=color, linestyle=style)
            axes[0].fill_between(steps, values.min(0), values.max(0), color=color, alpha=.12)
        axes[0].axhline(.25, color='black', linewidth=.8, linestyle='--', label='Target variance')
        axes[0].set(title='A  Known-answer scalar mechanism', xlabel='Generator update', ylabel='Endpoint variance', yscale='log')
        axes[0].legend(fontsize=7, frameon=False)
        shown = confirmations+long_rows
        columns = ['dilation', 'radial_1', 'radial_2', 'radial_3']
        values = np.array([[row['absolute_stein_z'][column] for column in columns] for row in shown])
        axes[1].imshow(np.minimum(values, 15), cmap='YlOrRd', vmin=0, vmax=15, aspect='auto')
        axes[1].set_xticks(range(4), ['Scale', 'r=1', 'r=2', 'r=3'])
        axes[1].set_yticks(range(len(shown)), [row['label'] for row in shown], fontsize=8)
        axes[1].set_title('B  Molecular score checks: 8,192 parents\nRequired: every |mean|/SE <= 3', fontsize=10)
        for i in range(len(shown)):
            for j in range(4):
                axes[1].text(j, i, f'{values[i,j]:.1f}', ha='center', va='center', fontsize=8,
                    color='white' if values[i,j] > 8 else 'black')
        fig.tight_layout()
        args.figure_prefix.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ['pdf', 'png']:
            fig.savefig(str(args.figure_prefix)+'.'+suffix, dpi=180, bbox_inches='tight')
        plt.close(fig)
    print(json.dumps({key: result[key] for key in ['both_confirmations_pass', 'any_longer_training_pass', 'molecular_actor_updated']}))


if __name__ == '__main__':
    main()
