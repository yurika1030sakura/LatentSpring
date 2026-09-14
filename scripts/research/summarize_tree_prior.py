#!/usr/bin/env python3
"""Build tables, reference registry and figures from completed tree-prior audits."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.project
    paths = {
        'unrestricted': 'research/evidence/tree_prior_study_v1.json',
        'connected_first': 'runs/tree_prior_connected_audit_v1/audit.json',
        'connected_confirmation': 'research/evidence/tree_prior_confirmation_audit_v2.json',
        'energy': 'research/evidence/tree_prior_energy_audit_v2.json',
    }
    audits = {key: read(root / path) for key, path in paths.items()}
    assert all(audit['complete'] for audit in audits.values())
    stages = {key: {field: value for field, value in audit.items() if field != 'rows'}
              for key, audit in audits.items()}
    table = []
    for stage in ['unrestricted', 'connected_first', 'connected_confirmation']:
        for method, values in audits[stage]['methods'].items():
            assert values['attempted'] == 512
            table.append(dict(stage=stage, method=method, **{key: values[key] for key in
                ['attempted', 'graph_supported', 'geometrically_supported', 'disconnected', 'overlap', 'validator_errors']}))
    evidence = root / 'research/evidence'
    with (evidence / 'tree_prior_results_v1.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(table)
    summary = dict(
        complete=True,
        sources={key: dict(path=path, sha256=digest(root / path)) for key, path in paths.items()},
        audit_implementations={name: digest(root / 'scripts/research' / name) for name in
                               ['audit_tree_confirmation.py', 'audit_tree_energy.py']},
        stages=stages,
        structural_table=table,
        generated_outputs=6144,
        raw_energy_queries=audits['energy']['producer_raw_oracle_calls'],
        decision={
            'retain_framework': True,
            'fragmentation_reduction_repeated': True,
            'graph_support_advantage_reliably_replicated': False,
            'extra_learned_affinity_advantage_over_fixed_established': False,
            'node_energy_signal_vs_gaussian': 'Negative conditional interval in the first connected training seed only; interval versus fixed spans zero.',
            'scale_current_pair_network': False,
            'scientific_submission_ready': False,
            'iclr_goal_achieved': False,
        },
        uncertainty_scope='Paired bootstrap within each of eight fixed development compositions, conditional on fitted models. Intervals are marginal, not multiplicity-adjusted; no population-of-compositions or training-seed uncertainty claim.',
        inference_scope='Composition, original total charge and spin are supplied. This is conditional 3D generation, not a demonstrated joint generator of composition and electronic state.',
        target_scope='Connected-data results target the same geometry-selected OMol25 training subset for all methods. They do not establish improvement on unrestricted OMol25.',
    )
    write(evidence / 'tree_prior_results_v1.json', summary)

    locations = {
        'unrestricted': ('runs/tree_prior_fm_fixed_v1/study', 'runs/tree_prior_fm_learned_v1/study'),
        'connected_first': ('runs/tree_prior_fm_connected_fixed_v1/study', 'runs/tree_prior_fm_connected_learned_v1/study'),
    }
    references = {}
    for stage in ['unrestricted', 'connected_first', 'connected_confirmation']:
        refs = {}
        for method, source in audits[stage]['sources'].items():
            if method == 'warm':
                checkpoint = 'runs/orbit_pairing_typed_v1/training/typed_rotation/last.ckpt'
            elif stage == 'connected_confirmation':
                task = 0 if method == 'gaussian' else 1
                checkpoint = f'runs/tree_prior_confirmation_v1/s{task}/study/{method}/last.ckpt'
            else:
                group = locations[stage][0 if method in ['gaussian', 'fixed'] else 1]
                checkpoint = f'{group}/{method}/last.ckpt'
            assert (root / checkpoint).is_file()
            refs[method] = dict(checkpoint=checkpoint, checkpoint_sha256=source['checkpoint_sha256'],
                source_prior_kind='gaussian' if method == 'warm' else method,
                prior_saved_in_checkpoint=method in ['fixed', 'node', 'pair'],
                observed_structural_readout=audits[stage]['methods'][method])
        references[stage] = refs
    old_registry = 'research/evidence/generator_reference_registry_v1.json'
    write(evidence / 'generator_reference_registry_v2.json', dict(
        previous_registry=dict(path=old_registry, sha256=digest(root / old_registry)),
        references=references,
        source_audits=summary['sources'],
        sampling_law=dict(midpoint_steps=64, terminal_time=1., terminal_COM_noise_std_A=.025,
                          model_input_kT_eV=1., absolute_density_qualified=False),
        scope='All checkpoints are development references, not a final winner or reserved-set selection. Tree models need their saved source prior; implicit Gaussian draws and Gaussian density are incompatible.',
    ))

    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'warm': '#888888', 'gaussian': '#4777A8', 'fixed': '#D28A30',
              'node': '#338B69', 'pair': '#8865A7'}
    labels = {'warm': 'Warm', 'gaussian': 'Gaussian', 'fixed': 'Fixed tree',
              'node': 'Node tree', 'pair': 'Pair tree'}
    figures = root / 'research/figures/tree_prior_v1'
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    columns = [('graph_supported', 'Graph check passed', 160),
               ('geometrically_supported', 'Geometry check passed', 400),
               ('disconnected', 'Disconnected outputs', 310)]
    for row, stage in enumerate(['connected_first', 'connected_confirmation']):
        methods = audits[stage]['methods']
        for col, (key, title, limit) in enumerate(columns):
            ax = axes[row, col]
            values = [value[key] for value in methods.values()]
            bars = ax.bar(list(range(len(methods))), values, color=[colors[m] for m in methods], width=.65)
            ax.bar_label(bars, padding=3)
            ax.set_xticks(list(range(len(methods))), [labels[m] for m in methods], rotation=24, ha='right')
            ax.set_ylim(0, limit)
            ax.set_ylabel('Count / 512')
            ax.set_title(('First continuation\n' if row == 0 else 'Independent continuation\n') + title, fontsize=10)
    fig.suptitle('A spatial tree prior reduces fragmentation; graph acceptance gain is not stable', fontsize=13)
    fig.text(.5, .012, 'Fixed minus Gaussian graph pass rate: first +5.08 pp [1.17, 8.79]; confirmation +0.39 pp [-3.52, 4.10].\n'
             '64 draws per condition; same 8 development compositions. Marginal 95% intervals condition on fitted models.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .95))
    for suffix in ['pdf', 'png']:
        fig.savefig(figures / f'tree_prior_structure_v1.{suffix}', dpi=180)
    plt.close(fig)

    energy = audits['energy']['comparisons']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={'width_ratios': [1.6, 1]})
    for offset, method in zip([-.18, 0., .18], ['fixed', 'node', 'pair']):
        comparison = energy[f'{method} minus gaussian']
        differences = [row['mean_energy_difference_all_eV'] for row in comparison['per_condition']]
        axes[0].scatter(np.arange(8) + offset, differences, label=labels[method], color=colors[method], s=38)
    axes[0].axhline(0, color='black', lw=.8)
    axes[0].set_xticks(range(8), ['0', '1', '2 (Al)', '3 (Re)', '4', '5', '6', '7 (Pt)'])
    axes[0].set_xlabel('Development condition')
    axes[0].set_ylabel('Mean E+ difference from Gaussian (eV)')
    axes[0].set_title('Every condition retained; lower is better')
    axes[0].legend(frameon=False)
    for i, method in enumerate(['fixed', 'node', 'pair']):
        comparison = energy[f'{method} minus gaussian']
        mean = comparison['mean_energy_difference_eV']
        low, high = comparison['energy_conditional_paired_bootstrap95']
        axes[1].errorbar(mean, i, xerr=np.array([[mean-low], [high-mean]]), fmt='o', color=colors[method], capsize=4)
    axes[1].axvline(0, color='black', lw=.8)
    axes[1].set_yticks(range(3), [labels[m] for m in ['fixed', 'node', 'pair']])
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Equal-condition mean difference (eV)')
    axes[1].set_title('Conditional paired 95% intervals')
    fig.suptitle('Energy readout: a node-network signal versus Gaussian, unreplicated', fontsize=13)
    fig.text(.5, .018, 'All 2,560 outputs scored; 5,120 eSEN calls including inversion. E+ = [E(x) + E(-x)] / 2.\n'
             'First connected training seed only. Node minus fixed: -0.229 eV [-0.739, 0.269]. No Boltzmann-law evidence.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .105, 1, .94))
    for suffix in ['pdf', 'png']:
        fig.savefig(figures / f'tree_prior_energy_v1.{suffix}', dpi=180)
    plt.close(fig)
    print(json.dumps({'tables': len(table), 'generated_outputs': 6144, 'raw_energy_queries': summary['raw_energy_queries'], 'figures': str(figures)}))


if __name__ == '__main__':
    main()
