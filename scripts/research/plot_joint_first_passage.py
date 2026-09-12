#!/usr/bin/env python3
"""Plot audited first-passage coverage against actual preparation-inclusive cost."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ['pilot', 'continued', 'out']:
        parser.add_argument('--'+arg, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    ap = root/'research/evidence/joint_chemical_audit_v1.json'
    ac = root/'research/evidence/joint_chemical_full_cost_audit_v1.json'
    audits = [json.loads(p.read_text()) for p in [ap, ac]]
    assert all(a['complete'] for a in audits)
    checked = {(row['method'], row['replica'], bool(row.get('immutable_prefix_checked'))): row for audit in audits for row in audit['rows']}
    methods = ['deterministic', 'uniform', 'site', 'tensor']
    labels = ['Deterministic exchange', 'Uniform geometry', 'Coordination-site prior', 'Learned tensor']
    colors = ['#666666', '#D98C00', '#007C91', '#8B39A5']
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    inputs = []
    reference = 'CS(F)(F)(F)(F)F'
    for replica, ax in enumerate(axes):
        for method, label, color in zip(methods, labels, colors):
            continued = method != 'tensor'
            directory = (args.continued if continued else args.pilot)/f'{method}_s{replica}'
            path = directory/'results.json'
            report = json.loads(path.read_text())
            row = checked[(method, replica, continued)]
            assert report['complete'] and row['trace_sha256'] == report['trace_sha256']
            assert row['total_raw_queries'] == report['history'][-1]['total_raw_queries']
            reached = [False]*4
            costs, coverage = [], []
            for h in report['history']:
                reached = [a or b == reference for a, b in zip(reached, h['smiles'])]
                costs.append(h['total_raw_queries'])
                coverage.append(sum(reached))
            assert sum(reached) == sum(hit is not None for hit in row['reference_connectivity_first_hit_step'])
            ax.step(costs, coverage, where='post', color=color, label=label,
                    linewidth=1.8 if method == 'tensor' else 1.25)
            ax.plot(costs[-1], coverage[-1], 'o', color=color, markersize=3)
            inputs.append(dict(method=method, replica=replica, results_sha256=sha(path),
                               final_coverage=sum(reached), final_total_raw_queries=costs[-1]))
        ax.set_title(f'Replica {replica}')
        ax.set_xlim(0, 12600)
        ax.set_ylim(-.12, 4.18)
        ax.set_yticks(range(5))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(3, 3))
        ax.set_xlabel('Total raw queries, including preparation')
        ax.grid(axis='y', alpha=.18)
    axes[0].set_ylabel('Starts ever reaching\nreference connectivity (of 4)')
    handles, legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend, loc='upper center', ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .84))
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'manifest.json').exists():
        raise FileExistsError(args.out/'manifest.json')
    fig.savefig(args.out/'first_passage.pdf', bbox_inches='tight', metadata={'CreationDate': None})
    fig.savefig(args.out/'first_passage.png', dpi=180, bbox_inches='tight')
    manifest = dict(complete=True, pilot_audit_sha256=sha(ap), continuation_audit_sha256=sha(ac), inputs=inputs,
        pdf_sha256=sha(args.out/'first_passage.pdf'), png_sha256=sha(args.out/'first_passage.png'),
        plot_script_sha256=sha(Path(__file__)), scientific_submission_ready=False,
        interpretation='Four repeatedly inspected development starts of one composition, two replicas; first passage is not equilibrium coverage, independence or a mixing certificate. Controls have actual reported residual budget gaps.')
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
