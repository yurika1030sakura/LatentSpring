#!/usr/bin/env python3
"""Compare completed solver panels without dropping unstable parents."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    sources, results = {}, []

    def read(relative):
        path = args.runs / relative
        data = json.loads(path.read_text())
        if not data['complete']:
            raise ValueError(f'Incomplete source: {relative}')
        sources[relative] = {'path': str(path.resolve()),
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        return data

    pairs = [
        ('displacement_development_rho0_v1/density/panel.json', 'rk4_density_rho0_v1/panel.json'),
        ('displacement_development_rho01_v2/density/panel.json', 'rk4_density_rho01_v1/panel.json')]
    for midpoint_path, rk4_path in pairs:
        midpoint, rk4 = read(midpoint_path), read(rk4_path)
        for key in ['checkpoint_sha256', 'config_sha256', 'shard_sha256', 'terminal_time',
                    'replicas', 'perturbation_indices', 'geometry_softening',
                    'position_parameterization', 'probe_policy']:
            if midpoint[key] != rk4[key]:
                raise ValueError(f'Incompatible {key} between source panels')
        if [r['parent_id'] for r in midpoint['rows']] != [r['parent_id'] for r in rk4['rows']]:
            raise ValueError('Different parents or probe-seed order')
        rows = []
        for left, right in zip(midpoint['rows'], rk4['rows']):
            if left['geometry_sha256'] != right['geometry_sha256']:
                raise ValueError('Different geometry')
            a, b = left['resolutions'][-1], right['resolutions'][-1]
            qa, qb = np.asarray(a['centered_log_q']), np.asarray(b['centered_log_q'])
            # Panel arrays are replicas by sibling geometry.
            if qa.shape != qb.shape or qa.shape[0] != midpoint['replicas']:
                raise ValueError('Unexpected replica layout')
            rows.append({'parent_id': left['parent_id'], 'n_atoms': left['n_atoms'],
                         'midpoint_steps': a['steps'], 'rk4_steps': b['steps'],
                         'midpoint_mean_drift_nat': a['max_mean_centered_change_from_previous'],
                         'rk4_mean_drift_nat': b['max_mean_centered_change_from_previous'],
                         'midpoint_replica_drift_nat': a['max_centered_change_from_previous'],
                         'rk4_replica_drift_nat': b['max_centered_change_from_previous'],
                         'cross_solver_mean_difference_nat': float(np.abs(qa.mean(0)-qb.mean(0)).max()),
                         'cross_solver_replica_difference_nat': float(np.abs(qa-qb).max()),
                         'midpoint_seconds': a['seconds'], 'rk4_seconds': b['seconds']})
        results.append({'geometry_softening': midpoint['geometry_softening'], 'rows': rows,
                        'midpoint_failures_above_0_1_nat': sum(r['midpoint_mean_drift_nat'] > .1 for r in rows),
                        'rk4_failures_above_0_1_nat': sum(r['rk4_mean_drift_nat'] > .1 for r in rows),
                        'cross_solver_failures_above_0_1_nat': sum(r['cross_solver_mean_difference_nat'] > .1 for r in rows)})
    report = {'sources': sources, 'results': results,
              'scope': 'Fixed-checkpoint numerical development; mean drift is a necessary screen, not density certification',
              'probe_policy': 'Eight replicas, probes fixed over time and across solver resolutions',
              'cost': 'Midpoint 256 and RK4 64 each use 256 trace stages; field evaluations are 512 and 256, respectively; neither wall time nor work is equal per step',
              'limitations': ['All eight predeclared parents are retained',
                              'Averaging trace replicas may conceal integration errors',
                              'An independent adaptive reference and stricter checks are still required']}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'solver_comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True, layout='constrained')
    colors = ['#3D5A80', '#B05A36', '#527D57']
    for ax, group in zip(axes, results):
        rows = group['rows']; indices = np.arange(len(rows))
        for offset, key, label, color in zip([-.24, 0, .24],
                ['midpoint_mean_drift_nat', 'rk4_mean_drift_nat', 'cross_solver_mean_difference_nat'],
                ['Midpoint 128 to 256', 'RK4 32 to 64', 'Midpoint 256 vs RK4 64'], colors):
            ax.bar(indices+offset, [r[key] for r in rows], width=.23, label=label, color=color)
        ax.axhline(.1, color='black', linestyle='--', linewidth=1)
        ax.set(yscale='log', ylim=(.001, 100), xticks=indices,
               xticklabels=[str(r['parent_id']) for r in rows], xlabel='Development parent',
               title=f"Geometry softening = {group['geometry_softening']} Å")
        ax.tick_params(axis='x', labelsize=8, rotation=45)
    axes[0].set_ylabel('Maximum centered mean change (nats)')
    axes[0].legend(fontsize=7, loc='upper left')
    fig.savefig(args.out / 'solver_comparison.pdf')
    fig.savefig(args.out / 'solver_comparison.png', dpi=180)
    print(json.dumps([{k: v for k, v in group.items() if k != 'rows'} for group in results], indent=2))


if __name__ == '__main__':
    main()
