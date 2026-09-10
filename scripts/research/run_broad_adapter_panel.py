#!/usr/bin/env python3
"""Retain every prescribed entropy-refinement arm, including failed processes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from molecular_tempered_pilot import sha, write_json


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-root', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--engineering-smoke', action='store_true')
p.add_argument('--condition-index', type=int)
args = p.parse_args()
if not args.engineering_smoke and args.condition_index is None:
    raise ValueError('Production runs use one prescribed condition per allocation')
indices = range(8) if args.condition_index is None else [args.condition_index]
if any(i not in range(8) for i in indices):
    raise ValueError('Invalid condition index')
output = args.out/'panel.json'
if output.exists():
    raise FileExistsError(output)
args.out.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[2]
replicas = [0] if args.engineering_smoke else [0, 1]
report = {'complete': False, 'scope': __doc__, 'engineering_only': args.engineering_smoke,
    'source_root': str(args.source_root.resolve()), 'condition_indices': list(indices),
    'methods': ['convex', 'affine', 'typed'], 'replicas': replicas,
    'expected_arms': len(indices)*len(replicas)*3, 'rows': [],
    'limitations': ['A complete process ledger is not scientific qualification.',
        'Every prescribed condition and optimization failure remains in the denominator.',
        'Source/pretraining/assessment compute remains additional.']}
write_json(output, report)
start = time.perf_counter()
for index in indices:
    for replica in replicas:
        for kind in report['methods']:
            name = f'condition_{index:02d}_{kind}_s{replica}'
            directory = args.out/name
            command = [sys.executable, '-u', str(root/'scripts/research/train_broad_entropy_adapter.py'),
                '--source-root', str(args.source_root), '--condition-index', str(index),
                '--out', str(directory), '--kind', kind, '--replica', str(replica)]
            if args.engineering_smoke:
                command += ['--engineering-smoke', '--steps', '2', '--eval-count', '16']
            with (args.out/f'{name}.log').open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            path = directory/'results.json'
            outcome = json.loads(path.read_text()) if path.exists() else {}
            row = {'condition_index': index, 'kind': kind, 'replica': replica, 'run': name,
                'exit_code': result.returncode, 'success': result.returncode == 0 and outcome.get('complete') is True
                    and outcome.get('checkpoint_loader_replay_passed') is True,
                'results_sha256': sha(path) if path.exists() else None,
                'log_sha256': sha(args.out/f'{name}.log'), 'oracle_evaluations': outcome.get('oracle_evaluations'),
                'known_query_lower_bound': outcome.get('oracle_evaluations_so_far', 0)}
            if not row['success']:
                row['failure'] = outcome.get('failure', 'Process or checkpoint qualification failed; inspect retained log')
            report['rows'].append(row)
            write_json(output, report)
            print(json.dumps(row), flush=True)
report.update(complete=True, successful_arms=sum(row['success'] for row in report['rows']),
    exact_query_accounting_complete=all(row['oracle_evaluations'] is not None for row in report['rows']),
    completed_queries_known=sum(row['oracle_evaluations'] or 0 for row in report['rows']), seconds=time.perf_counter()-start)
write_json(output, report)
