#!/usr/bin/env python3
"""Partition existing raw-output failures without new generation or energy calls."""
import argparse
import json
from pathlib import Path

import torch

from scripts.research.audit_source_utility import flags
from scripts.research.train_electronic_fm import sha
from scripts.research.tree_prior_fm import geometry_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    audit_path = args.project / 'research/evidence/fresh_physics_esen_audit_v1.json'
    audit = json.loads(audit_path.read_text())
    assert audit['complete']
    rows = []
    for artifact in audit['artifacts']:
        seed, method = artifact['seed'], artifact['method']
        root = Path(audit['generation_run']) / f's{seed}/study/evaluation'
        report_path = root / f'{method}_results.json'
        assert sha(report_path) == artifact['report_sha256']
        report = json.loads(report_path.read_text())
        counts = dict(attempted=0, disconnected_only=0, overlap_only=0,
                      both_disconnected_and_overlap=0,
                      geometry_ok_graph_rejected=0, graph_supported=0)
        for row in report['rows']:
            path = root / f"{method}_c{row['condition_index']}.pt"
            assert sha(path) == row['sample_sha256']
            sample = torch.load(path, map_location='cpu', weights_only=False)
            graph = flags(row, 'graph_supported').astype(bool)
            geometry = flags(row, 'geometrically_supported').astype(bool)
            for index, x in enumerate(sample['positions']):
                defects = geometry_counts(x[None], sample['condition']['numbers'])
                disconnected, overlap = defects['disconnected'], defects['overlap']
                assert bool(geometry[index]) == (not disconnected and not overlap)
                assert not graph[index] or geometry[index]
                if disconnected and overlap:
                    category = 'both_disconnected_and_overlap'
                elif disconnected:
                    category = 'disconnected_only'
                elif overlap:
                    category = 'overlap_only'
                elif graph[index]:
                    category = 'graph_supported'
                else:
                    category = 'geometry_ok_graph_rejected'
                counts[category] += 1
                counts['attempted'] += 1
        assert counts['attempted'] == sum(v for k, v in counts.items() if k != 'attempted')
        assert counts['graph_supported'] == audit['summary'][str(seed)][method]['graph_supported']
        rows.append(dict(seed=seed, method=method, **counts))
    pooled = {m: {k: sum(r[k] for r in rows if r['method'] == m)
                  for k in counts} for m in audit['methods']}
    result = dict(complete=True, rows=rows, pooled=pooled,
                  new_generation_or_physical_queries=0,
                  scope='Disjoint failure categories of the already frozen unoptimized outputs; no validator or denominator changes.')
    if args.out.exists():
        assert json.loads(args.out.read_text()) == result
        print('Existing raw-failure diagnosis independently reproduced.')
    else:
        args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(pooled, indent=2))


if __name__ == '__main__':
    main()
