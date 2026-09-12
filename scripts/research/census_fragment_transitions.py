#!/usr/bin/env python3
"""Count perceived graph changes across ALL move families of audited trajectories."""
import argparse
import json
from pathlib import Path
import torch
from scripts.research.evaluate_chemical_policy import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'remainder', 'audit', 'out']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    assert audit['complete']
    summaries = []
    for entry in audit['rows']:
        index, method, replica = entry['condition_index'], entry['method'], entry['replica']
        relative = Path(f'condition_{index}/{method}_s{replica}')
        directory = args.run/relative
        if not directory.exists():
            directory = args.remainder/relative
        report = json.loads((directory/'results.json').read_text())
        trace = directory/('trace.pt' if entry['sampling_arm_complete'] else 'failed_trace.pt')
        assert sha(trace) == entry['trace_sha256']
        data = torch.load(trace, map_location='cpu', weights_only=False)
        mask = torch.tensor([z not in (1, 9, 17, 35, 53) for z in report['condition']['numbers']])
        totals = {kind: dict(accepted=0, bond_order_matrix_changes=0, backbone_adjacency_changes=0,
            canonical_smiles_changes=0, backbone_and_smiles_changes=0, multiatom_changes=0)
            for kind in ['local', 'force_rotation', 'fragment_exchange']}
        events = []
        for row in data['transitions']:
            if not row['accepted']:
                continue
            before, after = [data['states'][row[key]] for key in ['old_state_id', 'new_state_id']]
            a, b = before['graph']['bond_orders'], after['graph']['bond_orders']
            adjacency = not torch.equal(a[mask][:, mask] > 0, b[mask][:, mask] > 0)
            orders = not torch.equal(a, b)
            smiles = before['graph']['connectivity_smiles'] != after['graph']['connectivity_smiles']
            counts = totals[row['kind']]
            counts['accepted'] += 1
            counts['bond_order_matrix_changes'] += orders
            counts['backbone_adjacency_changes'] += adjacency
            counts['canonical_smiles_changes'] += smiles
            counts['backbone_and_smiles_changes'] += adjacency and smiles
            counts['multiatom_changes'] += bool(row.get('multiatom', False))
            if orders or smiles:
                events.append(dict(phase=row['phase'], kind=row['kind'], old_state_id=row['old_state_id'],
                    new_state_id=row['new_state_id'], backbone_adjacency_changed=adjacency,
                    canonical_smiles_changed=smiles, old_smiles=before['graph']['connectivity_smiles'],
                    new_smiles=after['graph']['connectivity_smiles'],
                    potential_change_eV=float(after['potential_eV']-before['potential_eV'])))
        summary = dict(condition_index=index, method=method, replica=replica,
            sampling_arm_complete=entry['sampling_arm_complete'], moves=totals,
            all_move_backbone_changes=sum(v['backbone_adjacency_changes'] for v in totals.values()),
            all_move_backbone_and_smiles_changes=sum(v['backbone_and_smiles_changes'] for v in totals.values()),
            events=events, trace_sha256=entry['trace_sha256'])
        summaries.append(summary)
        print(index, method, replica, summary['all_move_backbone_changes'], summary['all_move_backbone_and_smiles_changes'])
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(dict(complete=True, rows=summaries, audit_sha256=sha(args.audit),
        new_physical_queries=0, scientific_submission_ready=False,
        limitation='Adjacency changes are labelled diagnostics and can include atom relabeling. Canonical SMILES may reflect charge/bond assignment changes. Neither alone is a physical reaction or equilibrium certificate.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
