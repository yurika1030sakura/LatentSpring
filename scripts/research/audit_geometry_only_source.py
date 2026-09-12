#!/usr/bin/env python3
"""Audit the fresh source's seeds/chunks and assess all generated structures."""
import argparse
from collections import Counter
import json
from pathlib import Path
import torch
from cfm_mol.chemical_moves import infer_chemical_graph, covalent_radii
from cfm_mol.geometric_domain import connected_nonoverlapping
from scripts.research.evaluate_chemical_policy import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'original-source', 'out']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    pp = root/'research/evidence/geometry_only_source_protocol_v1.json'
    protocol = json.loads(pp.read_text())
    report = json.loads((args.run/'results.json').read_text())
    assert report['complete'] and report['protocol_sha256'] == sha(pp)
    assert report['physical_queries'] == 0 and not report['energy_labels_computed']
    assert report['prefix_replay_max_error_A'] == 0 and report['global_rng_unchanged'] and report['input_positions_unchanged']
    assert sha(args.run/'samples.pt') == report['samples_sha256']
    data = torch.load(args.run/'samples.pt', map_location='cpu', weights_only=False)
    assert data['condition'] == report['condition'] and data['stream'] == report['stream'] == 'fresh_development'
    assert not set(data).intersection(['energy_eV', 'log_q', 'importance_weights', 'work'])
    assert data['sample_ids'] == list(range(protocol['count']))
    old_path = args.original_source/f"condition_{protocol['condition_index']:02d}/results.json"
    assert sha(old_path) == report['original_source_results_sha256']
    old = json.loads(old_path.read_text())
    assert old['condition'] == report['condition']
    old_seeds = {seed for stream in old['streams'].values() for seed in stream['batch_seeds']}
    expected = [1000000000*protocol['seed_namespace']+100003*report['condition']['candidate_index']+100000003*protocol['stream_number']+i for i in range(protocol['count']//protocol['batch'])]
    assert data['batch_seeds'] == report['batch_seeds'] == expected and not old_seeds.intersection(expected)
    for index in range(len(expected)):
        name = f'positions_{index:05d}.pt'
        path = args.run/'chunks'/name
        assert sha(path) == report['chunks'][name]
        chunk = torch.load(path, map_location='cpu', weights_only=False)
        start, end = index*protocol['batch'], (index+1)*protocol['batch']
        assert chunk['sample_ids'] == list(range(start, end)) and chunk['seed'] == expected[index]
        assert chunk['condition'] == data['condition'] and chunk['stream'] == data['stream']
        torch.testing.assert_close(chunk['positions'], data['positions'][start:end], atol=0, rtol=0)
    x = data['positions']
    assert torch.isfinite(x).all() and float(x.mean(1).abs().max()) < 1e-8
    condition = data['condition']
    radii = covalent_radii(condition['numbers'])
    geometric = connected_nonoverlapping(x, radii)
    records, identities = [], Counter()
    for parent in geometric.nonzero().flatten().tolist():
        try:
            graph = infer_chemical_graph(x[parent], condition['numbers'], condition['charge'])
            row = dict(parent_id=parent, supported=True, smiles=graph['connectivity_smiles'],
                radical_electrons=sum(graph['radical_electrons']))
            identities[row['smiles']] += 1
        except (ValueError, IndexError, RuntimeError) as exc:
            row = dict(parent_id=parent, supported=False, error_type=type(exc).__name__, reason=str(exc),
                validator_error=not isinstance(exc, ValueError))
        records.append(row)
    result = dict(complete=True, source_results_sha256=sha(args.run/'results.json'), source_samples_sha256=sha(args.run/'samples.pt'),
        protocol_sha256=sha(pp), attempted=len(x), geometrically_supported=int(geometric.sum()),
        chemically_supported=sum(row['supported'] for row in records),
        validator_errors=sum(row.get('validator_error', False) for row in records),
        supported_parent_ids=[row['parent_id'] for row in records if row['supported']],
        connectivity_counts=dict(identities), records=records,
        geometrically_rejected_parent_ids=(~geometric).nonzero().flatten().tolist(),
        no_seed_overlap_with_original_streams=True, all_chunk_rows_verified=True,
        physical_queries=0, generation_seconds=report['seconds'],
        neural_field_calls=report['generation_neural_field_calls'], scientific_submission_ready=False,
        scope='Fresh independent random draws for one DEVELOPMENT composition, not new molecular compositions or equilibrium samples. No energy ranking or fitting.')
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['complete', 'attempted', 'geometrically_supported', 'chemically_supported', 'validator_errors', 'physical_queries', 'generation_seconds']}))


if __name__ == '__main__':
    main()
