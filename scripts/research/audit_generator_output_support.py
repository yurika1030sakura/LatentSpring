#!/usr/bin/env python3
"""Common structural readout for existing FM and learned-generator outputs.

This does not reinterpret their original training targets or KL quantities.
All512 development parents are retained; auxiliary EACF variables are not weights.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
from cfm_mol.chemical_moves import infer_chemical_graph, covalent_radii
from cfm_mol.geometric_domain import connected_nonoverlapping
from scripts.research.evaluate_chemical_policy import sha


def assess(x, condition, parents):
    radii = covalent_radii(condition['numbers'])
    geometry = connected_nonoverlapping(x, radii)
    records, smiles = [], Counter()
    for index, pos in enumerate(x):
        row = dict(parent_id=int(parents[index]), geometrically_supported=bool(geometry[index]),
                   graph_supported=False, validator_error=False)
        if geometry[index]:
            try:
                graph = infer_chemical_graph(pos, condition['numbers'], condition['charge'])
                row.update(graph_supported=True, smiles=graph['connectivity_smiles'],
                    radical_electrons=sum(graph['radical_electrons']))
                smiles[row['smiles']] += 1
            except (ValueError, IndexError, RuntimeError) as exc:
                row.update(error_type=type(exc).__name__, reason=str(exc), validator_error=not isinstance(exc, ValueError))
        records.append(row)
    return dict(attempted=len(x), geometrically_supported=int(geometry.sum()),
        graph_supported=sum(r['graph_supported'] for r in records),
        validator_errors=sum(r['validator_error'] for r in records),
        distinct_connectivity=len(smiles), connectivity_counts=dict(smiles), records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for replica in [0, 1]:
        reference_dir = args.runs_root/f'parity_cpu_gpu_control_s{replica}_v1'
        reference = json.loads((reference_dir/'results.json').read_text())
        assert reference['complete'] and reference['checkpoint_loader_replay_passed']
        for name, digest in reference['artifacts'].items():
            assert sha(reference_dir/name) == digest
        base = torch.load(reference_dir/'base_samples.pt', map_location='cpu', weights_only=False)
        assert len(base['positions']) == 512
        arrays = [('FM64', base['positions'], None, None)]
        adapted = torch.load(reference_dir/'adapted_samples.pt', map_location='cpu', weights_only=False)
        arrays.append(('convex_refiner', adapted['positions'], sha(reference_dir/'results.json'), reference['parameters']))
        provenance = {}
        for method, stem, audit_stem in [('EACF_published', 'eacf_joint', 'eacf_joint_audit'),
                                           ('EACF_compact', 'eacf_compact', 'eacf_compact_audit')]:
            directory = args.runs_root/f'{stem}_condition_00_s{replica}_v1'
            report = json.loads((directory/'results.json').read_text())
            audit_path = args.runs_root/f'{audit_stem}_s{replica}_v1/audit.json'
            audit = json.loads(audit_path.read_text())
            assert report['complete'] and not report['engineering_only'] and audit['complete']
            assert audit['results_sha256'] == sha(directory/'results.json') and audit['parents_replayed'] == 512
            for key in ['condition', 'source_results_sha256', 'evaluation_sha256', 'refinement_protocol_sha256']:
                assert report[key] == reference[key]
            for name, digest in report['artifacts'].items():
                assert sha(directory/name) == digest
            saved = np.load(directory/'samples.npz')
            np.testing.assert_array_equal(saved['base_positions'], base['positions'].numpy())
            np.testing.assert_array_equal(saved['parent_ids'], base['sample_ids'])
            np.testing.assert_array_equal(saved['inversion_signs'], base['inversion_signs'].numpy())
            arrays.append((method, torch.from_numpy(saved['positions'].copy()), sha(directory/'results.json'), report['parameters']))
            provenance[method] = dict(audit_sha256=sha(audit_path), source_run=str(directory),
                original_training_target=report['target_kind'], oracle_evaluations=report['oracle_evaluations'])
        for method, positions, results_hash, parameters in arrays:
            assert positions.shape == base['positions'].shape and torch.isfinite(positions).all()
            assert float(positions.mean(1).abs().max()) < 1e-8
            assessment = assess(positions, reference['condition'], base['sample_ids'])
            if method == 'FM64':
                assert assessment['graph_supported'] == 4
            row = dict(method=method, replica=replica, parameters=parameters,
                results_sha256=results_hash, common_reference_sha256=sha(reference_dir/'results.json'),
                source_results_sha256=reference['source_results_sha256'], condition=reference['condition'],
                provenance=provenance.get(method), **assessment)
            rows.append(row)
            print(json.dumps({key: row[key] for key in ['method', 'replica', 'attempted', 'graph_supported',
                'geometrically_supported', 'validator_errors', 'distinct_connectivity']}), flush=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(dict(complete=True, rows=rows, new_physical_queries=0,
        scientific_submission_ready=False, current_MCMC_method_included=False,
        scope='Common-output structural pilot of existing frozen generator baselines on the same512 development parent IDs of condition0.',
        limitations=['Not a complete end-to-end benchmark or a claim that any method follows a Boltzmann law.',
                     'Original generator training objectives and measured costs are retained; this new support assessment does not relabel their old targets.',
                     'Two model seeds share development parents and are not1024 independent molecular conditions.',
                     'Current MCMC output must enter only after matching output count, source attempts, correlation and cost conventions.']), indent=2)+'\n')


if __name__ == '__main__':
    main()
