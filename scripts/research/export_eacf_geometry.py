#!/usr/bin/env python3
"""Export audited EACF physical coordinates for the independent geometry assessor.

Auxiliary variables, joint log volumes and joint KL changes are not physical
importance weights. This format deliberately contains no sampling weights.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result_path = args.run / 'results.json'
    report = json.loads(result_path.read_text())
    audit = json.loads(args.audit.read_text())
    reference = json.loads((args.reference / 'results.json').read_text())
    if (not report['complete'] or report['engineering_only'] or not audit['complete']
            or audit['results_sha256'] != sha(result_path)
            or audit['parents_replayed'] != report['evaluation_parents']):
        raise ValueError('Require all-parent audited production output')
    for name, digest in report['artifacts'].items():
        if sha(args.run / name) != digest:
            raise ValueError(f'Changed EACF artifact: {name}')
    for key in ['source_results_sha256', 'evaluation_sha256', 'refinement_protocol_sha256', 'condition']:
        if report[key] != reference[key]:
            raise ValueError(f'Mismatched physical comparison: {key}')
    base_path = args.reference / 'base_samples.pt'
    if not reference['complete'] or reference['artifacts']['base_samples.pt'] != sha(base_path):
        raise ValueError('Require intact reference parent artifact')
    base = torch.load(base_path, map_location='cpu', weights_only=False)
    saved = np.load(args.run / 'samples.npz')
    np.testing.assert_array_equal(saved['base_positions'], base['positions'].numpy())
    np.testing.assert_array_equal(saved['parent_ids'], base['sample_ids'])
    np.testing.assert_array_equal(saved['inversion_signs'], base['inversion_signs'].numpy())
    positions = torch.from_numpy(saved['positions'].copy())
    if positions.shape != base['positions'].shape or not torch.isfinite(positions).all():
        raise ValueError('Invalid physical coordinate array')
    args.out.mkdir(parents=True)
    output = args.out / 'physical_samples.pt'
    torch.save(dict(positions=positions, condition=report['condition'],
                    sample_ids=base['sample_ids'], inversion_signs=base['inversion_signs']), output)
    exported = {key: report[key] for key in ['condition', 'source_results_sha256',
        'evaluation_sha256', 'refinement_protocol_sha256', 'source_checkpoint_sha256',
        'source_kind', 'target_kind', 'kT_eV', 'restraint_eV_A2', 'replica']}
    exported.update(complete=True, scope=__doc__, kind='eacf_physical_coordinate_export',
        original_run=str(args.run.resolve()), original_results_sha256=sha(result_path),
        original_artifacts=report['artifacts'], audit_sha256=sha(args.audit),
        reference_results_sha256=sha(args.reference / 'results.json'),
        artifacts={output.name: sha(output)}, parents=len(positions),
        additional_oracle_evaluations=0, scientific_submission_ready=False)
    (args.out / 'results.json').write_text(json.dumps(exported, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(complete=True, parents=len(positions), new_oracle_queries=0)))


if __name__ == '__main__':
    main()
