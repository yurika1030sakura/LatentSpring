#!/usr/bin/env python3
"""Recover frozen-panel reference coordinates for assay calibration only."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    panel_path = args.project/'research/evidence/tree_transfer_panel_v1.json'
    candidates_path = args.project/'research/evidence/official_development_candidates.json'
    panel = json.loads(panel_path.read_text())
    candidates = json.loads(candidates_path.read_text())
    assert panel['complete'] and panel['role'] == candidates['role'] == 'new_development'
    assert digest(candidates_path) == panel['source_candidates_sha256']
    archive_audit = json.loads((args.project/'runs/official_validation_audit_v1/audit.json').read_text())
    assert archive_audit['complete'] and archive_audit['validation_archive_sha256'] == candidates['source_archive_sha256']
    from fairchem.core.datasets import AseDBDataset
    raw = AseDBDataset({'src': archive_audit['raw_directory']})
    assert len(raw) == archive_audit['source_raw_records']
    rows = []
    for index, condition in enumerate(panel['rows']):
        original = candidates['rows'][condition['candidate_index']]
        for key in ['raw_index', 'source', 'atomic_numbers', 'charge', 'spin_multiplicity', 'composition_hex']:
            assert condition[key] == original[key]
        atoms = raw.get_atoms(condition['raw_index'])
        assert atoms.numbers.tolist() == condition['atomic_numbers']
        assert atoms.info['source'] == condition['source']
        assert int(atoms.info['charge']) == condition['charge'] and int(atoms.info['spin']) == condition['spin_multiplicity']
        # This is a stored SinglePointCalculator label, not a new oracle query.
        assert float(atoms.get_potential_energy()) == original['energy_eV']
        x = atoms.positions.astype('float64')
        x -= x.mean(0)
        rows.append(dict(condition_index=index, condition=condition, positions=x.tolist()))
    report = dict(complete=True, panel_sha256=digest(panel_path), source_candidates_sha256=digest(candidates_path), rows=rows,
        scope='Post-experiment calibration of the unchanged structural readout on original reference coordinates. References were not supplied to generation, fitting or panel selection. Do not replace or filter generated outcomes with these results.',
        new_molecular_oracle_calls=0, scientific_submission_ready=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'Recovered{len(rows)} original reference geometries; zero new oracle queries.', flush=True)


if __name__ == '__main__':
    main()
