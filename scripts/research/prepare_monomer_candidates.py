#!/usr/bin/env python3
"""Recover prospectively eligible organic reference geometries, without generation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    source = args.project/'research/evidence/official_development_candidates.json'
    candidates = json.loads(source.read_text())
    assert candidates['complete'] and candidates['role'] == 'new_development'
    allowed = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}
    exclusions, used = {}, set()
    for name in ['development_panel_v1', 'transfer_development_panel_v1', 'proposal_training_panel_v1', 'tree_transfer_panel_v1']:
        path = args.project/'research/evidence'/f'{name}.json'
        exclusions[name] = sha(path)
        used.update(row['composition_hex'] for row in json.loads(path.read_text())['rows'])
    eligible = [(i, row) for i, row in enumerate(candidates['rows'])
        if 8 <= row['n_atoms'] <= 40 and row['charge'] == 0 and row['spin_multiplicity'] == 1
        and set(row['atomic_numbers']) <= allowed and 1 in row['atomic_numbers'] and 6 in row['atomic_numbers']
        and row['composition_hex'] not in used]
    old = json.loads((args.project/'runs/official_validation_audit_v1/audit.json').read_text())
    assert old['complete'] and old['validation_archive_sha256'] == candidates['source_archive_sha256']
    from fairchem.core.datasets import AseDBDataset
    from ase.calculators.singlepoint import SinglePointCalculator
    raw = AseDBDataset({'src': old['raw_directory']})
    assert len(raw) == old['source_raw_records']
    rows = []
    for index, row in eligible:
        atoms = raw.get_atoms(row['raw_index'])
        assert atoms.numbers.tolist() == row['atomic_numbers'] and atoms.info['source'] == row['source']
        assert int(atoms.info['charge']) == 0 and int(atoms.info['spin']) == 1
        assert isinstance(atoms.calc, SinglePointCalculator)
        assert float(atoms.get_potential_energy()) == row['energy_eV']
        x = atoms.positions.astype(np.float64)
        assert np.isfinite(x).all()
        x -= x.mean(0)
        condition = {k: v for k, v in row.items() if k != 'energy_eV'}
        condition['candidate_index'] = index
        rows.append(dict(condition=condition, reference_positions=x.tolist()))
    report = dict(complete=True, role='prospective_monomer_reference_pool', source_candidates_sha256=sha(source),
        excluded_panels=exclusions, excluded_compositions=sorted(used), rows=rows,
        domain=dict(allowed_atomic_numbers=sorted(allowed), require_carbon=True, require_hydrogen=True,
                    n_atoms=[8, 40], charge=0, spin_multiplicity=1),
        selection='Metadata only; all eligible references retained for qualification. Stored energies verify identity but never rank/select candidates.',
        new_molecular_oracle_calls=0, reference_coordinates_for_generation_or_training=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(references=len(rows), excluded_compositions=len(used))), flush=True)


if __name__ == '__main__':
    main()
