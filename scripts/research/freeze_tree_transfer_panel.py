#!/usr/bin/env python3
"""Select additional development compositions using only fixed metadata rules."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    root = args.project
    source = root/'research/evidence/official_development_candidates.json'
    candidates = json.loads(source.read_text())
    assert candidates['complete'] and candidates['role'] == 'new_development'
    exclusions = {}
    used = set()
    for filename in ['development_panel_v1.json', 'transfer_development_panel_v1.json', 'proposal_training_panel_v1.json']:
        path = root/'research/evidence'/filename
        panel = json.loads(path.read_text())
        used.update(row['composition_hex'] for row in panel['rows'])
        exclusions[filename] = sha(path)
    original_exclusions = sorted(used)
    seed = 30691
    rows, counts = [], {}
    for lo, hi in [(2, 12), (13, 24)]:
        for neutral, singlet in [(True, True), (True, False), (False, True), (False, False)]:
            stratum = [lo, hi, 'neutral' if neutral else 'charged', 'singlet' if singlet else 'open_shell']
            eligible = []
            for index, row in enumerate(candidates['rows']):
                if (row['partition'] == 'new_development' and lo <= row['n_atoms'] <= hi
                        and (row['charge'] == 0) == neutral and (row['spin_multiplicity'] == 1) == singlet
                        and row['composition_hex'] not in used):
                    rank = hashlib.sha256(f'{seed}|{index}|{row["source"]}'.encode()).hexdigest()
                    eligible.append((rank, index, row))
            counts[str(stratum)] = len({row['composition_hex'] for _, _, row in eligible})
            selected = 0
            for rank, index, row in sorted(eligible):
                if row['composition_hex'] in used:
                    continue
                numbers, charge, spin = row['atomic_numbers'], row['charge'], row['spin_multiplicity']
                assert len(numbers) == row['n_atoms'] and sum(numbers)-charge >= spin-1
                assert (sum(numbers)-charge-spin+1) % 2 == 0
                new = {key: value for key, value in row.items() if key != 'energy_eV'}
                new.update(candidate_index=index, panel_stratum=stratum, transfer_selection_sha256=rank)
                rows.append(new)
                used.add(row['composition_hex'])
                selected += 1
                if selected == 4:
                    break
            if selected != 4:
                raise ValueError('Insufficient independent compositions for '+str(stratum))
    assert len(rows) == len({row['composition_hex'] for row in rows}) == 32
    assert not set(original_exclusions).intersection(row['composition_hex'] for row in rows)
    result = dict(complete=True, role='new_development', intended_use='prospective_generator_transfer', seed=seed,
        source_candidates_sha256=sha(source), excluded_panels=exclusions, excluded_compositions=original_exclusions,
        eligible_compositions_by_stratum=counts, rows=rows,
        selection='Four lowest fixed hash ranks per size2-12/13-24, charge and spin stratum. Distinct compositions; exclude the three declared prior panels. No energy or generated-geometry selection; no element exclusion.',
        scope='Additional development compositions, not a certified composition-disjoint warm-pretraining test set. Full pretraining composition overlap is not audited. No outcomes have been queried for this panel in its current role.',
        reserved_outcomes_allowed=False, new_molecular_oracle_calls=0, scientific_submission_ready=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(selected=len(rows), excluded=len(original_exclusions), eligible=counts)), flush=True)


if __name__ == '__main__':
    main()
