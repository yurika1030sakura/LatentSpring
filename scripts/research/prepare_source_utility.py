#!/usr/bin/env python3
"""Select new train-corpus FIT and head-held-out compositions before generation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from cfm_mol.electronic_metadata import ElectronicMetadata
from scripts.research.audit_generator_output_support import assess
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['project', 'protocol', 'out']:
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    spec = json.loads(a.protocol.read_text()); assert spec['frozen']
    if a.out.exists(): raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    metadata = ElectronicMetadata(a.project/spec['metadata'], 'train', list(range(1, 84)))
    path = Path(spec['processed_train'])
    metadata.verify_processed_file(path)
    print('Verified training corpus hash', flush=True)
    data = torch.load(str(path), mmap=True, map_location='cpu', weights_only=False)
    ranges = data['node_idx_array']
    assert ranges.shape == (len(metadata.indices), 2)
    excluded, hashes = set(), {}
    for name in spec['exclude_manifests']:
        file = a.project/name
        hashes[name] = sha(file)
        assert hashes[name] == spec['exclude_manifest_sha256'][name]
        excluded.update(r['composition_hex'] for r in json.loads(file.read_text())['rows'])
    order = np.random.default_rng(spec['selection_seed']).permutation(len(ranges))
    bins = spec['size_bins']; selected = [[] for _ in bins]; decisions = []
    seen = set(excluded)
    for index in order[:spec['max_selection_scan']]:
        index = int(index)
        lo, hi = map(int, ranges[index]); n = hi-lo
        b = next((j for j, (low, high) in enumerate(bins) if low <= n <= high), None)
        if b is None or len(selected[b]) == spec['fit_per_bin']+spec['held_per_bin']: continue
        mi = metadata.indices[index]
        if not metadata.values['charge_known'][mi] or not metadata.values['spin_known'][mi]: continue
        if metadata.values['total_charge'][mi] != 0 or metadata.values['spin_multiplicity'][mi] != 1: continue
        numbers = data['atom_types'][lo:hi].long()+1
        species = set(numbers.tolist())
        if not species <= set(spec['allowed_atomic_numbers']) or not {1, 6} <= species: continue
        composition = np.bincount(numbers.numpy()-1, minlength=83).astype(np.uint8).tobytes().hex()
        if composition in seen: continue
        x = data['positions'][lo:hi].double(); x = x-x.mean(0)
        c = dict(atomic_numbers=numbers.tolist(), numbers=numbers.tolist(), n_atoms=n, charge=0,
                 spin_multiplicity=1, composition_hex=composition, processed_index=index,
                 raw_index=int(metadata.values['raw_indices'][mi]), source_split='train', size_bin=b,
                 panel_stratum=[*bins[b], 'neutral', 'singlet'])
        result = assess(x[None], c, [0])
        passed = result['graph_supported'] == 1 and result['validator_errors'] == 0
        decisions.append(dict(processed_index=index, composition_hex=composition, passed=passed, assessment=result))
        if not passed: continue
        selected[b].append(dict(condition=c, reference_positions=x.tolist()))
        seen.add(composition)
        if all(len(s) == spec['fit_per_bin']+spec['held_per_bin'] for s in selected): break
    assert all(len(s) == spec['fit_per_bin']+spec['held_per_bin'] for s in selected), 'Insufficient eligible compositions'
    fit = [r for group in selected for r in group[:spec['fit_per_bin']]]
    held = [r for group in selected for r in group[spec['fit_per_bin']:]]
    for name, rows in [('fit', fit), ('held', held)]:
        write(a.out/f'{name}_panel.json', dict(complete=True, role='new_development', rows=[r['condition'] for r in rows],
            intended_use='source_utility_fit' if name == 'fit' else 'head_held_out_fresh_generation',
            scope='Both panels originate from the FM training corpus; only the source-head FIT/held split is disjoint. No global pretraining holdout claim.',
            protocol_sha256=sha(a.protocol)))
    # Held references qualify the assay but never enter source fitting.
    write(a.out/'references.json', dict(fit=fit, held=held))
    write(a.out/'selection.json', dict(complete=True, protocol_sha256=sha(a.protocol), processed_train_sha256=spec['processed_train_sha256'],
        metadata_sha256=metadata.progress_sha256, exclusion_hashes=hashes, excluded_compositions=len(excluded), decisions=decisions,
        fit_compositions=len(fit), held_compositions=len(held), exact_fit_held_overlap=0,
        artifacts={name:sha(a.out/name) for name in ['fit_panel.json','held_panel.json','references.json']}, new_molecular_oracle_calls=0))
    print(json.dumps(dict(fit=len(fit), held=len(held), assessed=len(decisions))), flush=True)


if __name__ == '__main__': main()
