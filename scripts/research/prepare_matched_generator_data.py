#!/usr/bin/env python3
"""Shared OMol25 rows for independent, unpretrained generator comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from cfm_mol.electronic_metadata import ElectronicMetadata
from scripts.research.audit_generator_output_support import assess
from scripts.research.tree_prior_fm import geometry_counts
from scripts.research.train_electronic_fm import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['project', 'protocol', 'out']:
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.protocol.read_text())
    assert spec['frozen'] and not args.out.exists()
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    meta = ElectronicMetadata(args.project/spec['metadata'], 'train', list(range(1, 84)))
    path = Path(spec['processed_train'])
    meta.verify_processed_file(path)
    assert meta.state['protocol']['processed_sha256']['train'] == spec['processed_train_sha256']
    data = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    excluded = set()
    for name, digest in spec['exclusion_manifests'].items():
        file = args.project/name
        assert sha(file) == digest
        for row in json.loads(file.read_text())['rows']:
            excluded.add(row.get('condition', row)['composition_hex'])
    order = np.random.default_rng(spec['data_seed']).permutation(len(meta.indices))
    rows = dict(training=[], validation=[])
    counts = dict(metadata_eligible=0, reference_rejected=0, excluded=0)
    start = time.perf_counter()
    for scanned, index in enumerate(order[:spec['max_scan']], 1):
        index = int(index)
        low, high = map(int, data['node_idx_array'][index])
        n = high-low
        if not spec['min_atoms'] <= n <= spec['max_training_atoms']:
            continue
        m = meta.indices[index]
        if not meta.values['charge_known'][m] or not meta.values['spin_known'][m]:
            continue
        if meta.values['total_charge'][m] != 0 or meta.values['spin_multiplicity'][m] != 1:
            continue
        numbers = (data['atom_types'][low:high].long()+1).tolist()
        if not set(numbers) <= set(spec['atomic_numbers']) or not {1, 6} <= set(numbers):
            continue
        key = np.bincount(np.asarray(numbers)-1, minlength=83).astype(np.uint8).tobytes().hex()
        if key in excluded:
            counts['excluded'] += 1
            continue
        split = 'validation' if int.from_bytes(hashlib.sha256(('matched-generator-v1|'+key).encode()).digest()[:8], 'big') % 10 == 0 else 'training'
        if len(rows[split]) == spec[split+'_rows']:
            continue
        counts['metadata_eligible'] += 1
        x = data['positions'][low:high].double().clone()
        x -= x.mean(0)
        c = dict(numbers=numbers, atomic_numbers=numbers, n_atoms=n, charge=0, spin_multiplicity=1,
                 composition_hex=key, processed_index=index, source_split='train')
        result = assess(x[None], c, [0])
        if result['graph_supported'] != 1 or result['validator_errors'] or geometry_counts(x[None], numbers) != dict(disconnected=0, overlap=0):
            counts['reference_rejected'] += 1
            continue
        rows[split].append(dict(condition=c, positions=x))
        if len(rows['training']) % 1000 == 0:
            print(json.dumps(dict(training=len(rows['training']), validation=len(rows['validation']), scanned=scanned)), flush=True)
        if all(len(rows[s]) == spec[s+'_rows'] for s in rows):
            break
    assert all(len(rows[s]) == spec[s+'_rows'] for s in rows), {s:len(v) for s,v in rows.items()}
    assert not {r['condition']['composition_hex'] for r in rows['training']} & {r['condition']['composition_hex'] for r in rows['validation']}
    torch.save(rows, args.out/'data.pt')
    report = dict(complete=True, protocol_sha256=sha(args.protocol), data_sha256=sha(args.out/'data.pt'),
        processed_train_sha256=spec['processed_train_sha256'], metadata_progress_sha256=meta.progress_sha256,
        selected_indices={s:[r['condition']['processed_index'] for r in rr] for s,rr in rows.items()},
        composition_counts={s:len({r['condition']['composition_hex'] for r in rr}) for s,rr in rows.items()},
        counts=counts, scanned=scanned, seconds=time.perf_counter()-start, new_molecular_oracle_calls=0,
        selection='Fixed training-index permutation; neutral organic singlets, raw graph and geometry qualification; no scaffold-tree filter. Validation separated by composition hash. No pretraining.')
    (args.out/'selection.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'selected_indices'}), flush=True)


if __name__ == '__main__':
    main()
