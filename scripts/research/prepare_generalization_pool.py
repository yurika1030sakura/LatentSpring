#!/usr/bin/env python3
"""Freeze a wider composition pool using metadata, before any model generation."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['project', 'protocol', 'out']:
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.protocol.read_text())
    assert spec['frozen'] and not args.out.exists()
    oldpath = args.project/'runs/official_validation_audit_v1/audit.json'
    old = json.loads(oldpath.read_text())
    assert old['complete'] and sha(oldpath) == spec['validation_audit_sha256']
    excluded = set()
    for name, digest in spec['exclusion_manifests'].items():
        path = args.project/name
        assert sha(path) == digest
        previous = json.loads(path.read_text())
        for row in previous['rows']:
            excluded.add(row.get('condition', row)['composition_hex'])
    from fairchem.core.datasets import AseDBDataset
    raw = AseDBDataset({'src': old['raw_directory']})
    sizes = np.load(Path(old['raw_directory'])/'metadata.npz')['natoms']
    assert len(raw) == len(sizes) == old['source_raw_records']
    rows, seen, counts = [], set(), {}
    start = time.perf_counter()
    for bin_index, (low, high) in enumerate(spec['size_bins']):
        order = np.flatnonzero((sizes >= low) & (sizes <= high))
        np.random.default_rng(spec['selection_seed']+bin_index).shuffle(order)
        accepted, scanned = 0, 0
        for index in order:
            scanned += 1
            atoms = raw.get_atoms(int(index))
            numbers, info = atoms.numbers.tolist(), atoms.info
            if not set(numbers) <= set(spec['atomic_numbers']) or not {1, 6} <= set(numbers):
                continue
            if info.get('charge') != 0 or info.get('spin') != 1 or sum(numbers) % 2:
                continue
            key = np.bincount(np.asarray(numbers)-1, minlength=83).astype(np.uint8).tobytes()
            hexkey = key.hex()
            if hexkey in excluded or hexkey in seen:
                continue
            if int.from_bytes(hashlib.sha256(str(old['seed']).encode()+key).digest()[:8], 'big') % 5 != 0:
                continue
            x = atoms.positions.astype(np.float64)
            x -= x.mean(0)
            if not np.isfinite(x).all():
                continue
            condition = dict(raw_index=int(index), source=info['source'],
                reference_source=info.get('reference_source'), data_id=str(info.get('data_id', 'unknown')),
                atomic_numbers=numbers, n_atoms=len(numbers), charge=0, spin_multiplicity=1,
                composition_hex=hexkey, partition='new_development',
                selection_rank=hashlib.sha256((str(spec['rank_seed'])+'|'+hexkey).encode()).hexdigest(),
                panel_stratum=[low, high, 'neutral', 'singlet'])
            rows.append(dict(condition=condition, reference_positions=x.tolist()))
            seen.add(hexkey)
            accepted += 1
            if accepted == spec['pool_per_bin']:
                break
        counts[f'{low}-{high}'] = dict(selected=accepted, scanned=scanned)
        print(json.dumps(counts), flush=True)
        assert accepted == spec['pool_per_bin'] or spec.get('allow_exhausted_bins',False), counts
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(complete=True, protocol_sha256=sha(args.protocol),
        source_archive_sha256=old['validation_archive_sha256'], old_partition_seed=old['seed'],
        excluded_manifests=spec['exclusion_manifests'], excluded_compositions=sorted(excluded),
        rows=rows, counts=counts, seconds=time.perf_counter()-start,
        selection='Fixed metadata permutation within size strata; no generator outcomes or energy ranking.',
        reserved_outcomes_allowed=False, new_molecular_oracle_calls=0), indent=2)+'\n')


if __name__ == '__main__':
    main()
