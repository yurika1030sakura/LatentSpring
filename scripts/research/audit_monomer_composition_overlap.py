#!/usr/bin/env python3
"""Exact composition checks after a fast additive-hash screen of processed corpora."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from ase.data import atomic_numbers


def file_hash(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def matching_rows(types, nodes, keys, weights, batch=16384):
    fingerprints = np.array([np.sum(np.frombuffer(key, dtype=np.uint8).astype(np.uint64)*weights, dtype=np.uint64) for key in keys], dtype=np.uint64)
    selected = set(keys)
    matches = {key.hex(): [] for key in keys}
    for begin in range(0, len(nodes), batch):
        ranges = np.asarray(nodes[begin:begin+batch])
        lo, hi = int(ranges[:, 0].min()), int(ranges[:, 1].max())
        values = np.asarray(types[lo:hi])
        if values.ndim != 1 or values.min() < 0 or values.max() >= len(weights):
            raise ValueError('Expected verified zero-based atomic-type indices')
        prefix = np.empty(len(values)+1, dtype=np.uint64)
        prefix[0] = 0
        np.cumsum(weights[values], dtype=np.uint64, out=prefix[1:])
        sums = prefix[ranges[:, 1]-lo]-prefix[ranges[:, 0]-lo]
        for local in np.flatnonzero(np.isin(sums, fingerprints)):
            start, stop = ranges[local]
            key = np.bincount(np.asarray(types[start:stop]), minlength=len(weights)).astype(np.uint8).tobytes()
            if key in selected:  # Exact count-vector verification eliminates hash collisions.
                matches[key.hex()].append(begin+int(local))
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['project', 'panel', 'out']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    panel = json.loads(args.panel.read_text())
    assert panel['complete'] and panel['role'] == 'new_development'
    cfg = yaml.safe_load((args.project/'configs/sweep/a1_fm_only_s2.yaml').read_text())
    assert [atomic_numbers[s] for s in cfg['dataset']['atom_map']] == list(range(1, 84))
    state = json.loads((args.project/'runs/raw_metadata_replay_v1/progress.json').read_text())
    assert state['complete']
    keys = []
    for row in panel['rows']:
        key = np.bincount(np.array(row['atomic_numbers'])-1, minlength=83).astype(np.uint8).tobytes()
        assert key.hex() == row['composition_hex']
        keys.append(key)
    weights = np.random.default_rng(30991).integers(0, np.iinfo(np.uint64).max, size=83, dtype=np.uint64)
    # Deliberate hash collisions in a tiny control still require exact equality.
    example = np.array([0, 5, 0, 5, 0, 0, 7, 0], dtype=np.int8)
    bounds = np.array([[0, 3], [3, 6], [6, 8]])
    target = np.bincount(example[:3], minlength=83).astype(np.uint8).tobytes()
    assert matching_rows(example, bounds, [target], np.zeros(83, dtype=np.uint64), batch=2)[target.hex()] == [0, 1]
    start = time.perf_counter()
    reports = {}
    for split in ['train', 'val']:
        path = Path(cfg['dataset']['processed_data_dir'])/f'{split}_data_processed.pt'
        digest = file_hash(path)
        assert digest == state['protocol']['processed_sha256'][split]
        data = torch.load(str(path), map_location='cpu', mmap=True, weights_only=False)
        types, nodes = data['atom_types'].numpy(), data['node_idx_array'].numpy()
        found = matching_rows(types, nodes, keys, weights)
        reports[split] = dict(processed_sha256=digest, scanned_rows=len(nodes),
            overlapping_compositions=sum(bool(value) for value in found.values()), matches=found)
        del data, types, nodes
        print(split, reports[split]['scanned_rows'], 'rows; overlapping compositions', reports[split]['overlapping_compositions'], flush=True)
    output = dict(complete=True, panel_sha256=file_hash(args.panel), corpora=reports, seconds=time.perf_counter()-start,
        exact_count_vector_verification=True, new_molecular_oracle_calls=0,
        scope='Checks whole-system elemental counts against the two checksum-verified processed corpora, independent of charge/spin. Does not certify arbitrary unrecorded checkpoint pretraining or complete trajectory identities.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
