#!/usr/bin/env python3
"""Qualify references and exclude processed-corpus overlap before fixing a panel."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.research.audit_generator_output_support import assess
from scripts.research.audit_monomer_composition_overlap import file_hash, matching_rows
from scripts.research.tree_prior_fm import geometry_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['project', 'protocol', 'pool', 'out']:
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    spec = json.loads(args.protocol.read_text())
    pool = json.loads(args.pool.read_text())
    assert spec['frozen'] and pool['complete'] and pool['protocol_sha256'] == file_hash(args.protocol)
    start = time.perf_counter()
    decisions, qualified = [], []
    for i, row in enumerate(pool['rows']):
        c = dict(row['condition'], numbers=row['condition']['atomic_numbers'])
        x = torch.tensor(row['reference_positions'], dtype=torch.float64)[None]
        result = assess(x, c, [0])
        geometry = geometry_counts(x, c['numbers'])
        passed = result['graph_supported'] == 1 and result['validator_errors'] == 0 and geometry == dict(disconnected=0, overlap=0)
        decisions.append(dict(pool_index=i, qualified=passed, assessment=result, geometry=geometry))
        if passed:
            qualified.append(i)
    state = json.loads((args.project/'runs/raw_metadata_replay_v1/progress.json').read_text())
    assert state['complete']
    keys = [bytes.fromhex(pool['rows'][i]['condition']['composition_hex']) for i in qualified]
    weights = np.random.default_rng(30991).integers(0, np.iinfo(np.uint64).max, size=83, dtype=np.uint64)
    processed = Path(state['protocol'].get('processed_data_dir', '/n/holylabs/woo_lab/Lab/yulili/bgfm/processed_data/omol25_4m_processed'))
    corpora, overlaps = {}, set()
    for split in ['train', 'val']:
        path = processed/f'{split}_data_processed.pt'
        digest = file_hash(path)
        assert digest == state['protocol']['processed_sha256'][split]
        data = torch.load(str(path), mmap=True, weights_only=False, map_location='cpu')
        found = matching_rows(data['atom_types'].numpy(), data['node_idx_array'].numpy(), keys, weights)
        overlaps.update(k for k, indices in found.items() if indices)
        corpora[split] = dict(processed_sha256=digest, scanned_rows=len(data['node_idx_array']), matches=found)
        del data
        print(json.dumps(dict(split=split, overlapping_compositions=len(overlaps))), flush=True)
    selected, counts = [], {}
    for low, high in spec['size_bins']:
        candidates = [i for i in qualified if low <= pool['rows'][i]['condition']['n_atoms'] <= high
                      and pool['rows'][i]['condition']['composition_hex'] not in overlaps]
        candidates.sort(key=lambda i: pool['rows'][i]['condition']['selection_rank'])
        counts[f'{low}-{high}'] = len(candidates)
        selected.extend(dict(pool['rows'][i]['condition'], pool_index=i) for i in candidates[:spec['panel_per_bin']])
    audit = dict(complete=True, protocol_sha256=file_hash(args.protocol), pool_sha256=file_hash(args.pool),
        decisions=decisions, corpora=corpora, eligible_by_size=counts, selected=len(selected),
        seconds=time.perf_counter()-start, new_molecular_oracle_calls=0)
    (args.out/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    assert all(n >= spec['panel_per_bin'] for n in counts.values()), counts
    panel = dict(complete=True, role='new_development', rows=selected, source_pool=str(args.pool),
        source_pool_sha256=file_hash(args.pool), qualification_audit=str(args.out/'audit.json'),
        qualification_audit_sha256=file_hash(args.out/'audit.json'),
        selection='Fixed lowest metadata ranks among unchanged-assay references with zero exact composition overlap in both processed corpora.',
        reserved_outcomes_allowed=False, new_molecular_oracle_calls=0)
    (args.out/'panel.json').write_text(json.dumps(panel, indent=2)+'\n')
    print(json.dumps(dict(selected=len(selected), eligible_by_size=counts)), flush=True)


if __name__ == '__main__':
    main()
