#!/usr/bin/env python3
"""Real-oracle bounded-RPC replay on the first and last prescribed conditions."""
import argparse
import json
from pathlib import Path
import time

import torch

from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.entropy_source import load_entropy_source
from molecular_tempered_pilot import sha, write_json


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-root', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
args = p.parse_args()
output = args.out/'results.json'
if output.exists():
    raise FileExistsError(output)
root = Path(__file__).resolve().parents[2]
args.out.mkdir(parents=True, exist_ok=True)
report = {'complete': False, 'scope': __doc__, 'rows': [], 'maximum_rpc_structures': 32,
    'scientific_sampling_qualified': False}
write_json(output, report)
for index in [0, 7]:
    data = load_entropy_source(args.source_root, index, engineering=True,
        protocol_path=root/'research/evidence/species_breadth_source_protocol_v2.json',
        manifest_path=root/'research/evidence/development_panel_v1.json')
    condition, recipe = data['condition'], data['recipe']
    oracle_path = Path(recipe['oracle'])
    if sha(oracle_path) != data['source']['oracle_sha256']:
        raise ValueError('Physical checkpoint changed')
    x = torch.cat([data[k]['positions'] for k in ['training', 'development']])
    old_energy = torch.cat([data[k]['energy_eV'] for k in ['training', 'development']])
    with EnergyOracle(Path(recipe['oracle_python']), root/'scripts/research/oracle_worker.py', oracle_path,
            numbers=condition['numbers'], charge=condition['charge'],
            spin_multiplicity=condition['spin_multiplicity'], batch_size=16) as oracle:
        original = oracle.evaluate
        timings = []
        def evaluate(item):
            start = time.perf_counter()
            result = original(item)
            timings.append({'structures': len(item), 'seconds': time.perf_counter()-start})
            return result
        oracle.evaluate = evaluate
        energy, force = oracle.evaluate_chunked(x)
        error = float((energy-old_energy).abs().max())
        if error > .001 or not torch.isfinite(force).all() or oracle.requested_evaluations != oracle.evaluated or oracle.evaluated != 64:
            raise RuntimeError('Real bounded-RPC source replay failed')
        row = {'index': index, 'n_atoms': len(condition['numbers']), 'energy_replay_max_error_eV': error,
            'oracle_evaluations': oracle.evaluated, 'requested_evaluations': oracle.requested_evaluations,
            'rpc_timings': timings, 'source_results_sha256': data['source_results_sha256']}
        report['rows'].append(row)
        write_json(output, report)
        print(json.dumps(row), flush=True)
report.update(complete=True, oracle_evaluations=sum(r['oracle_evaluations'] for r in report['rows']))
write_json(output, report)
