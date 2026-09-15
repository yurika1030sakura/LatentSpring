#!/usr/bin/env python3
"""Read out frozen generator geometries without changing their coordinates."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import time

import torch

from cfm_mol.energy_oracle import EnergyOracle
from scripts.research.evaluate_fresh_primary_xtb import run_task
from scripts.research.train_electronic_fm import sha


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def load(args, spec):
    for key in ['condition_manifest', 'source_pool']:
        assert sha(args.project/spec[key]) == spec[key+'_sha256']
    panel = json.loads((args.project/spec['condition_manifest']).read_text())['rows']
    pool = json.loads((args.project/spec['source_pool']).read_text())['rows']
    conditions = {i:dict(c, numbers=c['atomic_numbers']) for i,c in enumerate(panel)
                  if i % spec['chunks'] == args.chunk}
    samples, references = {}, {}
    for i,c in conditions.items():
        row = pool[c['pool_index']]
        assert row['condition']['raw_index'] == c['raw_index']
        references[i] = torch.tensor(row['reference_positions'], dtype=torch.float64)
    for row in spec['outputs']:
        i, method = row['condition_index'], row['method']
        if i not in conditions:
            continue
        assert sha(args.project/row['sample']) == row['sample_sha256']
        assert sha(args.project/row['report']) == row['report_sha256']
        saved = torch.load(args.project/row['sample'], map_location='cpu', weights_only=False)
        c = conditions[i]
        assert saved['condition']['numbers'] == c['numbers']
        assert saved['condition']['composition_hex'] == c['composition_hex']
        assert saved['calls_per_sample'] == 128 and len(saved['positions']) == spec['samples_per_condition']
        samples[method,i] = (saved['positions'], row)
    return conditions, references, samples


def esen(args, spec, conditions, references, samples, report):
    worker = args.project/spec['oracle_worker']
    assert sha(worker) == spec['oracle_worker_sha256'] and sha(spec['oracle_checkpoint']) == spec['oracle_sha256']
    c = next(iter(conditions.values()))
    with EnergyOracle(spec['oracle_interpreter'], worker, spec['oracle_checkpoint'], numbers=c['numbers'],
                      charge=c['charge'], spin_multiplicity=c['spin_multiplicity'], device='cuda',
                      batch_size=8, timeout_seconds=180.) as oracle:
        assert oracle.handshake['base_precision_dtype'] == 'torch.float32' and not oracle.handshake['tf32']
        report['oracle_handshake'] = oracle.handshake
        for i,c in conditions.items():
            oracle.condition = dict(numbers=c['numbers'], charge=c['charge'], spin_multiplicity=c['spin_multiplicity'])
            for method in ['reference']+spec['methods']:
                x = references[i][None] if method == 'reference' else samples[method,i][0]
                n = len(x)
                energy, force = oracle.evaluate_chunked(torch.cat([x,-x]), max_request=16)
                file = args.out/f'{method}_c{i}.pt'
                payload = dict(positions=x, raw_energy_eV=energy, raw_force_eV_A=force,
                    even_energy_eV=(energy[:n]+energy[n:])/2, even_force_eV_A=(force[:n]-force[n:])/2,
                    condition=c, protocol_sha256=report['protocol_sha256'])
                if method != 'reference':
                    payload['source_sample_sha256'] = samples[method,i][1]['sample_sha256']
                torch.save(payload, file)
                report['rows'].append(dict(method=method, condition_index=i, artifact=file.name,
                    artifact_sha256=sha(file), raw_queries=2*n))
            report['raw_queries'] = oracle.evaluated
            write(args.out/'results.json', report)
            print(json.dumps(dict(condition=i, raw_queries=oracle.evaluated)), flush=True)
        assert oracle.evaluated == oracle.requested_evaluations == 2*len(conditions)*(1+len(spec['methods'])*32)


def xtb(args, spec, conditions, references, samples, report):
    binary = Path(spec['xtb_binary'])
    assert sha(binary) == spec['xtb_binary_sha256']
    version = subprocess.run([str(binary), '--version'], capture_output=True, text=True, check=True)
    report['version'] = version.stdout+version.stderr
    tasks = []
    for i,c in conditions.items():
        for method in ['reference']+spec['methods']:
            positions = references[i][None] if method == 'reference' else samples[method,i][0]
            keep = [True] if method == 'reference' else samples[method,i][1]['graph_valid']
            for j,x in enumerate(positions):
                if not keep[j]:
                    continue
                for minus in ([False, True] if method == 'reference' else [False]):
                    ident = f'{method}_c{i}_s{j}_'+('minus' if minus else 'plus')
                    task = dict(task_id=ident, method=method, replica=spec['model_seed'], parent_id=i*32+j,
                        inversion_check=minus, condition_index=i, sample_index=j,
                        positions=(-x if minus else x).tolist())
                    if minus:
                        task['mirrored_task_id'] = f'{method}_c{i}_s{j}_plus'
                    tasks.append(dict(task=task, condition=c))
    write(args.out/'tasks.json', dict(protocol_sha256=report['protocol_sha256'], tasks=tasks))
    report['tasks_sha256'] = sha(args.out/'tasks.json')
    report['requested_attempts'] = len(tasks)
    def work(item):
        row = run_task(item['task'], item['condition'], binary, args.out, spec['xtb'])
        row.update(condition_index=item['task']['condition_index'], sample_index=item['task']['sample_index'])
        return row
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(work, item) for item in tasks]
        for future in as_completed(futures):
            report['rows'].append(future.result())
            if len(report['rows']) % 32 == 0:
                write(args.out/'results.json', report)
                print(json.dumps(dict(completed=len(report['rows']), requested=len(tasks))), flush=True)
    report['rows'].sort(key=lambda r:r['task_id'])
    report['attempted'] = len(report['rows'])
    report['successful'] = sum(r['success'] for r in report['rows'])
    assert report['attempted'] == len(tasks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['project', 'protocol', 'out']:
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--kind', choices=['esen', 'xtb'], required=True)
    parser.add_argument('--chunk', type=int, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    spec = json.loads(args.protocol.read_text())
    assert spec['frozen'] and 0 <= args.chunk < spec['chunks']
    assert not args.out.exists()
    conditions, references, samples = load(args, spec)
    args.out.mkdir(parents=True)
    start = time.perf_counter()
    report = dict(complete=False, protocol_sha256=sha(args.protocol), kind=args.kind,
        chunk=args.chunk, rows=[], new_neural_outputs=0, new_training_steps=0)
    write(args.out/'results.json', report)
    (esen if args.kind == 'esen' else xtb)(args, spec, conditions, references, samples, report)
    report.update(complete=True, seconds=time.perf_counter()-start)
    write(args.out/'results.json', report)


if __name__ == '__main__':
    main()
