#!/usr/bin/env python3
"""Fresh shared-base confirmation of fixed linear entropy adapters."""
import argparse
import json
from pathlib import Path
import time

import torch

from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.linear_entropy_adapter import LinearEntropyAdapter, endpoint_kl_change
from cfm_mol.nonequilibrium import WeightedPaths
from molecular_tempered_pilot import sha, write_json


def summarize(x):return {'mean': float(x.mean()), 'sem': float(x.std()/len(x)**.5), 'samples': len(x)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root', type=Path, required=True); p.add_argument('--base-run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--device', default='cuda'); args = p.parse_args()
    output = args.out/'results.json'
    if output.exists():raise FileExistsError(output)
    args.out.mkdir(parents=True, exist_ok=True); start = time.perf_counter()
    base_report = json.loads((args.base_run/'results.json').read_text())
    if not base_report['complete'] or base_report['configuration']['seed'] != 9147 or base_report['configuration']['eval_particles'] != 1024:
        raise ValueError('Require declared fresh1024-path stream')
    data = torch.load(str(args.base_run/'final_samples.pt'), map_location='cpu', weights_only=False)
    recipe = base_report['configuration']; condition = data['condition']; x = data['positions'].double()
    report = {'complete': False, 'scope': __doc__, 'condition': condition, 'base_sha256': sha(args.base_run/'final_samples.pt'),
        'base_results_sha256': sha(args.base_run/'results.json'), 'source_checkpoint_sha256': base_report['trained_checkpoint_sha256'],
        'particles': len(x), 'seed': 9147, 'arms': {},
        'limitations': ['One condition and one trained adapter seed; this confirms evaluation, not training replication.',
            'KL changes are relative to the same fixed base; absolute KL and target coverage are unknown.',
            'Fresh path weights use the original trained reverse model, not the separately refitted development auxiliary.',
            'The adapters are established exact-volume refinement controls, not validated AI novelty.']}
    write_json(output, report); root = Path(__file__).resolve().parents[2]
    oracle_path = Path(recipe['oracle'])
    if sha(oracle_path) != base_report['oracle_sha256']:raise ValueError('Target oracle changed')
    outputs = {}
    with EnergyOracle(Path(recipe['oracle_python']), root/'scripts/research/oracle_worker.py', oracle_path,
        numbers=condition['numbers'], charge=condition['charge'], spin_multiplicity=condition['spin_multiplicity'], batch_size=16) as oracle, torch.no_grad():
        for kind in ['typed', 'scalar']:
            directory = args.runs_root/f'linear_entropy_adapter_{kind}_v1'
            old = json.loads((directory/'results.json').read_text()); checkpoint = directory/'adapter.ckpt'
            if not old['complete'] or old['source_checkpoint_sha256'] != base_report['trained_checkpoint_sha256'] or old['condition'] != condition:
                raise ValueError('Adapter and base source differ')
            if old['kT_eV'] != recipe['kT'] or old['restraint_eV_A2'] != recipe['restraint'] or old['oracle_sha256'] != base_report['oracle_sha256']:
                raise ValueError('Adapter physical target differs')
            if sha(checkpoint) != old['artifacts']['adapter.ckpt']:raise ValueError('Adapter checkpoint changed')
            state = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
            model = LinearEntropyAdapter(condition['numbers'], kind=kind).to(args.device)
            model.load_state_dict(state['state_dict'], strict=True)
            y, volume = model(x.to(args.device)); y = y.cpu(); volume = float(volume)
            energy, _ = oracle.evaluate(y)
            change = endpoint_kl_change(data['energy_eV'], energy, x, y, kT=recipe['kT'], restraint=recipe['restraint'], log_volume=volume)
            work = data['work']+change
            outputs[kind] = {'positions': y, 'energy_eV': energy, 'work': work, 'paired_endpoint_kl_change': change,
                'condition': condition, 'log_volume': volume}
            torch.save(outputs[kind], args.out/f'{kind}_samples.pt')
            report['arms'][kind] = {'paired_endpoint_kl_change': summarize(change), 'weights': WeightedPaths(y, -work, {}).summary(),
                'adapter_sha256': sha(checkpoint), 'samples_sha256': sha(args.out/f'{kind}_samples.pt'), 'log_volume': volume}
            write_json(output, report)
        report['new_adapter_oracle_queries'] = oracle.evaluated
    difference = endpoint_kl_change(outputs['scalar']['energy_eV'], outputs['typed']['energy_eV'], outputs['scalar']['positions'],
        outputs['typed']['positions'], kT=recipe['kT'], restraint=recipe['restraint'],
        log_volume=outputs['typed']['log_volume']-outputs['scalar']['log_volume'])
    report.update(complete=True, typed_minus_scalar_endpoint_kl=summarize(difference),
        base_weights=WeightedPaths(x, -data['work'], {}).summary(),
        total_confirmation_oracle_queries=base_report['oracle_evaluations']+report['new_adapter_oracle_queries'], seconds=time.perf_counter()-start)
    write_json(output, report); print(json.dumps({'arms': report['arms'], 'typed_minus_scalar': report['typed_minus_scalar_endpoint_kl']}), flush=True)


if __name__ == '__main__':main()
