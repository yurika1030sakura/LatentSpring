#!/usr/bin/env python3
"""Evaluate a fixed affine-context control, reusing the frozen convex comparison."""
import argparse
import json
from pathlib import Path
import time

import torch

from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.entropy_adapter_io import load_entropy_adapter
from cfm_mol.linear_entropy_adapter import endpoint_kl_change
from cfm_mol.nonequilibrium import WeightedPaths
from molecular_tempered_pilot import sha, write_json


def summarize(x):
    return {'mean': float(x.mean()), 'sem': float(x.std()/len(x)**.5), 'samples': len(x)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--affine-run', default='species_affine_adapter_1000_v1')
    p.add_argument('--convex-confirmation', default='species_entropy_adapter_confirmation_v1')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    output = args.out/'results.json'
    if output.exists():
        raise FileExistsError(output)
    start = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    base_dir = args.runs_root/'species_entropy_adapter_confirmation_v1/base'
    convex_dir = args.runs_root/args.convex_confirmation
    source = json.loads((base_dir/'results.json').read_text())
    convex_report = json.loads((convex_dir/'results.json').read_text())
    if not source['complete'] or not convex_report['complete'] or convex_report['base_sha256'] != sha(base_dir/'final_samples.pt'):
        raise ValueError('Incomplete or changed shared confirmation source')
    recipe = source['configuration']
    if recipe['seed'] != 9169 or recipe['eval_particles'] != 2048:
        raise ValueError('Require prescribed2048-row confirmation')
    convex_path = convex_dir/'nonlinear_samples.pt'
    if sha(convex_path) != convex_report['arms']['nonlinear']['samples_sha256']:
        raise ValueError('Stored nonlinear samples changed')
    model, trained, checkpoint_sha = load_entropy_adapter(args.runs_root/args.affine_run, args.device)
    if trained['kind'] != 'species_affine' or trained['oracle_evaluations'] != 16512:
        raise ValueError('Require matched1000-step affine control')
    prescribed = {'species_entropy_adapter_confirmation_v1': 'species_entropy_adapter_1000_v1',
        'species_entropy_adapter_replica_confirmation_v1': 'species_entropy_adapter_1000_s9162'}
    if args.convex_confirmation not in prescribed:
        raise ValueError('Require one of the two prespecified convex training pairs')
    convex_training_dir = args.runs_root/prescribed[args.convex_confirmation]
    convex_training = json.loads((convex_training_dir/'results.json').read_text())
    if (not convex_training['complete'] or sha(convex_training_dir/'adapter.ckpt') !=
            convex_report['arms']['nonlinear']['checkpoint_sha256']):
        raise ValueError('Convex training provenance changed')
    for key in ['source_checkpoint_sha256', 'oracle_sha256', 'condition', 'training_sha256', 'evaluation_sha256',
            'init_seed', 'selection_seed', 'steps', 'batch', 'lr_start', 'lr_end', 'parameters', 'adapter_configuration']:
        if trained[key] != convex_training[key]:
            raise ValueError(f'Matched neural-context control differs: {key}')
    if (trained['source_checkpoint_sha256'] != source['trained_checkpoint_sha256']
            or trained['oracle_sha256'] != source['oracle_sha256'] or trained['condition'] != source['condition']
            or trained['kT_eV'] != recipe['kT'] or trained['restraint_eV_A2'] != recipe['restraint']):
        raise ValueError('Source or target mismatch')
    oracle_path = Path(recipe['oracle'])
    if sha(oracle_path) != source['oracle_sha256']:
        raise ValueError('Physical oracle changed')
    base = torch.load(base_dir/'final_samples.pt', map_location='cpu', weights_only=False)
    convex = torch.load(convex_path, map_location='cpu', weights_only=False)
    x = base['positions'].double()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {'complete': False, 'scope': __doc__, 'condition': trained['condition'],
        'base_sha256': sha(base_dir/'final_samples.pt'), 'affine_checkpoint_sha256': checkpoint_sha,
        'affine_training_results_sha256': sha(args.runs_root/args.affine_run/'results.json'),
        'convex_results_sha256': sha(convex_dir/'results.json'), 'convex_samples_sha256': sha(convex_path),
        'source_checkpoint_sha256': trained['source_checkpoint_sha256'], 'oracle_sha256': trained['oracle_sha256'],
        'kT_eV': recipe['kT'], 'restraint_eV_A2': recipe['restraint'],
        'affine_run': args.affine_run, 'convex_confirmation': args.convex_confirmation,
        'limitations': ['One condition and one training pair on a reused2048-row panel; this is mechanism development.',
            'Both families have nonlinear geometry-conditioned full maps; only the active point transform is ablated.',
            'Row SEM does not measure training-seed uncertainty. No calibrated sampling or general novelty claim.']}
    write_json(output, report)
    with torch.no_grad(), EnergyOracle(Path(recipe['oracle_python']), root/'scripts/research/oracle_worker.py', oracle_path,
            numbers=trained['condition']['numbers'], charge=trained['condition']['charge'],
            spin_multiplicity=trained['condition']['spin_multiplicity'], batch_size=16) as oracle:
        outputs = [model(x[i:i+64].to(args.device)) for i in range(0, len(x), 64)]
        y = torch.cat([r[0].cpu() for r in outputs])
        volume = torch.cat([r[1].cpu() for r in outputs])
        restored, inverse_volume, info = model.inverse(y[:32].to(args.device))
        torch.testing.assert_close(restored.cpu(), x[:32], atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(inverse_volume.cpu(), -volume[:32], atol=1e-9, rtol=1e-9)
        energy, _ = oracle.evaluate(y)
        change = endpoint_kl_change(base['energy_eV'], energy, x, y,
            kT=recipe['kT'], restraint=recipe['restraint'], log_volume=volume)
        convex_change = endpoint_kl_change(base['energy_eV'], convex['energy_eV'], x, convex['positions'],
            kT=recipe['kT'], restraint=recipe['restraint'], log_volume=convex['log_volume'])
        torch.testing.assert_close(convex_change, convex['paired_endpoint_kl_change'], atol=1e-10, rtol=1e-10)
        work = base['work']+change
        torch.save({'positions': y, 'energy_eV': energy, 'log_volume': volume, 'work': work,
            'paired_endpoint_kl_change': change, 'condition': trained['condition']}, args.out/'affine_samples.pt')
        report.update(complete=True, affine_delta_kl=summarize(change), convex_delta_kl=summarize(convex_change),
            convex_minus_affine=summarize(convex_change-change), affine_inverse_check=info,
            affine_weights=WeightedPaths(y, -work, {}).summary(),
            new_oracle_queries=oracle.evaluated, reused_base_energy_rows=len(x), reused_convex_energy_rows=len(x),
            samples_sha256=sha(args.out/'affine_samples.pt'), seconds=time.perf_counter()-start)
    write_json(output, report)
    print(json.dumps({'affine_delta_kl': report['affine_delta_kl'], 'convex_minus_affine': report['convex_minus_affine']}), flush=True)


if __name__ == '__main__':
    main()
