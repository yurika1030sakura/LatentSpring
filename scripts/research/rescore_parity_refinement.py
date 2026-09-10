#!/usr/bin/env python3
"""New-target evaluation of fixed N8 maps under source inversion augmentation."""
import argparse
import json
from pathlib import Path

import torch

from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.linear_entropy_adapter import endpoint_kl_change
from cfm_mol.nonequilibrium import WeightedPaths
from cfm_mol.parity_refinement import parity_work_change, randomize_inversion
from molecular_tempered_pilot import sha, write_json


def summary(x):
    return {'mean': float(x.mean()), 'sem': float(x.std()/len(x)**.5), 'parents': len(x)}


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--runs-root', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
args = p.parse_args()
output = args.out/'results.json'
if output.exists():
    raise FileExistsError(output)
args.out.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[2]
base_dir = args.runs_root/'species_entropy_adapter_confirmation_v1/base'
source = json.loads((base_dir/'results.json').read_text())
if not source['complete'] or source['configuration']['eval_particles'] != 2048:
    raise ValueError('Require frozen2048-row source')
recipe = source['configuration']
specs = [('base', base_dir/'final_samples.pt', None)]
for pair, confirmation, affine in [
    (1, 'species_entropy_adapter_confirmation_v1', 'species_affine_confirmation_v1'),
    (2, 'species_entropy_adapter_replica_confirmation_v1', 'species_affine_replica_confirmation_v1'),
]:
    for label in ['linear', 'nonlinear']:
        directory = args.runs_root/confirmation
        r = json.loads((directory/'results.json').read_text())
        if not r['complete'] or r['base_sha256'] != sha(base_dir/'final_samples.pt'):
            raise ValueError('Raw-target comparison changed')
        specs.append((f'{label}_{pair}', directory/f'{label}_samples.pt', r['arms'][label]['samples_sha256']))
    directory = args.runs_root/affine
    r = json.loads((directory/'results.json').read_text())
    if not r['complete'] or r['base_sha256'] != sha(base_dir/'final_samples.pt'):
        raise ValueError('Affine comparison changed')
    specs.append((f'affine_{pair}', directory/'affine_samples.pt', r['samples_sha256']))
oracle_path = Path(recipe['oracle'])
if sha(oracle_path) != source['oracle_sha256']:
    raise ValueError('Raw potential checkpoint changed')
report = {'complete': False, 'scope': __doc__, 'condition': source['condition'], 'arms': {},
    'source_kind': 'uniform_inversion_mixture_of_original_frozen_path_source',
    'target_kind': 'inversion_energy_average', 'raw_oracle_sha256': source['oracle_sha256'],
    'kT_eV': recipe['kT'], 'restraint_eV_A2': recipe['restraint'], 'sign_seed': 9203,
    'limitations': ['New source/target protocol; no raw-potential result is retrospectively relabelled.',
        'Refinement KL compares the augmented source with its deterministic equivariant transport.',
        'Source augmentation has a separate unknown nonnegative KL gain toward this even target.',
        'One condition and shared2048 original parents; not fresh independent validation.',
        'The work auxiliary includes a uniform inversion sign; the original source need not be parity invariant.',
        'Repairing symmetry does not certify coverage, physical potential accuracy or novelty.']}
write_json(output, report)
outputs = {}
with EnergyOracle(Path(recipe['oracle_python']), root/'scripts/research/oracle_worker.py', oracle_path,
        numbers=source['condition']['numbers'], charge=source['condition']['charge'],
        spin_multiplicity=source['condition']['spin_multiplicity'], batch_size=16) as oracle, torch.no_grad():
    for label, path, expected in specs:
        if expected is not None and sha(path) != expected:
            raise ValueError('Stored raw-target sample artifact changed')
        data = torch.load(path, map_location='cpu', weights_only=False)
        if data['condition'] != source['condition'] or len(data['positions']) != 2048:
            raise ValueError('Physical condition or parent count differs')
        x = data['positions'].double()
        inverse_energy, _ = oracle.evaluate_chunked(-x)
        energy = .5*(data['energy_eV']+inverse_energy)
        work = parity_work_change(data['work'], data['energy_eV'], energy, kT=recipe['kT'])
        volume = data.get('log_volume', torch.zeros(len(x), dtype=torch.float64))
        sampled, signs = randomize_inversion(x, generator=torch.Generator().manual_seed(report['sign_seed']))
        value = {'positions': sampled, 'unflipped_positions': x, 'inversion_signs': signs,
            'energy_eV': energy, 'work': work, 'log_volume': volume, 'condition': source['condition']}
        if label == 'base':
            base = value
            change = torch.zeros(len(x), dtype=torch.float64)
        else:
            change = endpoint_kl_change(base['energy_eV'], energy, base['unflipped_positions'], x,
                kT=recipe['kT'], restraint=recipe['restraint'], log_volume=volume)
            torch.testing.assert_close(work-base['work'], change, atol=1e-9, rtol=1e-9)
        value['paired_endpoint_kl_change'] = change
        torch.save(value, args.out/f'{label}_samples.pt')
        outputs[label] = value
        report['arms'][label] = {'raw_samples_sha256': sha(path), 'samples_sha256': sha(args.out/f'{label}_samples.pt'),
            'paired_endpoint_kl_change': summary(change), 'path_weights': WeightedPaths(sampled, -work, {}).summary(),
            'raw_inversion_energy_difference_eV': summary(inverse_energy-data['energy_eV'])}
        report['new_oracle_evaluations_so_far'] = oracle.evaluated
        write_json(output, report)
        print(json.dumps({'arm': label, 'delta_kl': summary(change)}), flush=True)
    report['new_oracle_evaluations'] = oracle.evaluated
report['paired'] = {}
for pair in [1, 2]:
    for control in ['linear', 'affine']:
        difference = outputs[f'nonlinear_{pair}']['paired_endpoint_kl_change']-outputs[f'{control}_{pair}']['paired_endpoint_kl_change']
        report['paired'][f'nonlinear_minus_{control}_{pair}'] = summary(difference)
report['complete'] = True
write_json(output, report)
