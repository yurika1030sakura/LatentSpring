#!/usr/bin/env python3
"""Independent real-potential force/energy and symmetry checks on fixed source rows."""
import argparse
import json
import math
from pathlib import Path

import torch

from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.entropy_source import load_entropy_source
from cfm_mol.parity_refinement import evaluate_even_potential
from molecular_tempered_pilot import sha, write_json


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-root', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--even-target', action='store_true')
p.add_argument('--fd-steps', type=float, nargs='+', default=[.001, .0003])
args = p.parse_args()
root = Path(__file__).resolve().parents[2]
output = args.out/'results.json'
if output.exists():
    raise FileExistsError(output)
args.out.mkdir(parents=True, exist_ok=True)
if any(step <= 0 or not math.isfinite(step) for step in args.fd_steps):
    raise ValueError('Positive finite difference steps required')
report = {'complete': False, 'scope': __doc__, 'rows': [], 'source_indices': [0, 7, 15, 31],
    'fd_steps_A': args.fd_steps, 'seed': 9193,
    'potential': 'inversion_energy_average' if args.even_target else 'raw_esen',
    'thresholds': {'energy_symmetry_eV': .0001, 'force_symmetry_relative': .001,
        'fd_absolute_eV_A': .01, 'fd_relative': .001},
    'limitations': ['Four fixed source geometries per condition; not a global smoothness or physical-accuracy certificate.',
        'These checks concern the defined ML potential and its actual numerical interface.',
        'No equilibrium sampling or method advantage follows from this engineering audit.']}
write_json(output, report)
angle = .731
rotation = torch.tensor([[math.cos(angle), -math.sin(angle), 0.],
    [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]], dtype=torch.float64)
reflection = torch.diag(torch.tensor([-1., 1., 1.], dtype=torch.float64))
for index in range(8):
    source = load_entropy_source(args.source_root, index, engineering=True,
        protocol_path=root/'research/evidence/species_breadth_source_protocol_v2.json',
        manifest_path=root/'research/evidence/development_panel_v1.json')
    condition, recipe = source['condition'], source['recipe']
    x = source['training']['positions'][report['source_indices']]
    generator = torch.Generator().manual_seed(report['seed']+index)
    direction = torch.randn(x.shape, dtype=torch.float64, generator=generator)
    direction -= direction.mean(1, keepdim=True)
    direction /= direction.square().sum((1, 2), keepdim=True).sqrt()
    oracle_path = Path(recipe['oracle'])
    if sha(oracle_path) != source['source']['oracle_sha256']:
        raise ValueError('Physical checkpoint changed')
    row = {'index': index, 'condition': condition, 'source_results_sha256': source['source_results_sha256'],
        'symmetry': {}, 'finite_differences': {}}
    with EnergyOracle(Path(recipe['oracle_python']), root/'scripts/research/oracle_worker.py', oracle_path,
            numbers=condition['numbers'], charge=condition['charge'],
            spin_multiplicity=condition['spin_multiplicity'], batch_size=16) as oracle:
        if args.even_target:
            def evaluate(value):
                energy, force, _ = evaluate_even_potential(oracle, value)
                return energy, force
            energy, force, components = evaluate_even_potential(oracle, x)
            replay_energy = components['raw_energy_eV']
        else:
            evaluate = oracle.evaluate
            energy, force = evaluate(x)
            replay_energy = energy
        row['cached_raw_energy_max_error_eV'] = float((replay_energy-source['training']['energy_eV'][report['source_indices']]).abs().max())
        for name, transform in [('rotation', rotation), ('reflection', reflection)]:
            new_energy, new_force = evaluate(x@transform)
            residual = (new_force-force@transform).reshape(len(x), -1).norm(dim=-1)
            relative = residual/(force.reshape(len(x), -1).norm(dim=-1)+1.)
            row['symmetry'][name] = {'energy_difference_eV': (new_energy-energy).tolist(),
                'force_residual_norm_eV_A': residual.tolist(), 'force_residual_relative': relative.tolist()}
        order = torch.arange(x.shape[1]-1, -1, -1)
        oracle.condition['numbers'] = [condition['numbers'][i] for i in order.tolist()]
        new_energy, new_force = evaluate(x[:, order])
        residual = (new_force-force[:, order]).reshape(len(x), -1).norm(dim=-1)
        row['symmetry']['joint_permutation'] = {'energy_difference_eV': (new_energy-energy).tolist(),
            'force_residual_norm_eV_A': residual.tolist(),
            'force_residual_relative': (residual/(force.reshape(len(x), -1).norm(dim=-1)+1.)).tolist()}
        oracle.condition['numbers'] = condition['numbers']
        derivative = -(force*direction).sum((1, 2))
        for step in report['fd_steps_A']:
            plus, _ = evaluate(x+step*direction)
            minus, _ = evaluate(x-step*direction)
            fd = (plus-minus)/(2*step)
            row['finite_differences'][str(step)] = {'analytic_eV_A': derivative.tolist(),
                'central_fd_eV_A': fd.tolist(), 'error_eV_A': (fd-derivative).tolist()}
        row['oracle_evaluations'] = oracle.evaluated
        row['oracle_requested_evaluations'] = oracle.requested_evaluations
    row['symmetry_checks_passed'] = all(max(map(abs, r['energy_difference_eV'])) <= .0001
        and max(r['force_residual_relative']) <= .001 for r in row['symmetry'].values())
    row['fd_checks_passed'] = all(all(abs(error) <= .01+.001*abs(exact)
        for error, exact in zip(r['error_eV_A'], r['analytic_eV_A'])) for r in row['finite_differences'].values())
    report['rows'].append(row)
    write_json(output, report)
    print(json.dumps({'index': index, 'symmetry_passed': row['symmetry_checks_passed'], 'fd_passed': row['fd_checks_passed']}), flush=True)
report.update(complete=True, oracle_evaluations=sum(r['oracle_evaluations'] for r in report['rows']),
    all_checked_cases_passed=all(r['symmetry_checks_passed'] and r['fd_checks_passed'] for r in report['rows']))
write_json(output, report)
