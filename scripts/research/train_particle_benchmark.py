#!/usr/bin/env python3
"""Exact-entropy adapter on the standard DW-4 / LJ-13 Boltzmann benchmarks.

Reports finite-sample reverse ESS using exact labelled density ratios. Each
family records its own symmetry contract. Fixed-index convex/affine controls
are diagnostics; the whole-element and global-pair families retain permutation
equivariance. No molecular or ICLR performance qualification follows here.

Annealing acts on the TRAINING PATH ONLY. Every reported number uses the
benchmark temperature tau = 1.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from cfm_mol.benchmarks.harness import (CentredGaussianSource, effective_sample_size,
    reverse_diagnostics)
from cfm_mol.benchmarks.particle_systems import PARAMETERS, reduced_energy
from cfm_mol.species_coupling_adapter import SpeciesCouplingAdapter
from cfm_mol.affine_species_adapter import AffineSpeciesCouplingAdapter
from cfm_mol.entropy_adapter_io import build_species_adapter
from cfm_mol.equivariant_pair_adapter import EquivariantPairAdapter


def write_report(path, report):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(report, indent=1, allow_nan=False))
    temporary.replace(path)


def annealed_tau(step, steps, start, fraction):
    """Geometric ladder from `start` down to tau = 1, then constant."""
    if start is None:
        return 1.
    if not math.isfinite(start) or start < 1. or not 0 < fraction <= 1:
        raise ValueError('Anneal must start at tau >= 1 with fraction in (0, 1]')
    if steps < 1 or not 0 <= step < steps:
        raise ValueError('Invalid update index or step count')
    if steps == 1:
        return 1.
    span = min(steps-1, max(1, int(round(fraction*steps))))
    return start**(1.-min(1., step/span))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--system', choices=['dw4', 'lj13'], required=True)
    p.add_argument('--kind', choices=['convex', 'affine', 'whole_convex', 'whole_affine', 'pair', 'pair_affine'], default='convex')
    p.add_argument('--sweeps', type=int, default=4)
    p.add_argument('--pair-curvature-bound',type=float,default=.25)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--scale', type=float, default=None, help='source Gaussian scale')
    p.add_argument('--anneal-from-tau', type=float, default=None)
    p.add_argument('--anneal-fraction', type=float, default=.5)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--eval-samples', type=int, default=8192)
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if min(args.steps, args.batch, args.eval_samples) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('Positive finite optimization and sample settings required')
    annealed_tau(0, args.steps, args.anneal_from_tau, args.anneal_fraction)
    if not args.kind.startswith('pair') and args.pair_curvature_bound!=.25:
        raise ValueError('Pair curvature option cannot modify a different family')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Atomic reservation: two launchers must never share a results destination.
    with args.out.open('x') as reserved:
        reserved.write(json.dumps({'complete': False, 'state': 'reserved'}))

    n = PARAMETERS[args.system]['n_particles']
    scale = args.scale if args.scale is not None else {'dw4': 1.6, 'lj13': 1.0}[args.system]
    torch.manual_seed(args.seed)
    source = CentredGaussianSource(n, scale=scale, device=args.device)
    if args.kind.startswith('pair'):
        adapter=EquivariantPairAdapter([6]*n,charge=0,spin_multiplicity=1,kT=1.,sweeps=args.sweeps,
                                      affine=args.kind=='pair_affine',curvature_bound=args.pair_curvature_bound)
    else:
        model_class=AffineSpeciesCouplingAdapter if args.kind.endswith('affine') else SpeciesCouplingAdapter
        adapter=model_class([6]*n,charge=0,spin_multiplicity=1,kT=1.,sweeps=args.sweeps,
                              split_groups=not args.kind.startswith('whole_'))
    adapter = adapter.to(args.device).double()
    optimiser = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.)
    train_gen = torch.Generator().manual_seed(args.seed+1000)
    eval_gen = torch.Generator().manual_seed(args.seed+2000)

    report = {'complete': False, 'scope': __doc__, 'system': args.system,
        'parameters': PARAMETERS[args.system], 'configuration': vars(args) | {'out': str(args.out)},
        'source': {'kind': 'centred_isotropic_gaussian', 'scale': scale,
                   'dimension': source.dimension, 'note': 'Declared diagnostic scale; no linked scale-selection evidence'},
        'adapter_configuration': adapter.configuration,
        'permutation_equivariant': adapter.permutation_equivariant,
        'target_definition': 'unordered pair reference energy with no distance floor; DW tau=1, LJ tau=1',
        'scientific_submission_ready': False,
        'target_energy_evaluations': 0,
        'anneal_scope': 'training path only; every reported number uses tau = 1',
        'n_parameters': sum(q.numel() for q in adapter.parameters()),
        'history': [], 'limitations': [
            'Reverse ESS on model samples does not certify mode coverage.',
            'A Gaussian source is not a pretrained molecular generator.',
            'Finite-sample ESS does not have zero uncertainty.',
            'The singular LJ energy may have infinite expectation under a smooth positive proposal density.']}
    if not adapter.permutation_equivariant:
        report['limitations'].append('Fixed index halves break permutation equivariance and retain each half centroid.')
    elif args.kind.startswith('pair'):
        report['limitations'].append('Pair-potential prototype uses dense3N determinants and a finite radial basis; no universality or novelty is established.')
    else:
        report['limitations'].append('Whole-element contexts have limited angular expressivity on homogeneous systems.')
    source_root=Path(__file__).resolve().parents[2]
    report['source_sha256']={str(p.relative_to(source_root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__).resolve(), source_root/'cfm_mol/benchmarks/particle_systems.py',
                  source_root/'cfm_mol/benchmarks/harness.py', source_root/'cfm_mol/species_coupling_adapter.py',
                  source_root/'cfm_mol/affine_species_adapter.py', source_root/'cfm_mol/centered_convex_flow.py',
                  source_root/'cfm_mol/equivariant_pair_adapter.py']}
    write_report(args.out, report)

    initial = reverse_diagnostics(args.system, source, adapter, args.eval_samples,
                                  generator=torch.Generator().manual_seed(args.seed+2000),
                                  device=args.device)
    report['initial'] = initial
    report['target_energy_evaluations'] += initial['target_energy_evaluations']
    start = time.perf_counter()
    for step in range(args.steps):
        tau = annealed_tau(step, args.steps, args.anneal_from_tau, args.anneal_fraction)
        x = source.sample(args.batch, generator=train_gen).to(args.device)
        optimiser.zero_grad(set_to_none=True)
        y, volume = adapter(x)
        loss = (reduced_energy(args.system, y)/tau-volume).mean()
        report['target_energy_evaluations'] += args.batch
        if not torch.isfinite(loss):
            report['failure'] = f'nonfinite objective at step {step}'
            write_report(args.out, report)
            raise FloatingPointError(report['failure'])
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 100.)
        if not torch.isfinite(gradient):
            report['failure'] = f'nonfinite gradient at step {step}'
            write_report(args.out, report)
            raise FloatingPointError(report['failure'])
        optimiser.step()
        if step == 0 or (step+1) % 100 == 0:
            row = {'step': step+1, 'objective': float(loss.detach()),
                   'training_tau': tau, 'mean_log_volume': float(volume.detach().mean()),
                   'gradient_norm': float(gradient), 'seconds': time.perf_counter()-start}
            report['history'].append(row)
            write_report(args.out, report)
            print(json.dumps(row), flush=True)

    samples_path=args.out.with_suffix('.samples.pt')
    report['final'] = reverse_diagnostics(args.system, source, adapter, args.eval_samples,
                                          generator=eval_gen, device=args.device, samples_out=samples_path)
    report['target_energy_evaluations'] += report['final']['target_energy_evaluations']
    checkpoint=args.out.with_suffix('.adapter.pt')
    torch.save({'state_dict':adapter.state_dict(),'optimizer_state_dict':optimiser.state_dict(),
                'configuration':adapter.configuration,'kind':args.kind,'source':report['source'],
                'target_definition':report['target_definition'],'source_sha256':report['source_sha256']}, checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    loaded=(EquivariantPairAdapter([6]*n,**saved['configuration']) if args.kind.startswith('pair') else
            build_species_adapter([6]*n,saved['configuration'],affine=args.kind.endswith('affine'))).double()
    loaded.load_state_dict(saved['state_dict'])
    panel=torch.load(samples_path,map_location='cpu',weights_only=False)
    with torch.no_grad():
        positions,log_volume=loaded(panel['parent_positions'][:16])
    torch.testing.assert_close(positions,panel['positions'][:16],atol=1e-9,rtol=1e-9)
    torch.testing.assert_close(log_volume,panel['log_volume'][:16],atol=1e-9,rtol=1e-9)
    for path,digest in report['source_sha256'].items():
        if hashlib.sha256((source_root/path).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Source changed during training; run an immutable snapshot')
    report['checkpoint_replay_passed']=True
    report['artifacts']={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in [checkpoint,samples_path]}
    report['seconds'] = time.perf_counter()-start
    report['complete'] = True
    write_report(args.out, report)
    print(json.dumps({'system': args.system, 'kind': args.kind, 'sweeps': args.sweeps,
        'base_ess': report['initial']['base']['ess'],
        'final_ess': report['final']['adapted']['ess']}), flush=True)


if __name__ == '__main__':
    main()
