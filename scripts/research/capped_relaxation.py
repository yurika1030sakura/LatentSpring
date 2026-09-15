#!/usr/bin/env python3
"""Separate capped-relaxation diagnostic; preserve all original raw outputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import BFGS
from scripts.research.evaluate_fresh_primary_xtb import run_task
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write
from scripts.research.audit_generator_output_support import assess


def aligned_rmsd(x, y):
    a = x - x.mean(0)
    b = y - y.mean(0)
    u, _, vh = np.linalg.svd(a.T @ b)
    sign = np.eye(3)
    sign[-1, -1] = np.linalg.det(u @ vh)
    return float(np.sqrt(np.mean(np.sum((a @ u @ sign @ vh - b)**2, axis=-1))))


def run_one(payload):
    task, spec, root = payload
    torch.set_num_threads(1)
    out = Path(root)/task['id']
    out.mkdir(parents=True)
    condition = task['condition']
    initial = np.asarray(task['positions'], dtype=float)
    cache = task['cached_singlepoint']
    calls = []
    if not cache['success']:
        graph = assess(torch.tensor(initial[None], dtype=torch.float64), condition, [task['sample_index']])
        raw = dict(step=0, positions=initial.tolist(), energy_eV=None, max_force_eV_A=None,
                   force_rms_eV_A=None, aligned_rmsd_A=0., graph=graph, converged=False)
        result = dict(id=task['id'], method=task['method'], seed=task['seed'], condition_index=task['condition_index'],
            sample_index=task['sample_index'], source_sample_sha256=task['source_sample_sha256'],
            cached_singlepoint_task_id=cache['task_id'], complete=True, optimizer_converged=False,
            error=dict(type='CachedSinglepointFailure', message=cache['failure']), actual_steps=0,
            new_singlepoint_attempts=0, failed_singlepoints=0, seconds=0.,
            stages={'0':raw, **{str(cap):None for cap in spec['step_caps']}})
        write(out/'result.json', result)
        return result

    class SinglePoint(Calculator):
        implemented_properties = ['energy', 'forces']

        def calculate(self, atoms=None, properties=('energy', 'forces'), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            x = atoms.get_positions()
            if np.array_equal(x, initial):
                result = cache
            else:
                query = dict(task_id=f'query_{len(calls):03}', method=task['method'], replica=task['seed'],
                    parent_id=task['sample_index'], inversion_check=False, positions=x.tolist())
                result = run_task(query, condition, Path(spec['xtb_binary']), out, spec['singlepoint'])
                calls.append(result)
                write(out/'queries.json', calls)
            if not result['success']:
                raise RuntimeError('xTB query failed: '+result['failure'])
            self.results = dict(energy=result['energy_eV'], forces=np.asarray(result['force_eV_A']))

    atoms = Atoms(numbers=condition['numbers'], positions=initial)
    atoms.calc = SinglePoint()
    frames = []
    optimizer = BFGS(atoms, logfile=str(out/'optimizer.log'), maxstep=spec['max_step_A'])

    def save_frame():
        x = atoms.get_positions()
        forces = atoms.get_forces()
        frame = dict(step=optimizer.nsteps, positions=x.tolist(), energy_eV=float(atoms.get_potential_energy()),
            max_force_eV_A=float(np.linalg.norm(forces, axis=-1).max()),
            force_rms_eV_A=float(np.sqrt(np.mean(np.sum(forces**2, axis=-1)))),
            aligned_rmsd_A=aligned_rmsd(x, initial))
        frames.append(frame)

    optimizer.attach(save_frame, interval=1)
    started = time.perf_counter()
    error = None
    try:
        converged = bool(optimizer.run(fmax=spec['force_tolerance_eV_A'], steps=max(spec['step_caps'])))
    except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        converged = False
        error = dict(type=type(exc).__name__, message=str(exc))
    stages = {}
    for cap in [0]+spec['step_caps']:
        selected = next((f for f in frames if f['step'] == cap), None)
        if selected is None and converged and frames and frames[-1]['step'] < cap:
            selected = frames[-1]
        if selected is None:
            stages[str(cap)] = None
            continue
        graph = assess(torch.tensor([selected['positions']], dtype=torch.float64), condition, [task['sample_index']])
        stages[str(cap)] = dict(selected, graph=graph, converged=selected['max_force_eV_A'] < spec['force_tolerance_eV_A'])
    assert stages['0'] is not None
    result = dict(id=task['id'], method=task['method'], seed=task['seed'], condition_index=task['condition_index'],
        sample_index=task['sample_index'], source_sample_sha256=task['source_sample_sha256'],
        cached_singlepoint_task_id=cache['task_id'], complete=True, optimizer_converged=converged,
        error=error, actual_steps=optimizer.nsteps, new_singlepoint_attempts=len(calls),
        failed_singlepoints=sum(not r['success'] for r in calls), seconds=time.perf_counter()-started, stages=stages)
    write(out/'trajectory.json', frames)
    write(out/'result.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ['project', 'protocol', 'out']:
        p.add_argument('--'+k, type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    spec = json.loads(a.protocol.read_text())
    assert spec['frozen'] and not a.out.exists() and sha(spec['xtb_binary']) == spec['xtb_binary_sha256']
    tasks = []
    for seed in [0, 1]:
        path = a.project/f'runs/fresh_physics_xtb_v1/s{seed}/xtb/results.json'
        assert sha(path) == spec['cached_xtb_sha256'][str(seed)]
        results = json.loads(path.read_text())
        by_id = {r['task_id']:r for r in results['rows']}
        for method in spec['methods']:
            report_path = a.project/f'runs/fresh_physics_v1/s{seed}/study/evaluation/{method}_results.json'
            assert sha(report_path) == spec['generation_report_sha256'][f'{seed}/{method}']
            expected = {r['condition_index']:r['sample_sha256'] for r in json.loads(report_path.read_text())['rows']}
            for i in range(24):
                source = a.project/f'runs/fresh_physics_v1/s{seed}/study/evaluation/{method}_c{i}.pt'
                assert sha(source) == expected[i]
                sample = torch.load(source, map_location='cpu', weights_only=False)
                for j in spec['sample_indices']:
                    cached = by_id[f'{method}_c{i}_s{j}']
                    tasks.append(dict(id=f's{seed}_{method}_c{i}_s{j}', seed=seed, method=method,
                        condition_index=i, sample_index=j, condition=sample['condition'], positions=sample['positions'][j].tolist(),
                        source_sample_sha256=sha(source), cached_singlepoint=cached))
    # One original-reference trajectory per composition, independent of model seeds.
    by_id = {r['task_id']:r for r in json.loads((a.project/'runs/fresh_physics_xtb_v1/s0/xtb/results.json').read_text())['rows']}
    for i in range(24):
        source = a.project/f'runs/fresh_physics_v1/s0/study/physical_eval/reference_c{i}.pt'
        sample = torch.load(source, map_location='cpu', weights_only=False)
        condition = tasks[i*len(spec['sample_indices'])]['condition']
        tasks.append(dict(id=f'reference_c{i}', seed=0, method='reference', condition_index=i, sample_index=-1,
            condition=condition, positions=sample['positions'].tolist(), source_sample_sha256=sha(source),
            cached_singlepoint=by_id[f'reference_c{i}_plus']))
    assert len(tasks) == spec['trajectories']
    a.out.mkdir(parents=True)
    write(a.out/'tasks.json', tasks)
    report = dict(complete=False, protocol_sha256=sha(a.protocol), tasks_sha256=sha(a.out/'tasks.json'),
        rows=[], scientific_submission_ready=False, scope=spec['scope'])
    with ProcessPoolExecutor(max_workers=spec['workers']) as pool:
        futures = [pool.submit(run_one, (t, spec, str(a.out/'trajectories'))) for t in tasks]
        for future in as_completed(futures):
            report['rows'].append(future.result())
            write(a.out/'results.json', report)
            if len(report['rows']) % 24 == 0:
                print(json.dumps(dict(completed=len(report['rows']), total=len(tasks))), flush=True)
    report['rows'].sort(key=lambda r:r['id'])
    report.update(complete=True, trajectories=len(tasks), new_raw_singlepoint_attempts=sum(r['new_singlepoint_attempts'] for r in report['rows']))
    write(a.out/'results.json', report)


if __name__ == '__main__':
    main()
