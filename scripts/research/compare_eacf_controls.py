#!/usr/bin/env python3
"""Audit matched-source EACF cost-quality points against the convex control."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from cfm_mol.entropy_adapter_io import load_entropy_adapter
from cfm_mol.linear_entropy_adapter import endpoint_kl_change


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    values = np.asarray(values)
    return dict(mean=float(values.mean()), row_sem=float(values.std(ddof=1) / np.sqrt(len(values))),
                parents=len(values))


def update_interval(report):
    history = {row['step']: row for row in report['history']}
    return dict(begin=20, end=1000, seconds=history[1000]['seconds'] - history[20]['seconds'],
                seconds_per_attempt=(history[1000]['seconds'] - history[20]['seconds']) / 980)


def read_run(path):
    report = json.loads((path / 'results.json').read_text())
    submission = json.loads((path / 'submission.json').read_text())
    if not report['complete'] or report.get('engineering_only', False):
        raise ValueError('Require completed production output')
    for name, digest in report['artifacts'].items():
        if sha(path / name) != digest:
            raise ValueError(f'Changed artifact: {path / name}')
    return report, submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    rows, jobs = [], []
    for seed in [0, 1]:
        own = args.runs_root / f'parity_cpu_gpu_control_s{seed}_v1'
        r, submission = read_run(own)
        if not r['checkpoint_loader_replay_passed'] or r['neural_device'] != 'cpu':
            raise ValueError('Require the CPU-neural/GPU-oracle control')
        base = torch.load(own / 'base_samples.pt', map_location='cpu', weights_only=False)
        sample = torch.load(own / 'adapted_samples.pt', map_location='cpu', weights_only=False)
        model, _, _ = load_entropy_adapter(own, 'cpu')
        with torch.no_grad():
            for begin in range(0, 512, 64):
                y, volume = model(base['positions'][begin:begin+64])
                torch.testing.assert_close(y, sample['positions'][begin:begin+64], atol=1e-9, rtol=1e-9)
                torch.testing.assert_close(volume, sample['log_volume'][begin:begin+64], atol=1e-9, rtol=1e-9)
        change = endpoint_kl_change(base['energy_eV'], sample['energy_eV'], base['positions'], sample['positions'],
            kT=r['kT_eV'], restraint=r['restraint_eV_A2'], log_volume=sample['log_volume']).numpy()
        np.testing.assert_allclose(change, sample['paired_endpoint_kl_change'], atol=1e-9, rtol=0)
        jobs.append(submission['job_id'])
        for profile, stem, audit_stem in [('published', 'eacf_joint', 'eacf_joint_audit'),
                                          ('compact', 'eacf_compact', 'eacf_compact_audit')]:
            path = args.runs_root / f'{stem}_condition_00_s{seed}_v1'
            other, other_submission = read_run(path)
            audit_path = args.runs_root / f'{audit_stem}_s{seed}_v1/audit.json'
            audit = json.loads(audit_path.read_text())
            if (not audit['complete'] or audit['parents_replayed'] != 512
                    or audit['results_sha256'] != sha(path / 'results.json')):
                raise ValueError('Missing complete EACF replay')
            for key in ['condition', 'oracle_evaluations', 'source_checkpoint_sha256', 'source_results_sha256',
                        'evaluation_sha256', 'training_sha256', 'refinement_protocol_sha256', 'kT_eV',
                        'restraint_eV_A2', 'oracle_runtime']:
                if r[key] != other[key]:
                    raise ValueError(f'Comparison differs in {key}')
            saved = np.load(path / 'samples.npz')
            np.testing.assert_array_equal(saved['base_positions'], base['positions'].numpy())
            np.testing.assert_array_equal(saved['parent_ids'], base['sample_ids'])
            joint = (saved['base_log_target'] - saved['adapted_log_target'] + saved['base_aux_log_prob']
                     - saved['adapted_aux_log_prob'] - saved['log_volume'])
            np.testing.assert_allclose(joint, saved['paired_joint_kl_change'], atol=1e-9, rtol=0)
            jobs.append(other_submission['job_id'])
            rows.append(dict(profile=profile, seed=seed, convex_job=submission['job_id'],
                eacf_job=other_submission['job_id'], convex_parameters=r['parameters'],
                eacf_parameters=other['parameters'], queries_per_arm=r['oracle_evaluations'],
                convex_marginal_change=stats(change), eacf_joint_change=stats(joint),
                eacf_joint_minus_convex_marginal=stats(joint-change),
                convex_update_interval=update_interval(r), eacf_update_interval=update_interval(other),
                eacf_ignored_updates=other['ignored_updates'],
                convex_results_sha256=sha(own / 'results.json'), eacf_results_sha256=sha(path / 'results.json'),
                eacf_audit_sha256=sha(audit_path), convex_parents_replayed=512,
                eacf_parents_replayed=audit['parents_replayed']))
    raw = subprocess.check_output(['sacct', '-j', ','.join(sorted(set(jobs))), '-X', '-n', '-P',
        '--format=JobIDRaw,State,ElapsedRaw,NodeList,AllocTRES'], text=True)
    accounting = {row.split('|')[0]: row.split('|')[1:] for row in raw.strip().splitlines()}
    for row in rows:
        a, b = accounting[row['convex_job']], accounting[row['eacf_job']]
        if a[0] != 'COMPLETED' or b[0] != 'COMPLETED':
            raise ValueError('Require terminal successful allocation states')
        row.update(convex_allocation_seconds=int(a[1]), eacf_allocation_seconds=int(b[1]),
                   allocation_time_ratio=int(b[1])/int(a[1]))
    report = dict(complete=True, scope=__doc__, rows=rows, slurm_accounting=accounting,
        additional_oracle_evaluations=0, scientific_submission_ready=False,
        interpretation='Joint change upper-bounds marginal change in expectation. More-negative joint than convex marginal is adverse evidence for convex quality superiority; reverse ranking is inconclusive.',
        limitations=['One molecular condition and two training seeds; row SEM is not seed uncertainty.',
                     'No independent target-population reference or coverage certificate.',
                     'Different achieved quality; allocation ratios are not equal-quality speedups.',
                     'The update interval excludes initial compilation but is not an isolated neural benchmark.',
                     'Matched allocation/GPU class does not control physical CPU host or contention.',
                     'Compact14900 and convex19365 are similar-size controls, not identical parameter counts.'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for row in rows:
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
