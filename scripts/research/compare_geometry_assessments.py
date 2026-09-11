#!/usr/bin/env python3
"""Join immutable geometry assessments on their exact prespecified sample subset."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from assess_work_panel import paired_outcomes, quantiles
from eval_position_xtb import ENERGY, HARTREE_EV


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assessment', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    common = None
    rows, geometry, summaries, inputs, raw_hashes = [], [], [], [], {}
    names = set()
    for path in args.assessment:
        report = json.loads(path.read_text())
        if not report['complete'] or not report['shared_parent_contract_verified']:
            raise ValueError('Require a complete paired assessment')
        settings = {key: report[key] for key in ['condition', 'identity_namespace', 'identity_index',
            'xtb_subset_indices', 'subset_seed', 'xtb_sha256', 'contact_factor', 'overlap_factor', 'radii_source']}
        settings['assessor_source_hashes'] = {Path(k).name: v for k, v in report['source_sha256'].items()}
        if common is None:
            common = settings
        elif settings != common:
            raise ValueError('Different physical conditions, subset, evaluator or binary')
        selected = report['xtb_subset_indices']
        source_names = {s['arm'] for s in report['sources']}
        if names & source_names:
            raise ValueError('Duplicate arm name across assessments')
        names |= source_names
        if report['requested_xtb'] != len(source_names) * len(selected):
            raise ValueError('Prescribed denominator differs')
        for source in report['sources']:
            samples_path = Path(source['samples'])
            if (sha(samples_path) != source['samples_sha256']
                    or sha(samples_path.parent / 'results.json') != source['results_sha256']):
                raise ValueError('Assessed source artifact changed')
            saved = torch.load(samples_path, map_location='cpu', weights_only=False)
            arm_rows = [r for r in report['xtb_rows'] if r['arm'] == source['arm']]
            if sorted(r['sample_id'] for r in arm_rows) != selected:
                raise ValueError('Missing or duplicated attempted sample')
            for row in arm_rows:
                if (row['charge_recorded'] != common['condition']['charge']
                        or row['spin'] != common['condition']['spin_multiplicity']
                        or not row['spin_metadata_available']):
                    raise ValueError('Electronic state differs')
                folder = path.parent / 'details' / f"{row['arm']}_{row['validation_index']}_{row['sample_id']}"
                xyz_path = folder / 'input.xyz'
                xyz = np.asarray([[float(v) for v in line.split()[1:]]
                                  for line in xyz_path.read_text().splitlines()[2:]])
                np.testing.assert_allclose(xyz, saved['positions'][row['sample_id']], atol=1e-9, rtol=1e-11)
                for name in ['input.xyz', 'single_point.stdout', 'single_point.stderr',
                             'relaxation.stdout', 'relaxation.stderr']:
                    log = folder / name
                    if log.exists():
                        raw_hashes[str(log)] = sha(log)
                if row['success']:
                    sp = (folder / 'single_point.stdout').read_text()
                    opt = (folder / 'relaxation.stdout').read_text()
                    if 'GEOMETRY OPTIMIZATION CONVERGED' not in opt.upper():
                        raise ValueError('Successful row lacks optimizer convergence')
                    energies = [float(ENERGY.findall(text)[-1].replace('D', 'E')) * HARTREE_EV
                                for text in [sp, opt]]
                    np.testing.assert_allclose(energies,
                        [row['initial_energy_eV'], row['relaxed_energy_eV']], atol=1e-9, rtol=0)
                    np.testing.assert_allclose(energies[0] - energies[1], row['strain_eV'], atol=1e-9, rtol=0)
            successful = [r for r in arm_rows if r['success']]
            summary = next(s for s in report['xtb_summaries'] if s['arm'] == source['arm'])
            if summary['attempted'] != len(arm_rows) or summary['converged'] != len(successful):
                raise ValueError('Summary denominator differs')
            computed = quantiles([r['strain_eV'] for r in successful])
            if computed != summary['strain_eV_success_only']:
                raise ValueError('Success-only quantiles disagree')
            failures = {}
            for row in arm_rows:
                if not row['success']:
                    failure = row.get('failure', 'unspecified')
                    failures[failure] = failures.get(failure, 0) + 1
            summaries.append({**summary, 'failures': failures})
        inputs.append(dict(path=str(path), sha256=sha(path), attempts=report['requested_xtb']))
        rows.extend(report['xtb_rows'])
        geometry.extend({k: v for k, v in g.items() if k != 'contact_rows'} for g in report['geometry'])
    paired = []
    for first, second in itertools.combinations(sorted(names), 2):
        a = [r for r in rows if r['arm'] == first]
        b = [r for r in rows if r['arm'] == second]
        by_id = {r['sample_id']: r for r in b}
        changes = [by_id[r['sample_id']]['strain_eV'] - r['strain_eV'] for r in a
                   if r['success'] and by_id[r['sample_id']]['success']]
        paired.append(dict(before=first, after=second, **paired_outcomes(a, b),
                           both_successful=len(changes), strain_change_both_successful=quantiles(changes)))
    result = dict(complete=True, scope=__doc__, settings=common, assessments=inputs,
        attempts=len(rows), converged=sum(r['success'] for r in rows),
        failed=sum(not r['success'] for r in rows), summaries=summaries, paired=paired,
        geometry=geometry, raw_log_hashes=raw_hashes, additional_oracle_evaluations=0,
        additional_xtb_evaluations=0, scientific_submission_ready=False,
        limitations=['Strain is not chemical identity/validity or target-population coverage.',
                     'Success-only statistics must accompany all failure denominators.',
                     'One condition and32 selected parents; no broad performance certification.'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(attempts=result['attempts'], converged=result['converged'], failed=result['failed'])))


if __name__ == '__main__':
    main()
