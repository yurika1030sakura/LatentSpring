#!/usr/bin/env python3
"""Audit every prespecified corrected-target training arm and paired contrast."""
import argparse
import json
from pathlib import Path

import torch

from cfm_mol.entropy_adapter_io import load_entropy_adapter
from cfm_mol.entropy_source import file_sha
from cfm_mol.linear_entropy_adapter import endpoint_kl_change
from molecular_tempered_pilot import write_json


def summarize(x):
    return {'mean': float(x.mean()), 'sem': float(x.std()/len(x)**.5), 'independent_parents': len(x)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--require-complete', action='store_true')
    p.add_argument('--engineering-smoke', action='store_true', help='Audit the actual24-arm engineering panel; never label it production')
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    root = Path(__file__).resolve().parents[2]
    protocol_path = root/'research/evidence/parity_training_protocol_v1.json'
    protocol = json.loads(protocol_path.read_text())
    replicas = [0] if args.engineering_smoke else [0, 1]
    parent_count = 16 if args.engineering_smoke else 512
    expected_queries = 96 if args.engineering_smoke else protocol['oracle_evaluations_per_production_arm']
    report = {'complete': False, 'scope': __doc__, 'protocol_sha256': file_sha(protocol_path),
        'expected_conditions': 8, 'expected_arms': 8*3*len(replicas), 'conditions': [],
        'engineering_only': args.engineering_smoke,
        'all_arms_terminal': False, 'scientific_submission_ready': False,
        'limitations': ['Completion describes this audit, not scientific readiness.',
            'Missing terminal ledger entries are unresolved; inspect live Slurm to distinguish pending, running or interrupted work.',
            'Each512-row paired contrast conditions on trained maps; rows are shared across seeds.',
            'No absolute density, importance ESS or calibrated population claim is available for this source.',
            'Source generation, failed branches, pretraining and independent assessments cost additional compute.']}
    for index in range(8):
        directory = (args.runs_root/'parity_entropy_condition_00_v1' if index == 0 else
            args.runs_root/'parity_entropy_conditions_1_7_v1'/f'condition_{index:02d}')
        if args.engineering_smoke:
            directory = args.runs_root/'parity_entropy_smoke_v1'
        ledger_path = directory/'panel.json'
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'rows': []}
        completed = {}
        condition_report = {'index': index, 'arms': [], 'paired': {}}
        for replica in replicas:
            for kind in ['convex', 'affine', 'typed']:
                name = f'condition_{index:02d}_{kind}_s{replica}'
                row = {'kind': kind, 'replica': replica, 'run': str(directory/name), 'state': 'unresolved'}
                entries = [v for v in ledger['rows'] if v['run'] == name]
                if len(entries) > 1:
                    raise ValueError('Duplicate arm ledger entries')
                result_path = directory/name/'results.json'
                if entries and not entries[0]['success']:
                    row.update(state='failed', failure=entries[0].get('failure'),
                        exit_code=entries[0]['exit_code'], oracle_evaluations=entries[0].get('oracle_evaluations'),
                        log_sha256=entries[0].get('log_sha256'))
                elif entries:
                    trained = json.loads(result_path.read_text())
                    if not trained['complete'] or not trained.get('checkpoint_loader_replay_passed'):
                        raise ValueError('Success ledger disagrees with checkpoint qualification')
                    if file_sha(result_path) != entries[0]['results_sha256']:
                        raise ValueError('Completed training result changed')
                    expected_kind = 'typed' if kind == 'typed' else 'species_'+kind
                    expected_recipe = {'kind': expected_kind, 'steps': 2 if args.engineering_smoke else 1000,
                        'batch': 16, 'eval_count': parent_count, 'init_seed': 9161+replica,
                        'selection_seed': 9141+replica, 'orientation_seed': 9205+replica,
                        'evaluation_sign_seed': 9207+index, 'reference_geometry_used_to_initialize': False,
                        'reference_energy_used_for_training': False}
                    if any(trained.get(key) != value for key, value in expected_recipe.items()):
                        raise ValueError('Prescribed model, optimizer stream or evaluation recipe differs')
                    if (trained['source_kind'] != protocol['source_kind'] or trained['target_kind'] != protocol['target_kind']
                            or trained['refinement_protocol_sha256'] != report['protocol_sha256']
                            or trained['source_protocol_sha256'] != protocol['source_protocol_sha256']
                            or trained['oracle_sha256'] != protocol['raw_oracle_sha256']
                            or trained['kT_eV'] != protocol['kT_eV']
                            or trained['restraint_eV_A2'] != protocol['restraint_eV_A2']
                            or trained['oracle_evaluations'] != expected_queries
                            or trained['engineering_only'] != args.engineering_smoke
                            or trained['condition']['manifest_index'] != index or trained['replica'] != replica):
                        raise ValueError('Source/target/condition or oracle budget differs')
                    for filename, digest in trained['artifacts'].items():
                        if file_sha(directory/name/filename) != digest:
                            raise ValueError('A completed model artifact changed')
                    base = torch.load(directory/name/'base_samples.pt', map_location='cpu', weights_only=False)
                    samples = torch.load(directory/name/'adapted_samples.pt', map_location='cpu', weights_only=False)
                    if len(base['positions']) != parent_count or base['condition'] != samples['condition']:
                        raise ValueError('Evaluation parent count or electronic condition differs')
                    if any(key in samples for key in ['work', 'importance_weights']):
                        raise ValueError('This source has no qualified importance weights')
                    signs = base['inversion_signs']
                    expected_signs = 2*torch.randint(2, (parent_count,), generator=torch.Generator().manual_seed(9207+index))-1
                    torch.testing.assert_close(signs, expected_signs, atol=0, rtol=0)
                    torch.testing.assert_close(signs, samples['inversion_signs'], atol=0, rtol=0)
                    torch.testing.assert_close(base['positions'], base['unflipped_positions']*signs[:, None, None], atol=0, rtol=0)
                    torch.testing.assert_close(samples['positions'], samples['unflipped_positions']*signs[:, None, None], atol=0, rtol=0)
                    model, _, digest = load_entropy_adapter(directory/name)
                    replay_error = 0.
                    with torch.no_grad():
                        for start in range(0, parent_count, 64):
                            y, volume = model(base['positions'][start:start+64])
                            if volume.ndim == 0:
                                volume = volume.expand(len(y))
                            torch.testing.assert_close(y, samples['positions'][start:start+64], atol=1e-9, rtol=1e-9)
                            torch.testing.assert_close(volume, samples['log_volume'][start:start+64], atol=1e-9, rtol=1e-9)
                            replay_error = max(replay_error, float((y-samples['positions'][start:start+64]).abs().max()))
                    change = endpoint_kl_change(base['energy_eV'], samples['energy_eV'], base['positions'], samples['positions'],
                        kT=protocol['kT_eV'], restraint=protocol['restraint_eV_A2'], log_volume=samples['log_volume'])
                    torch.testing.assert_close(change, samples['paired_endpoint_kl_change'], atol=1e-10, rtol=1e-10)
                    row.update(state='complete', delta_kl=summarize(change),
                        delta_kl_per_internal_dof=summarize(change/(3*(len(trained['condition']['numbers'])-1))),
                        oracle_evaluations=trained['oracle_evaluations'], seconds=trained['seconds'],
                        parameters=trained['parameters'], results_sha256=file_sha(result_path),
                        checkpoint_sha256=digest, replay_max_error_A=replay_error,
                        maximum_COM_error_A=trained['maximum_COM_error_A'])
                    completed[(kind, replica)] = (trained, base, change)
                    condition_report['condition'] = trained['condition']
                condition_report['arms'].append(row)
        for replica in replicas:
            for control in ['affine', 'typed']:
                if ('convex', replica) not in completed or (control, replica) not in completed:
                    continue
                first, first_base, first_change = completed[('convex', replica)]
                second, second_base, second_change = completed[(control, replica)]
                for key in ['training_sha256', 'evaluation_sha256', 'source_results_sha256', 'source_checkpoint_sha256',
                        'selection_seed', 'orientation_seed', 'evaluation_sign_seed', 'condition']:
                    if first[key] != second[key]:
                        raise ValueError(f'Paired arms differ in {key}')
                torch.testing.assert_close(first_base['positions'], second_base['positions'], atol=0, rtol=0)
                if first_base['sample_ids'] != second_base['sample_ids']:
                    raise ValueError('Paired parent identities differ')
                if control == 'affine':
                    for key in ['init_seed', 'lr_start', 'lr_end', 'parameters', 'adapter_configuration']:
                        if first[key] != second[key]:
                            raise ValueError('Same-neural-context control differs')
                difference = first_change-second_change
                condition_report['paired'][f'convex_minus_{control}_s{replica}'] = {
                    'delta_kl': summarize(difference),
                    'delta_kl_per_internal_dof': summarize(difference/(3*(len(first['condition']['numbers'])-1)))}
        report['conditions'].append(condition_report)
    arms = [a for c in report['conditions'] for a in c['arms']]
    report.update(complete=True, completed_arms=sum(a['state'] == 'complete' for a in arms),
        failed_arms=sum(a['state'] == 'failed' for a in arms),
        unresolved_arms=sum(a['state'] == 'unresolved' for a in arms),
        all_arms_terminal=all(a['state'] != 'unresolved' for a in arms),
        acknowledged_queries_in_completed_arms=sum(a['oracle_evaluations'] for a in arms if a['state'] == 'complete'))
    report['acknowledged_queries_in_failed_arms_known'] = sum(a['oracle_evaluations'] or 0 for a in arms if a['state'] == 'failed')
    report['failed_query_accounting_incomplete'] = any(a.get('oracle_evaluations') is None for a in arms if a['state'] == 'failed')
    if args.require_complete and not report['all_arms_terminal']:
        raise ValueError('Prescribed campaign has unfinished arms')
    write_json(args.out, report)
    print(json.dumps({k: report[k] for k in ['completed_arms', 'failed_arms', 'unresolved_arms']}))


if __name__ == '__main__':
    main()
