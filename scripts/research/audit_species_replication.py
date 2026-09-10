#!/usr/bin/env python3
"""Replay both trained adapter pairs and retain their shared-panel confirmation."""
import argparse
import json
from pathlib import Path

import torch

from cfm_mol.entropy_adapter_io import load_entropy_adapter
from cfm_mol.linear_entropy_adapter import endpoint_kl_change
from molecular_tempered_pilot import sha, write_json


def summary(x):
    return {'mean': float(x.mean()), 'sem': float(x.std()/len(x)**.5), 'samples': len(x)}


def read(path):
    r = json.loads(path.read_text())
    if not r['complete']:
        raise ValueError(f'Incomplete evidence: {path}')
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    root = args.runs_root
    base_dir = root/'species_entropy_adapter_confirmation_v1/base'
    base_report = read(base_dir/'results.json')
    base = torch.load(base_dir/'final_samples.pt', map_location='cpu', weights_only=False)
    pairs = [
        ('main', 'species_entropy_adapter_1000_v1', 'linear_entropy_adapter_typed_1000_v1',
         'species_entropy_adapter_confirmation_v1', 9161, 9141),
        ('replica', 'species_entropy_adapter_1000_s9162', 'linear_entropy_adapter_typed_1000_s9142',
         'species_entropy_adapter_replica_confirmation_v1', 9162, 9142),
    ]
    output = {'complete': False, 'scope': __doc__, 'pairs': {},
        'base_sha256': sha(base_dir/'final_samples.pt'), 'condition': base['condition'],
        'source_checkpoint_sha256': base_report['trained_checkpoint_sha256'],
        'oracle_sha256': base_report['oracle_sha256'],
        'kT_eV': base_report['configuration']['kT'],
        'restraint_eV_A2': base_report['configuration']['restraint'],
        'fresh_base_weights': None, 'xtb': {}, 'development_screens': {},
        'replay_verified': False, 'sampling_calibrated': False, 'method_novelty_established': False,
        'limitations': [
            'One eight-atom condition; two independently selected training streams and neural initializations.',
            'Both pairs use the same4096 training parent pool,256 development rows and2048 confirmation rows.',
            'SEM is across evaluation rows conditional on a trained model; it is not uncertainty across training seeds.',
            'No pooled independent-sample count across seeds: evaluation noise is shared.',
            'Linear has13 parameters and lr .01 to .0001; nonlinear has19365 parameters and lr .001 to .00001.',
            'Oracle budgets are matched, while wall times and network computation differ.',
            'Small relative endpoint-KL gains do not certify target populations or effective importance sampling.',
            'xTB relaxation success is not full chemical validity; no strong coupling-flow/HMC superiority is established.',
            'Base pretraining, source training, sample generation and earlier failed method branches cost additional compute.',
        ]}
    with torch.no_grad():
        for label, nonlinear_name, linear_name, confirm_name, init_seed, selection_seed in pairs:
            confirmation_dir = root/confirm_name
            confirmation = read(confirmation_dir/'results.json')
            if confirmation['base_sha256'] != output['base_sha256']:
                raise ValueError('Confirmation panels differ')
            pair = {'arms': {}, 'confirmation_results_sha256': sha(confirmation_dir/'results.json')}
            changes = {}
            development_changes = {}
            training_hashes = []
            for arm, name in [('nonlinear', nonlinear_name), ('linear', linear_name)]:
                directory = root/name
                model, report, digest = load_entropy_adapter(directory)
                for key in ['condition', 'source_checkpoint_sha256', 'oracle_sha256', 'kT_eV', 'restraint_eV_A2']:
                    if report[key] != output[key]:
                        raise ValueError(f'Target/source mismatch: {key}')
                if report['oracle_evaluations'] != 16512 or report['steps'] != 1000 or report['batch'] != 16:
                    raise ValueError('Matched training budget failed')
                if report.get('selection_seed', report.get('seed')) != selection_seed:
                    raise ValueError('Training selection seed differs')
                if arm == 'nonlinear' and report['init_seed'] != init_seed:
                    raise ValueError('Neural initialization seed differs')
                training_hashes.append((report['training_sha256'], report['evaluation_sha256']))
                for file, expected in report['artifacts'].items():
                    if sha(directory/file) != expected:
                        raise ValueError(f'Changed artifact: {name}/{file}')
                data_path = confirmation_dir/f'{arm}_samples.pt'
                if sha(data_path) != confirmation['arms'][arm]['samples_sha256'] or digest != confirmation['arms'][arm]['checkpoint_sha256']:
                    raise ValueError('Confirmation artifact/checkpoint differs')
                data = torch.load(data_path, map_location='cpu', weights_only=False)
                max_error = 0.
                for begin in range(0, len(base['positions']), 64):
                    y, volume = model(base['positions'][begin:begin+64].double())
                    if volume.ndim == 0:
                        volume = volume.expand(len(y))
                    expected_y = data['positions'][begin:begin+64]
                    torch.testing.assert_close(y, expected_y, atol=1e-10, rtol=1e-10)
                    torch.testing.assert_close(volume, data['log_volume'][begin:begin+64], atol=1e-10, rtol=1e-10)
                    max_error = max(max_error, float((y-expected_y).abs().max()))
                change = endpoint_kl_change(base['energy_eV'], data['energy_eV'], base['positions'], data['positions'],
                    kT=report['kT_eV'], restraint=report['restraint_eV_A2'], log_volume=data['log_volume'])
                torch.testing.assert_close(change, data['paired_endpoint_kl_change'], atol=1e-10, rtol=1e-10)
                torch.testing.assert_close(data['work']-base['work'], change, atol=1e-10, rtol=1e-10)
                changes[arm] = change
                dev = torch.load(directory/'adapted_samples.pt', map_location='cpu', weights_only=False)
                development_changes[arm] = dev['paired_endpoint_kl_change']
                pair['arms'][arm] = {key: report[key] for key in ['oracle_evaluations', 'seconds', 'parameters', 'lr_start', 'lr_end']}
                pair['arms'][arm].update(run=name, checkpoint_sha256=digest, replay_max_error_A=max_error,
                    development_delta_kl=summary(development_changes[arm]), confirmation_delta_kl=summary(change),
                    confirmation_weights=confirmation['arms'][arm]['weights'],
                    training_results_sha256=sha(directory/'results.json'))
            if len(set(training_hashes)) != 1:
                raise ValueError('Within-pair parent pools differ')
            pair.update(training_sha256=training_hashes[0][0], evaluation_sha256=training_hashes[0][1],
                neural_init_seed=init_seed, selection_seed=selection_seed,
                development_nonlinear_minus_linear=summary(development_changes['nonlinear']-development_changes['linear']),
                confirmation_nonlinear_minus_linear=summary(changes['nonlinear']-changes['linear']))
            reported = confirmation['nonlinear_minus_linear']
            if abs(pair['confirmation_nonlinear_minus_linear']['mean']-reported['mean']) > 1e-10:
                raise ValueError('Reported paired result differs')
            output['pairs'][label] = pair
            output['fresh_base_weights'] = confirmation['base_weights']
    for label, name, count in [
        ('steps200', 'species_entropy_adapter_200_xtb_v2', 128),
        ('main1000', 'species_entropy_adapter_1000_xtb_v1', 96),
        ('replica1000', 'species_entropy_adapter_replica_xtb_v1', 96),
    ]:
        path = root/name/'assessment.json'
        r = read(path)
        if len(r['xtb_rows']) != count or sum(row['attempted'] for row in r['xtb_summaries']) != count:
            raise ValueError('Missing xTB failure denominators')
        output['xtb'][label] = {'run': name, 'sha256': sha(path), 'summaries': r['xtb_summaries'], 'paired': r['paired']}
    for name in ['species_entropy_adapter_smoke_v1', 'species_entropy_adapter_200_v1']:
        path = root/name/'results.json'
        r = read(path)
        output['development_screens'][name] = {key: r[key] for key in ['oracle_evaluations', 'paired_endpoint_kl_change']}
        output['development_screens'][name]['results_sha256'] = sha(path)
    output['new_species_branch_oracle_queries'] = 64+3712+4*16512+6144+4096
    output['new_species_branch_xtb_attempts'] = 320
    output.update(complete=True, replay_verified=True)
    write_json(args.out, output)
    print(json.dumps({name: row['confirmation_nonlinear_minus_linear'] for name, row in output['pairs'].items()}))


if __name__ == '__main__':
    main()
