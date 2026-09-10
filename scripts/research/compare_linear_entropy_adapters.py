#!/usr/bin/env python3
"""Verify exact-volume refinement artifacts and join every development control."""
import argparse
import json
from pathlib import Path

import torch

from cfm_mol.linear_entropy_adapter import LinearEntropyAdapter, endpoint_kl_change
from molecular_tempered_pilot import sha, write_json


def summarize(x):
    return {'mean': float(x.mean()), 'sem': float(x.std()/len(x)**.5), 'samples': len(x)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    assessment_path = args.runs_root/'linear_entropy_adapter_xtb_v1/assessment.json'
    assessment = json.loads(assessment_path.read_text())
    if not assessment['complete'] or len(assessment['xtb_rows']) != 96:raise ValueError('Require all96 xTB attempts')
    reports, samples, sources = {}, {}, {}
    for kind in ['typed', 'scalar']:
        directory = args.runs_root/f'linear_entropy_adapter_{kind}_v1'
        r = json.loads((directory/'results.json').read_text())
        if not r['complete'] or r['oracle_evaluations'] != 3712 or r['steps'] != 200:raise ValueError('Incomplete or mismatched adapter budget')
        for file, expected in r['artifacts'].items():
            if sha(directory/file) != expected:raise ValueError('Artifact changed')
        data = torch.load(str(directory/'adapted_samples.pt'), map_location='cpu', weights_only=False)
        base = torch.load(str(directory/'base_samples.pt'), map_location='cpu', weights_only=False)
        state = torch.load(str(directory/'adapter.ckpt'), map_location='cpu', weights_only=False)
        model = LinearEntropyAdapter(r['condition']['numbers'], kind=kind)
        model.load_state_dict(state['state_dict'], strict=True)
        generated, volume = model(base['positions'])
        torch.testing.assert_close(generated, data['positions'], atol=1e-10, rtol=1e-10)
        if abs(float(volume)-r['log_volume']) > 1e-10:raise ValueError('Log-volume replay differs')
        change = endpoint_kl_change(base['energy_eV'], data['energy_eV'], base['positions'], data['positions'],
            kT=r['kT_eV'], restraint=r['restraint_eV_A2'], log_volume=volume)
        torch.testing.assert_close(change, data['paired_endpoint_kl_change'], atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(data['work']-base['work'], change, atol=1e-10, rtol=1e-10)
        reports[kind] = r; samples[kind] = data
        sources[kind] = {'results_sha256': sha(directory/'results.json'), 'samples_sha256': sha(directory/'adapted_samples.pt')}
        if kind == 'typed':first_base = base
        else:torch.testing.assert_close(base['positions'], first_base['positions'], atol=0, rtol=0)
    for key in ['condition', 'source_checkpoint_sha256', 'training_sha256', 'evaluation_sha256', 'kT_eV', 'restraint_eV_A2', 'seed']:
        if reports['typed'][key] != reports['scalar'][key]:raise ValueError('Control protocols differ')
    difference = endpoint_kl_change(samples['scalar']['energy_eV'], samples['typed']['energy_eV'],
        samples['scalar']['positions'], samples['typed']['positions'], kT=reports['typed']['kT_eV'],
        restraint=reports['typed']['restraint_eV_A2'], log_volume=reports['typed']['log_volume']-reports['scalar']['log_volume'])
    summary = {'complete': True, 'scope': __doc__, 'sources': sources, 'assessment_sha256': sha(assessment_path),
        'condition': reports['typed']['condition'], 'arms': {kind: {key: reports[kind][key] for key in
            ['paired_endpoint_kl_change', 'weights', 'mean_energy_eV', 'log_volume', 'transform_eigenvalues', 'oracle_evaluations', 'seconds']}
            for kind in ['typed', 'scalar']}, 'typed_minus_scalar_endpoint_kl': summarize(difference),
        'xtb': assessment['xtb_summaries'], 'new_queries_including_smoke': 2*3712+64,
        'replay_verified': True, 'sampling_calibrated': False, 'method_novelty_established': False,
        'limitations': ['One condition and one adapter seed, on an existing development evaluation panel.',
            'Negative mean change estimates relative endpoint KL improvement, not absolute closeness to the target.',
            'Path importance weights remain degenerate; xTB convergence is not complete chemical validity.',
            'Inherited source-training and sample-generation costs are additional; no matched-HMC superiority claim.']}
    write_json(args.out, summary)
    print(json.dumps({'typed_minus_scalar_endpoint_kl': summary['typed_minus_scalar_endpoint_kl'], 'xtb': summary['xtb']}))


if __name__ == '__main__':main()
