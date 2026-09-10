#!/usr/bin/env python3
"""Replay both same-neural-context ablations on their shared confirmation panel."""
import argparse
import json
from pathlib import Path

import torch

from cfm_mol.entropy_adapter_io import load_entropy_adapter
from cfm_mol.linear_entropy_adapter import endpoint_kl_change
from molecular_tempered_pilot import sha, write_json


def summary(x):
    return {'mean': float(x.mean()), 'sem': float(x.std()/len(x)**.5), 'samples': len(x)}


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--runs-root', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
args = p.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)
root = args.runs_root
base_path = root/'species_entropy_adapter_confirmation_v1/base/final_samples.pt'
base = torch.load(base_path, map_location='cpu', weights_only=False)
rows = []
with torch.no_grad():
    for affine_run, confirm_run, convex_confirm in [
        ('species_affine_adapter_1000_v1', 'species_affine_confirmation_v1', 'species_entropy_adapter_confirmation_v1'),
        ('species_affine_adapter_1000_s9162', 'species_affine_replica_confirmation_v1', 'species_entropy_adapter_replica_confirmation_v1'),
    ]:
        model, trained, digest = load_entropy_adapter(root/affine_run)
        path = root/confirm_run/'results.json'
        r = json.loads(path.read_text())
        data_path = root/confirm_run/'affine_samples.pt'
        if (not r['complete'] or r['base_sha256'] != sha(base_path) or digest != r['affine_checkpoint_sha256']
                or sha(data_path) != r['samples_sha256'] or r['new_oracle_queries'] != 2048):
            raise ValueError('Affine confirmation provenance or budget differs')
        affine = torch.load(data_path, map_location='cpu', weights_only=False)
        convex_path = root/convex_confirm/'nonlinear_samples.pt'
        if sha(convex_path) != r['convex_samples_sha256']:
            raise ValueError('Convex comparison changed')
        convex = torch.load(convex_path, map_location='cpu', weights_only=False)
        if base['condition'] != trained['condition'] or affine['condition'] != base['condition'] or convex['condition'] != base['condition']:
            raise ValueError('Physical conditions differ')
        maximum_error = 0.
        for start in range(0, len(base['positions']), 64):
            y, volume = model(base['positions'][start:start+64])
            expected = affine['positions'][start:start+64]
            torch.testing.assert_close(y, expected, atol=1e-10, rtol=1e-10)
            torch.testing.assert_close(volume, affine['log_volume'][start:start+64], atol=1e-10, rtol=1e-10)
            maximum_error = max(maximum_error, float((y-expected).abs().max()))
        change = endpoint_kl_change(base['energy_eV'], affine['energy_eV'], base['positions'], affine['positions'],
            kT=r['kT_eV'], restraint=r['restraint_eV_A2'], log_volume=affine['log_volume'])
        difference = endpoint_kl_change(affine['energy_eV'], convex['energy_eV'], affine['positions'], convex['positions'],
            kT=r['kT_eV'], restraint=r['restraint_eV_A2'], log_volume=convex['log_volume']-affine['log_volume'])
        torch.testing.assert_close(change, affine['paired_endpoint_kl_change'], atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(affine['work']-base['work'], change, atol=1e-10, rtol=1e-10)
        calculated = summary(difference)
        if abs(calculated['mean']-r['convex_minus_affine']['mean']) > 1e-10 or abs(calculated['sem']-r['convex_minus_affine']['sem']) > 1e-10:
            raise ValueError('Reported paired result differs')
        rows.append({'training_run': affine_run, 'confirmation_run': confirm_run,
            'training_sha256': sha(root/affine_run/'results.json'), 'confirmation_sha256': sha(path),
            'checkpoint_sha256': digest, 'replay_max_error_A': maximum_error,
            'affine_delta_kl': summary(change), 'convex_minus_affine': calculated,
            'affine_path_weights': r['affine_weights'], 'training_queries': trained['oracle_evaluations'],
            'confirmation_queries': r['new_oracle_queries'], 'training_seconds': trained['seconds']})
out = {'complete': True, 'scope': __doc__, 'condition': base['condition'], 'base_sha256': sha(base_path),
    'rows': rows, 'replay_verified': True, 'new_affine_branch_queries_including_smoke': 64+2*(16512+2048),
    'limitations': ['One molecular condition, two training streams and shared2048-row evaluation; not4096 independent rows.',
        'Row SEM conditions on a trained model; this does not measure training-seed uncertainty.',
        'This ablation compares active point nonlinearity with its affine tangent while retaining the same neural context.',
        'The small difference does not establish broad performance, calibration or ICLR novelty.',
        'Existing source, convex/typed training, failed branches and independent assessment cost additional compute.']}
write_json(args.out, out)
print(json.dumps([row['convex_minus_affine'] for row in rows]))
