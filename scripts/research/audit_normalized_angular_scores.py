#!/usr/bin/env python3
"""Compare fitted quadratic scores and actual normalized-mixture angular scores.

Diagnostic on the original TRAINING table only, with no fitting or physical calls.
The normalized mixture used for graph edits need not preserve the force fit of
the Fisher--Bingham score used to train its coefficients.
"""
import argparse
import json
from pathlib import Path
import torch
from cfm_mol.masked_angular_data import training_angular_examples
from cfm_mol.masked_angular_guide import angular_surface_score
from cfm_mol.angular_envelope import envelope_parameters
from scripts.research.evaluate_joint_chemical import load_model, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['table', 'models', 'out']:
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    protocol = json.loads((root/'research/evidence/joint_chemical_protocol_v1.json').read_text())
    header = json.loads((args.table/'results.json').read_text())
    assert sha(args.table/'results.json') == protocol['table_results_sha256']
    assert sha(args.table/'training.pt') == header['artifacts']['training']
    d = training_angular_examples(torch.load(args.table/'training.pt', map_location='cpu', weights_only=False))
    fit = json.loads((root/'research/evidence/masked_angular_protocol_v1.json').read_text())
    rows = []
    for name in ['vector', 'tensor']:
        for replica in [0, 1]:
            model, metadata = load_model(args.models, name, replica, protocol, header['artifacts']['training'])
            squared = {key: [] for key in ['quadratic', 'normalized_mixture', 'zero']}
            with torch.no_grad():
                for begin in range(0, len(d['examples']), 256):
                    ex = d['examples'][begin:begin+256]
                    sid = ex[:, 0]
                    eta, a, _ = model(d['x'][sid], d['bonds'][sid], d['numbers'], d['electronic'], ex[:, 1:3])
                    u = d['directions'][begin:begin+len(ex)]
                    target = d['targets'][begin:begin+len(ex)]
                    target = target*torch.clamp(fit['score_clip']/target.norm(dim=1, keepdim=True).clamp_min(1e-300), max=1)
                    parameters = envelope_parameters(eta, a)
                    gap, axis = parameters['gap'], parameters['axis']
                    raw = eta+(gap*torch.tanh(gap*(axis*u).sum(1)))[:, None]*axis
                    normalized = raw-(raw*u).sum(1, keepdim=True)*u
                    quadratic = angular_surface_score(u, eta, a)
                    for key, prediction in [('quadratic', quadratic), ('normalized_mixture', normalized), ('zero', torch.zeros_like(target))]:
                        squared[key].append((prediction-target).square().mean(1))
            errors = {key: torch.cat(value) for key, value in squared.items()}
            assert abs(float(errors['quadratic'].mean())-metadata['training_mse']) < 1e-7
            if name == 'vector':
                torch.testing.assert_close(errors['quadratic'], errors['normalized_mixture'], atol=1e-10, rtol=1e-10)
            rows.append(dict(model=name, replica=replica, checkpoint_sha256=metadata['checkpoint_sha256'],
                example_weighted_mse={key: float(value.mean()) for key, value in errors.items()},
                parent_balanced_mse={key: float(torch.stack([value[group].mean() for group in d['parent_groups']]).mean()) for key, value in errors.items()}))
    result = dict(complete=True, training_sha256=header['artifacts']['training'], training_parents=len(d['parent_groups']),
        angular_examples=len(d['examples']), development_coordinates_loaded=False, reference_coordinates_loaded=False,
        new_physical_queries=0, rows=rows, scientific_submission_ready=False,
        interpretation='Pointwise TRAINING force fit. Mixture normalization changes the tensor directional model; this is not a causal explanation of molecular performance or equilibrium accuracy.')
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
