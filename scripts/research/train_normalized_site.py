#!/usr/bin/env python3
"""Bounded force regression for the actual normalized directional guide."""
import argparse
import json
from pathlib import Path
import time
import torch
from cfm_mol.masked_angular_data import training_angular_examples
from cfm_mol.masked_angular_guide import masked_angular_context
from cfm_mol.normalized_site_guide import NormalizedSiteGuide, mixture_surface_score, physical_site_parameter
from scripts.research.evaluate_chemical_policy import sha, write


def clipped_target(target, cap):
    return target*torch.clamp(cap/target.norm(dim=1, keepdim=True).clamp_min(1e-300), max=1)


@torch.no_grad()
def errors(model, data, protocol):
    values = {key: [] for key in ['model', 'physical_site', 'zero']}
    for start in range(0, len(data['examples']), 256):
        ex = data['examples'][start:start+256]
        sid = ex[:, 0]
        x, bonds, roots = data['x'][sid], data['bonds'][sid], ex[:, 1:3]
        params, weights = model(x, bonds, data['numbers'], data['electronic'], roots)
        direction = data['directions'][start:start+len(ex)]
        target = clipped_target(data['targets'][start:start+len(ex)], protocol['score_clip'])
        prediction = mixture_surface_score(direction, params, weights)
        masked, _, roles = masked_angular_context(x, roots)
        site = physical_site_parameter(masked, roles, bonds, roots, protocol['architecture']['site_concentration'])
        physical = site-(site*direction).sum(1, keepdim=True)*direction
        for key, value in [('model', prediction), ('physical_site', physical), ('zero', torch.zeros_like(target))]:
            values[key].append((value-target).square().mean(1))
    values = {key: torch.cat(value) for key, value in values.items()}
    return dict(example_weighted={key: float(value.mean()) for key, value in values.items()},
        parent_balanced={key: float(torch.stack([value[group].mean() for group in data['parent_groups']]).mean()) for key, value in values.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', choices=['mixture', 'vector'], required=True)
    parser.add_argument('--replica', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    pp = root/'research/evidence/normalized_site_training_protocol_v1.json'
    protocol = json.loads(pp.read_text())
    header = json.loads((args.table/'results.json').read_text())
    training = args.table/'training.pt'
    assert header['complete'] and sha(args.table/'results.json') == protocol['table_results_sha256']
    assert sha(training) == protocol['training_artifact_sha256']
    data = training_angular_examples(torch.load(training, map_location='cpu', weights_only=False))
    torch.manual_seed(protocol['model_seeds'][args.replica])
    model = NormalizedSiteGuide(**protocol['architecture'], mixture=args.model == 'mixture').double()
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    initial_errors = errors(model, data, protocol)
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol['learning_rate'])
    rng = torch.Generator().manual_seed(protocol['data_seeds'][args.replica])
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out/'results.json'
    if output.exists():
        raise FileExistsError(output)
    report = dict(complete=False, model=args.model, replica=args.replica,
        protocol_sha256=sha(pp), training_sha256=sha(training), initial_errors=initial_errors,
        new_physical_queries=0, development_coordinates_loaded=False, reference_coordinates_loaded=False,
        scientific_submission_ready=False, history=[])
    write(output, report)
    start = time.perf_counter()
    for step in range(protocol['training_steps']):
        parents = torch.randint(len(data['parent_groups']), (protocol['batch_size'],), generator=rng)
        chosen = torch.tensor([int(data['parent_groups'][i][int(torch.randint(len(data['parent_groups'][i]), (1,), generator=rng))]) for i in parents.tolist()])
        ex = data['examples'][chosen]
        sid = ex[:, 0]
        params, weights = model(data['x'][sid], data['bonds'][sid], data['numbers'], data['electronic'], ex[:, 1:3])
        prediction = mixture_surface_score(data['directions'][chosen], params, weights)
        target = clipped_target(data['targets'][chosen], protocol['score_clip'])
        mse = (prediction-target).square().mean()/protocol['score_clip']**2
        penalty = params.square().sum(2).mean()/protocol['architecture']['bound']**2
        loss = mse+protocol['coefficient_regularization']*penalty
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite actual-density force loss')
        optimizer.zero_grad()
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Nonfinite model derivative')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        if step % 100 == 0 or step+1 == protocol['training_steps']:
            row = dict(step=step+1, normalized_mse=float(mse.detach()), penalty=float(penalty.detach()))
            report['history'].append(row)
            print(json.dumps(row), flush=True)
    fitting_seconds = time.perf_counter()-start
    final_errors = errors(model, data, protocol)
    artifact = dict(configuration=model.configuration, state_dict=model.state_dict(), initial_state_dict=initial,
        protocol_sha256=sha(pp), training_sha256=sha(training), parent_ids=data['parent_ids'],
        model=args.model, replica=args.replica, data_generator_state=rng.get_state())
    torch.save(artifact, args.out/'model.pt')
    report.update(complete=True, checkpoint_sha256=sha(args.out/'model.pt'), parameters=sum(p.numel() for p in model.parameters()),
        parameters_with_gradient=sum(p.numel() for p in model.parameters() if p.grad is not None),
        final_errors=final_errors, fitting_seconds=fitting_seconds, fitting_and_final_audit_seconds=time.perf_counter()-start,
        inherited_training_raw_queries=header['streams']['training']['raw_queries'],
        training_parents=len(data['parent_groups']), scored_states=len(data['x']), angular_examples=len(data['examples']),
        limitation='Only pointwise normalized conditional force fitting on generated training states; no equilibrium or generalization theorem.')
    write(output, report)


if __name__ == '__main__':
    main()
