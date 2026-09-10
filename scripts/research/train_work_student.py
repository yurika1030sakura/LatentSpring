#!/usr/bin/env python3
"""Controlled empirical weighted-CFM projection, with independent full-target checks.

The finite teacher is not assumed calibrated. ODE projection and the stochastic
path sampler are evaluated separately; no score identity is used for either.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from flowmol.model_utils.load import read_config_file, model_from_config

from cfm_mol.condition_systems import graph_from_condition
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.importance_curriculum import weight_controls
from cfm_mol.molecular_path_drift import MolecularPathDrift
from cfm_mol.nonequilibrium import centered_orthonormal_basis, WeightedPaths
from cfm_mol.path_balance import fixed_backward_residuals, backward_log_probability, fixed_path_log_factors
from cfm_mol.path_work import gaussian_training_path
from cfm_mol.radial_reference import prepare_research_backbone
from cfm_mol.smooth_geometry import patch_smooth_geometry
from molecular_tempered_pilot import sha, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arm', choices=['source', 'uniform', 'linear', 'power'], required=True)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--eval-paths', type=int, default=256)
    p.add_argument('--refit-paths', type=int, default=2048)
    p.add_argument('--refit-steps', type=int, default=500)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=9085)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if min(args.steps, args.batch, args.eval_paths, args.refit_paths, args.refit_steps) < 1:
        raise ValueError('Positive counts required')
    if args.eval_paths % 64 or args.refit_paths % 64:
        raise ValueError('Evaluation and refit counts must be multiples of64')
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('Positive finite learning rate required')
    output = args.out/'results.json'
    if output.exists():
        raise FileExistsError(output)
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    torch.manual_seed(args.seed)
    teacher_report = json.loads((args.teacher/'results.json').read_text())
    teacher_path = args.teacher/'teacher.pt'
    if not teacher_report['complete'] or sha(teacher_path) != teacher_report['teacher_sha256']:
        raise ValueError('Require complete immutable teacher')
    teacher = torch.load(str(teacher_path), map_location='cpu', weights_only=False)
    controls, control_report = weight_controls(-teacher['work'], .5)
    for name, weights in controls.items():
        torch.testing.assert_close(weights, teacher['training_weights'][name], rtol=0, atol=0)
    if control_report != teacher_report['controls']:
        raise ValueError('Teacher weighting protocol changed')
    recipe = teacher_report['source_recipe']
    protocol = teacher_report['research_protocol']
    if recipe.get('reference_kernel') != 'gaussian' or recipe.get('precision_kind', 'none') != 'none':
        raise ValueError('Require scalar Gaussian path recipe')
    if protocol['position_parameterization'] != 'displacement' or protocol['data_endpoint_time'] != 1.:
        raise ValueError('Require T1 displacement field')
    source_run = Path(teacher_report['configuration']['source_run'])
    reverse_run = Path(teacher_report['configuration']['backward_refit'])
    source_checkpoint = source_run/'last.ckpt'
    reverse_checkpoint = reverse_run/'backward_refit.ckpt'
    if sha(source_checkpoint) != teacher_report['source_checkpoint_sha256'] or sha(reverse_checkpoint) != teacher_report['backward_checkpoint_sha256']:
        raise ValueError('Initialization checkpoint changed')
    source = torch.load(str(source_checkpoint), map_location='cpu', weights_only=False)
    source_report = json.loads((source_run/'results.json').read_text())
    if source['proposal_protocol'] != recipe or source['research_protocol'] != protocol:
        raise ValueError('Source protocol mismatch')
    cfg = read_config_file(Path(recipe['config']))
    cfg['mol_fm'].pop('bgfm', None)
    cfg['mol_fm']['prior_config']['x']['align'] = False
    if cfg['dataset']['max_atoms'] != 200 or cfg['mol_fm']['total_loss_weights']['e'] != 0:
        raise ValueError('Require bond-free max_atoms200')

    def load(weights):
        model = model_from_config(cfg)
        prepare_research_backbone(model, protocol)
        model.load_state_dict(weights, strict=True)
        patch_smooth_geometry(model, protocol.get('geometry_softening', 0.))
        return model.to(args.device).float().eval()

    forward = load(source['forward_state_dict'])
    backward = load(torch.load(str(reverse_checkpoint), map_location='cpu', weights_only=False)['backward_state_dict'])
    del source
    condition = teacher['condition']
    if condition != teacher_report['condition'] or condition != source_report['condition']:
        raise ValueError('Physical condition changed')
    n = len(condition['numbers'])
    dimension = 3*(n-1)
    prior_std = source_report['prior_std']
    basis = centered_orthonormal_basis(n, device=args.device)
    base = graph_from_condition({'atomic_numbers': condition['numbers'], 'charge': condition['charge'],
        'spin_multiplicity': condition['spin_multiplicity'], 'requested_kT_eV': recipe['kT']},
        cfg['dataset']['atom_map'], n_bond_classes=5 if cfg['mol_fm'].get('explicit_aromaticity', False) else 4).to(args.device)
    field = MolecularPathDrift(forward, base)
    reverse = MolecularPathDrift(backward, base, sign=-1.)
    targets = torch.einsum('nk,bnd->bkd', basis.cpu(), teacher['positions'].double()).reshape(-1, dimension)
    if not torch.isfinite(targets).all() or teacher['positions'].mean(1).abs().max() > 1e-8:
        raise ValueError('Invalid centered teacher endpoints')
    target_weights = controls['uniform' if args.arm == 'source' else args.arm]
    report = {'complete': False, 'format': 'empirical_work_cfm_student_v1', 'arm': args.arm,
        'configuration': {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'condition': condition, 'source_recipe': recipe, 'research_protocol': protocol,
        'source_checkpoint_sha256': sha(source_checkpoint), 'backward_initialization_sha256': sha(reverse_checkpoint),
        'teacher_sha256': sha(teacher_path), 'teacher_results_sha256': sha(args.teacher/'results.json'),
        'teacher_controls': control_report, 'temperature_reset_reapplied': False, 'history': [],
        'seeds': {'training': args.seed, 'ode': args.seed+1, 'reverse_training': args.seed+2,
                  'stochastic_evaluation': args.seed+3, 'reverse_minibatches': args.seed+4},
        'limitations': ['The finite weighted teacher has no established target calibration.',
            'ODE and stochastic kernels need not have the same endpoint law.',
            'Pair-distance projections are partial diagnostics, not joint-distribution validation.',
            'All full-target work checks use fresh paths and the original target; no independent eight-atom normalizer is known.',
            'This one-condition experiment is not a novel-method or broad-performance claim.']}
    write_json(output, report)
    parameters = [v for v in forward.parameters() if v.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.)
    selector = torch.Generator().manual_seed(args.seed)
    noise = torch.Generator(device=args.device).manual_seed(args.seed)
    for step in range(0 if args.arm == 'source' else args.steps):
        indices = torch.multinomial(target_weights, args.batch, replacement=True, generator=selector)
        x1 = targets[indices].to(args.device)
        x0 = torch.randn(x1.shape, dtype=torch.float64, device=args.device, generator=noise)*prior_std
        t = torch.rand((args.batch,), dtype=torch.float64, device=args.device, generator=noise)
        xt = (1-t[:, None])*x0+t[:, None]*x1
        optimizer.zero_grad(set_to_none=True)
        loss = (field(xt, t)-(x1-x0)).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite CFM objective')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step+1) % 50 == 0:
            row = {'step': step+1, 'cfm_loss': float(loss.detach()), 'gradient_norm': float(norm),
                   'seconds': time.perf_counter()-start}
            report['history'].append(row)
            write_json(output, report)
            print(json.dumps(row), flush=True)
    report['forward_updates'] = 0 if args.arm == 'source' else args.steps
    frozen = {key: value.detach().cpu().clone() for key, value in forward.state_dict().items()}
    for parameter in forward.parameters():
        parameter.requires_grad_(False)
    times = torch.linspace(0, 1, recipe['path_steps']+1, dtype=torch.float64)
    settings = {'prior_std': prior_std, 'terminal_std': math.sqrt(recipe['kT']/recipe['restraint']),
        'max_drift_norm': recipe['max_drift_per_sqrt_dimension']*math.sqrt(dimension),
        'mean_parameterization': recipe['mean_parameterization'], 'noise_annealing_power': recipe['noise_annealing_power']}

    def positions(z):
        return torch.einsum('nk,bkd->bnd', basis.cpu(), z.cpu().reshape(-1, n-1, 3))

    @torch.no_grad()
    def generate(count, seed):
        states, logs = [], []
        zero = lambda z, t: torch.zeros_like(z)
        for index in range(count//64):
            gen = torch.Generator(device=args.device).manual_seed(seed+100003*index)
            x0 = torch.randn((64, dimension), dtype=torch.float64, device=args.device, generator=gen)*prior_std
            path = gaussian_training_path(x0, field, zero, times, recipe['noise'], gen, retain_states=True, **settings)
            r, c = fixed_backward_residuals(path.states, zero, times, recipe['noise'], **settings)
            logq = path.log_initial+path.log_forward_minus_backward+backward_log_probability(r, c, dimension)
            if index == 0:
                initial, lf, _ = fixed_path_log_factors(path.states[:16], field, zero, times, recipe['noise'], **settings)
                error = float((initial+lf-logq[:16]).abs().max())
                if error > .05:
                    raise RuntimeError('Forward path density cache mismatch')
                report.setdefault('forward_density_checks', []).append({'seed': seed, 'max_error': error})
            states.append(path.states.cpu()); logs.append(logq.cpu())
        return {'states': torch.cat(states), 'log_forward_path': torch.cat(logs)}

    @torch.no_grad()
    def ode(count, steps):
        outputs = []
        for index in range(count//64):
            gen = torch.Generator(device=args.device).manual_seed(args.seed+1+100003*index)
            z = torch.randn((64, dimension), dtype=torch.float64, device=args.device, generator=gen)*prior_std
            for k in range(steps):
                midpoint = z+field(z, k/steps)/(2*steps)
                z = z+field(midpoint, (k+.5)/steps)/steps
            outputs.append(z.cpu())
        return positions(torch.cat(outputs))

    def projection(x):
        pairs = torch.triu_indices(n, n, 1)
        a = torch.cdist(x, x)[:, pairs[0], pairs[1]].numpy()
        b = torch.cdist(teacher['positions'], teacher['positions'])[:, pairs[0], pairs[1]].numpy()
        distances = [wasserstein_distance(a[:, j], b[:, j], v_weights=target_weights.numpy()) for j in range(a.shape[1])]
        return {'mean_pair_marginal_w1_A': float(np.mean(distances)), 'maximum_pair_marginal_w1_A': float(np.max(distances))}

    x32 = ode(args.eval_paths, 32)
    x64 = ode(args.eval_paths, 64)
    report['ode_step_check'] = {'coordinate_rms_A': float((x32-x64).square().mean().sqrt()),
        'maximum_atom_displacement_A': float((x32-x64).norm(dim=-1).max()), 'paired_paths': args.eval_paths}
    report['ode_projection'] = projection(x64)
    training = generate(args.refit_paths, args.seed+2)
    heldout = generate(args.eval_paths, args.seed+3)
    x = positions(heldout['states'][:, -1])
    report['stochastic_projection'] = projection(x)
    root = Path(__file__).resolve().parents[2]
    if sha(Path(recipe['oracle'])) != teacher_report['oracle_sha256']:
        raise ValueError('Target oracle changed')
    with EnergyOracle(Path(recipe['oracle_python']), root/'scripts/research/oracle_worker.py', Path(recipe['oracle']),
        numbers=condition['numbers'], charge=condition['charge'], spin_multiplicity=condition['spin_multiplicity'], batch_size=16) as oracle:
        ode_energy, _ = oracle.evaluate(x64)
        energy, _ = oracle.evaluate(x)
        report['new_oracle_evaluations'] = oracle.evaluated
    reduced = (energy-teacher_report['energy_zero_eV']+recipe['restraint']/2*x.square().sum((1, 2)))/recipe['kT']
    report['mean_energy_eV'] = {'ode64': float(ode_energy.mean()), 'stochastic16': float(energy.mean())}

    @torch.no_grad()
    def residuals(data):
        residual, norm = [], []
        for begin in range(0, len(data['states']), 16):
            r, c = fixed_backward_residuals(data['states'][begin:begin+16].to(args.device), reverse, times, recipe['noise'], **settings)
            residual.append(r.cpu()); norm.append(c.cpu())
        return torch.cat(residual), torch.cat(norm)

    works = {}
    def evaluate(stage):
        tr, _ = residuals(training)
        er, en = residuals(heldout)
        ratios = (tr.mean(0)/dimension).clamp(.25, 1.9)
        work = reduced+heldout['log_forward_path']-backward_log_probability(er, en, dimension, ratios)
        works[stage] = work
        report.setdefault('full_target_evaluations', {})[stage] = {'weights': WeightedPaths(x, -work, {}).summary(),
            'mean_work': float(work.mean()), 'work_std': float(work.std()), 'variance_ratios': ratios.tolist()}
        write_json(output, report)
        print(json.dumps({'evaluation': stage, **report['full_target_evaluations'][stage]}), flush=True)
        return ratios

    evaluate('before_reverse_refit')
    reverse_parameters = [v for v in backward.parameters() if v.requires_grad]
    reverse_optimizer = torch.optim.AdamW(reverse_parameters, lr=args.lr, weight_decay=0.)
    selector = torch.Generator().manual_seed(args.seed+4)
    for step in range(args.refit_steps):
        indices = torch.randint(args.refit_paths, (16,), generator=selector)
        reverse_optimizer.zero_grad(set_to_none=True)
        r, c = fixed_backward_residuals(training['states'][indices].to(args.device), reverse, times, recipe['noise'], **settings)
        loss = -backward_log_probability(r, c, dimension).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite reverse objective')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reverse_parameters, 1., error_if_nonfinite=True)
        reverse_optimizer.step()
        if (step+1) % 100 == 0:
            print(json.dumps({'reverse_step': step+1, 'loss': float(loss.detach())}), flush=True)
    ratios = evaluate('after_reverse_refit')
    if not all(torch.equal(value.cpu(), frozen[key]) for key, value in forward.state_dict().items()):
        raise RuntimeError('Forward state changed during reverse-only fitting')
    torch.save({'positions': x64, 'positions32': x32, 'energy_eV': ode_energy, 'condition': condition}, args.out/'ode_samples.pt')
    torch.save({'positions': x, 'energy_eV': energy, 'work': works['after_reverse_refit'], 'works': works,
        'states': heldout['states'], 'log_forward_path': heldout['log_forward_path'], 'condition': condition}, args.out/'stochastic_samples.pt')
    torch.save(training, args.out/'reverse_training_paths.pt')
    torch.save({'forward_state_dict': forward.state_dict(), 'backward_state_dict': backward.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), 'reverse_optimizer_state_dict': reverse_optimizer.state_dict(),
        'research_protocol': protocol, 'source_proposal_protocol': recipe, 'student_protocol': report['configuration'],
        'backward_variance_ratios': ratios, 'teacher_sha256': sha(teacher_path)}, args.out/'student.ckpt')
    report.update(complete=True, seconds=time.perf_counter()-start, forward_state_unchanged_during_reverse_refit=True,
        artifacts={name: sha(args.out/name) for name in ['ode_samples.pt', 'stochastic_samples.pt', 'reverse_training_paths.pt', 'student.ckpt']})
    write_json(output, report)


if __name__ == '__main__':
    main()
