#!/usr/bin/env python3
"""Qualify a frozen molecular proposal score before any endpoint-entropy actor update."""
import argparse
import json
import math
from pathlib import Path
import time

import torch
from flowmol.model_utils.load import read_config_file, model_from_config

from cfm_mol.condition_systems import graph_from_condition
from cfm_mol.endpoint_entropy import normalized_dsm_loss
from cfm_mol.innovation_posterior import GaussianInnovationMap
from cfm_mol.molecular_path_drift import MolecularPathDrift
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.proposal_score import ProposalEnergyCritic, invariant_stein_rows
from cfm_mol.radial_reference import prepare_research_backbone
from cfm_mol.smooth_geometry import patch_smooth_geometry
from molecular_tempered_pilot import sha, write_json


def summary(values):
    values = values.detach().double().cpu()
    mean = float(values.mean()); sem = float(values.std()/math.sqrt(len(values)))
    return {'mean': mean, 'sem': sem, 'count': len(values), 'within_three_sem_of_zero': abs(mean) <= 3*sem}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--training-pool', type=Path, required=True)
    p.add_argument('--heldout-pool', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--objective', choices=['raw', 'scaled_iid', 'grouped_iid', 'antithetic', 'mean_cv'], default='raw')
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--seed', type=int, default=9101)
    p.add_argument('--device', default='cuda')
    p.add_argument('--critic-checkpoint', type=Path)
    p.add_argument('--fresh-heldout-parents', type=int, default=0)
    p.add_argument('--heldout-seed', type=int, default=9107)
    p.add_argument('--heldout-means', type=Path)
    p.add_argument('--evaluation-seed', type=int)
    args = p.parse_args()
    if args.batch % 2 or args.batch < 1 or args.steps < 0 or (args.steps == 0 and args.critic_checkpoint is None):
        raise ValueError('Positive training counts required')
    if args.fresh_heldout_parents < 0 or args.fresh_heldout_parents % 64:
        raise ValueError('Fresh heldout count must be a nonnegative multiple of64')
    if args.fresh_heldout_parents and args.heldout_means is not None:
        raise ValueError('Choose fresh generation or a verified saved mean pool')
    output = args.out/'results.json'
    if output.exists():
        raise FileExistsError(output)
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter(); torch.manual_seed(args.seed)
    source = json.loads((args.source_run/'results.json').read_text())
    training_report = json.loads((args.training_pool/'results.json').read_text())
    heldout_report = json.loads((args.heldout_pool/'results.json').read_text())
    checkpoint = args.source_run/'last.ckpt'; checkpoint_sha = sha(checkpoint)
    if not all(r['complete'] for r in [source, training_report, heldout_report]):
        raise ValueError('Completed source runs required')
    if any(r['source_checkpoint_sha256'] != checkpoint_sha for r in [training_report, heldout_report]):
        raise ValueError('Pools came from different forward models')
    heldout_parent_seed = args.heldout_seed if args.fresh_heldout_parents else heldout_report['train_seed']
    if training_report['configuration']['seed'] == heldout_parent_seed:
        raise ValueError('Independent parent path streams required')
    train_path = args.training_pool/'teacher.pt'; test_path = args.heldout_pool/'training_paths.pt'
    if sha(train_path) != training_report['teacher_sha256'] or sha(test_path) != heldout_report['training_paths_sha256']:
        raise ValueError('Frozen parent pool changed')
    train_states = torch.load(str(train_path), map_location='cpu', weights_only=False)['states']
    test_states = None if args.fresh_heldout_parents or args.heldout_means is not None else torch.load(str(test_path), map_location='cpu', weights_only=False)['states']
    state = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    recipe = state['proposal_protocol']; protocol = state['research_protocol']
    if recipe != source['configuration'] or recipe.get('precision_kind', 'none') != 'none':
        raise ValueError('Require original scalar Gaussian proposal')
    if protocol['position_parameterization'] != 'displacement' or protocol['data_endpoint_time'] != 1.:
        raise ValueError('Require declared T1 displacement field')
    cfg = read_config_file(Path(recipe['config']))
    cfg['mol_fm'].pop('bgfm', None); cfg['mol_fm']['prior_config']['x']['align'] = False
    if cfg['dataset']['max_atoms'] != 200 or cfg['mol_fm']['total_loss_weights']['e'] != 0:
        raise ValueError('Require bond-free max_atoms200')
    forward = model_from_config(cfg); prepare_research_backbone(forward, protocol)
    forward.load_state_dict(state['forward_state_dict'], strict=True)
    patch_smooth_geometry(forward, protocol.get('geometry_softening', 0.))
    forward.to(args.device).float().eval()
    for parameter in forward.parameters():
        parameter.requires_grad_(False)
    del state
    condition = source['condition']; n = len(condition['numbers']); dimension = 3*(n-1)
    if any(r['condition'] != condition for r in [training_report, heldout_report]):
        raise ValueError('Physical condition mismatch')
    graph = graph_from_condition({'atomic_numbers': condition['numbers'], 'charge': condition['charge'],
        'spin_multiplicity': condition['spin_multiplicity'], 'requested_kT_eV': recipe['kT']},
        cfg['dataset']['atom_map'], n_bond_classes=5 if cfg['mol_fm'].get('explicit_aromaticity', False) else 4).to(args.device)
    field = MolecularPathDrift(forward, graph)
    settings = {'prior_std': source['prior_std'],
        'terminal_std': math.sqrt(recipe['kT']/recipe['restraint']) if recipe['reference_kernel'] == 'gaussian' else None,
        'max_drift_norm': recipe['max_drift_per_sqrt_dimension']*math.sqrt(dimension),
        'mean_parameterization': recipe['mean_parameterization'], 'noise_annealing_power': recipe['noise_annealing_power']}
    mapping = GaussianInnovationMap(field, dimension, torch.linspace(0, 1, recipe['path_steps']+1, dtype=torch.float64), recipe['noise'], **settings)
    sigma = mapping.terminal_noise_std
    def means(states):
        if states.shape[1:] != (recipe['path_steps']+1, dimension):
            raise ValueError('Unexpected path shape')
        with torch.no_grad():
            return torch.cat([mapping.mean(states[begin:begin+64, -2].to(args.device), mapping.steps-1).cpu()
                for begin in range(0, len(states), 64)])
    train_means = means(train_states)
    if args.fresh_heldout_parents:
        outputs = []
        with torch.no_grad():
            for block in range(args.fresh_heldout_parents//64):
                generator = torch.Generator(device=args.device).manual_seed(args.heldout_seed+100003*block)
                latent = torch.randn((64, mapping.latent_dimension), dtype=torch.float64, device=args.device, generator=generator)
                outputs.append(mapping(latent).cpu())
                if (block+1) % 16 == 0:
                    print(json.dumps({'fresh_heldout_parents': (block+1)*64, 'seconds': time.perf_counter()-start}), flush=True)
        test_means = torch.cat(outputs)
        test_path = args.out/'fresh_means.pt'
        torch.save({'means': test_means, 'source_checkpoint_sha256': checkpoint_sha, 'condition': condition,
            'terminal_noise_std': sigma, 'seed': args.heldout_seed}, test_path)
    elif args.heldout_means is not None:
        test_path = args.heldout_means
        parent_report = json.loads((test_path.parent/'results.json').read_text())
        if not parent_report['complete'] or parent_report['heldout_pool_sha256'] != sha(test_path):
            raise ValueError('Heldout mean pool lacks a complete verified source')
        cached = torch.load(str(test_path), map_location='cpu', weights_only=False)
        if cached['source_checkpoint_sha256'] != checkpoint_sha or cached['condition'] != condition or cached['terminal_noise_std'] != sigma:
            raise ValueError('Heldout mean population changed')
        heldout_parent_seed = cached['seed']; test_means = cached['means']
        if heldout_parent_seed == training_report['configuration']['seed']:
            raise ValueError('Heldout mean seed overlaps training')
    else:
        test_means = means(test_states)
    if test_means.ndim != 2 or test_means.shape[1] != dimension or not torch.isfinite(test_means).all():
        raise ValueError('Invalid heldout means')
    del mapping, field, forward, graph, train_states, test_states
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    basis = centered_orthonormal_basis(n, device=args.device)
    numbers = torch.tensor(condition['numbers'], dtype=torch.long, device=args.device)
    electronic = torch.tensor([condition['charge']/5., (condition['spin_multiplicity']-1)/5., math.log(recipe['kT'])],
        dtype=torch.float64, device=args.device)
    critic = ProposalEnergyCritic().to(args.device).double()
    initialization_sha = None
    if args.critic_checkpoint is not None:
        loaded = torch.load(str(args.critic_checkpoint), map_location='cpu', weights_only=False)
        if loaded['source_checkpoint_sha256'] != checkpoint_sha or loaded['condition'] != condition or loaded['terminal_noise_std'] != sigma:
            raise ValueError('Critic checkpoint represents a different proposal')
        critic.load_state_dict(loaded['critic_state_dict'], strict=True)
        initialization_sha = sha(args.critic_checkpoint)
    optimizer = torch.optim.AdamW(critic.parameters(), lr=.001, weight_decay=0.)
    selection = torch.Generator().manual_seed(args.seed)
    noise = torch.Generator(device=args.device).manual_seed(args.seed+10000019)
    gaussian_precision = dimension/float(train_means.square().sum(-1).mean()+dimension*sigma**2)
    report = {'complete': False, 'scope': __doc__, 'configuration': {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'condition': condition, 'source_checkpoint_sha256': checkpoint_sha, 'training_pool_sha256': sha(train_path),
        'heldout_pool_sha256': sha(test_path), 'training_parent_seed': training_report['configuration']['seed'],
        'heldout_parent_seed': heldout_parent_seed, 'train_parents': len(train_means), 'heldout_parents': len(test_means),
        'terminal_noise_std': sigma, 'gaussian_baseline_precision': gaussian_precision, 'new_oracle_evaluations': 0,
        'forward_model_updates': 0, 'critic_initialization_sha256': initialization_sha,
        'evaluation_seed': args.seed+1 if args.evaluation_seed is None else args.evaluation_seed,
        'critic_parameters': sum(v.numel() for v in critic.parameters()),
        'score_point_evaluations_per_update': args.batch,
        'independent_parents_per_update': args.batch if args.objective in ['raw', 'scaled_iid'] else args.batch//2,
        'clipped_updates': 0, 'maximum_preclip_gradient_norm': 0.,
        'history': [], 'limitations': ['Frozen proposal only: no molecular actor or generated-distribution update.',
            'Fresh final noise is reused across controls; parent means are independent between training and assessment.',
            'The finite training mixture differs from the population proposal; independent heldout parents detect some overfitting.',
            'Denoising and four Stein moments are necessary checks, not a complete score or calibration certificate.',
            'Standard invariant energy network and score-distillation tools are not claimed novel.']}
    write_json(output, report)
    for step in range(args.steps):
        grouped = args.objective in ['grouped_iid', 'antithetic', 'mean_cv']
        count = args.batch//2 if grouped else args.batch
        index = torch.randint(len(train_means), (count,), generator=selection)
        epsilon = torch.randn((count, dimension), dtype=torch.float64, device=args.device, generator=noise)
        mean = train_means[index].to(args.device)
        y = mean+sigma*epsilon
        if args.objective == 'mean_cv':
            y = torch.cat([y, mean])
        elif args.objective == 'antithetic':
            y = torch.cat([y, mean-sigma*epsilon]); epsilon = torch.cat([epsilon, -epsilon])
        elif args.objective == 'grouped_iid':
            second = torch.randn(epsilon.shape, dtype=epsilon.dtype, device=epsilon.device, generator=noise)
            y = torch.cat([y, mean+sigma*second]); epsilon = torch.cat([epsilon, second])
        optimizer.zero_grad(set_to_none=True)
        score = critic.score(y, basis, numbers, electronic, create_graph=True)
        if args.objective == 'raw':
            loss = (sigma*score+epsilon).square().mean()
        elif args.objective == 'mean_cv':
            loss = normalized_dsm_loss(score[:count], epsilon, sigma, mean_score=score[count:])
        else:
            loss = normalized_dsm_loss(score, epsilon, sigma)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite critic loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        report['clipped_updates'] += int(float(norm) > 10.)
        report['maximum_preclip_gradient_norm'] = max(report['maximum_preclip_gradient_norm'], float(norm))
        if step == 0 or (step+1) % 50 == 0:
            row = {'step': step+1, 'training_objective': args.objective, 'training_loss': float(loss.detach()), 'gradient_norm': float(norm), 'seconds': time.perf_counter()-start}
            report['history'].append(row); write_json(output, report); print(json.dumps(row), flush=True)
    evaluation_noise = torch.randn(test_means.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(report['evaluation_seed']))
    y_all = test_means+sigma*evaluation_noise
    scores = []
    for begin in range(0, len(test_means), 64):
        scores.append(critic.score(y_all[begin:begin+64].to(args.device), basis, numbers, electronic).detach().cpu())
    learned_score = torch.cat(scores)
    score_arms = {'zero': torch.zeros_like(learned_score), 'gaussian': -gaussian_precision*y_all, 'learned': learned_score}
    losses = {key: (sigma*score+evaluation_noise).square().mean(-1) for key, score in score_arms.items()}
    positions = torch.einsum('nk,bkd->bnd', basis.cpu(), y_all.reshape(len(y_all), n-1, 3))
    probes = {key: invariant_stein_rows(positions, torch.einsum('nk,bkd->bnd', basis.cpu(), score.reshape(len(y_all), n-1, 3)))
        for key, score in score_arms.items()}
    report['heldout_dsm'] = {key: summary(value) for key, value in losses.items()}
    report['paired_dsm_difference'] = {key: summary(losses['learned']-losses[key]) for key in ['zero', 'gaussian']}
    report['stein'] = {key: {name: summary(value) for name, value in rows.items()} for key, rows in probes.items()}
    dsm_pass = all(row['mean']+2*row['sem'] < 0 for row in report['paired_dsm_difference'].values())
    stein_pass = all(row['within_three_sem_of_zero'] for row in report['stein']['learned'].values())
    report['qualification_gate'] = {'paired_dsm_improvement_over_both_controls': dsm_pass,
        'four_stein_moments_within_three_sem': stein_pass, 'passed': dsm_pass and stein_pass,
        'scope': 'Necessary frozen-critic gate only; a pass does not establish global score accuracy or molecular sampling benefit.'}
    torch.save({'critic_state_dict': critic.state_dict(), 'condition': condition, 'terminal_noise_std': sigma,
        'source_checkpoint_sha256': checkpoint_sha, 'optimizer_state_dict': optimizer.state_dict(),
        'selection_rng_state': selection.get_state(), 'noise_rng_state': noise.get_state(),
        'training_steps': args.steps, 'training_objective': args.objective,
        'training_pool_sha256': sha(train_path)}, args.out/'critic.ckpt')
    torch.save({'positions': positions, 'intrinsic': y_all, 'epsilon': evaluation_noise, 'score_arms': score_arms,
        'losses': losses, 'stein_rows': probes, 'condition': condition}, args.out/'heldout.pt')
    torch.save({'training': train_means, 'heldout': test_means}, args.out/'parent_means.pt')
    report.update(complete=True, seconds=time.perf_counter()-start,
        artifacts={name: sha(args.out/name) for name in ['critic.ckpt', 'heldout.pt', 'parent_means.pt']})
    write_json(output, report); print(json.dumps(report['qualification_gate']), flush=True)


if __name__ == '__main__':
    main()
