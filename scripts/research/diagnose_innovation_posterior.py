#!/usr/bin/env python3
"""Frozen molecular paths: compare sequential and global Gaussian auxiliaries."""
import argparse
import json
import math
from pathlib import Path
import time

import torch
from flowmol.model_utils.load import read_config_file, model_from_config

from cfm_mol.condition_systems import graph_from_condition
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.innovation_posterior import (GaussianInnovationMap, independent_map_jacobian,
    fit_gauss_newton_auxiliary, innovation_log_joint)
from cfm_mol.molecular_path_drift import MolecularPathDrift
from cfm_mol.nonequilibrium import centered_orthonormal_basis, WeightedPaths
from cfm_mol.path_balance import fixed_path_log_factors, fixed_backward_residuals, backward_log_probability
from cfm_mol.path_work import gaussian_training_path
from cfm_mol.radial_reference import prepare_research_backbone
from cfm_mol.smooth_geometry import patch_smooth_geometry
from molecular_tempered_pilot import sha, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--backward-refit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--particles', type=int, default=32)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--iterations', type=int, default=4)
    p.add_argument('--seed', type=int, default=9091)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if min(args.particles, args.batch, args.iterations) < 1 or args.particles % args.batch:
        raise ValueError('Positive divisible counts required')
    output = args.out/'results.json'
    if output.exists():
        raise FileExistsError(output)
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    old = json.loads((args.source_run/'results.json').read_text())
    refit = json.loads((args.backward_refit/'results.json').read_text())
    checkpoint = args.source_run/'last.ckpt'
    reverse_checkpoint = args.backward_refit/'backward_refit.ckpt'
    if not old['complete'] or not refit['complete'] or not refit['forward_state_unchanged']:
        raise ValueError('Require complete frozen source and reverse refit')
    if sha(checkpoint) != refit['source_checkpoint_sha256'] or sha(reverse_checkpoint) != refit['backward_checkpoint_sha256']:
        raise ValueError('Checkpoint provenance mismatch')
    state = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    recipe = state['proposal_protocol']; protocol = state['research_protocol']
    if recipe != old['configuration'] or recipe.get('precision_kind', 'none') != 'none':
        raise ValueError('Require scalar Gaussian original sampler')
    if protocol['position_parameterization'] != 'displacement' or protocol['data_endpoint_time'] != 1.:
        raise ValueError('Require declared T1 displacement semantics')
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
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model.to(args.device).float().eval()

    forward = load(state['forward_state_dict'])
    reverse_model = load(torch.load(str(reverse_checkpoint), map_location='cpu', weights_only=False)['backward_state_dict'])
    del state
    frozen = {key: value.detach().cpu().clone() for key, value in forward.state_dict().items()}
    condition = old['condition']; n = len(condition['numbers']); dimension = 3*(n-1)
    graph = graph_from_condition({'atomic_numbers': condition['numbers'], 'charge': condition['charge'],
        'spin_multiplicity': condition['spin_multiplicity'], 'requested_kT_eV': recipe['kT']},
        cfg['dataset']['atom_map'], n_bond_classes=5 if cfg['mol_fm'].get('explicit_aromaticity', False) else 4).to(args.device)
    field = MolecularPathDrift(forward, graph)
    reverse = MolecularPathDrift(reverse_model, graph, sign=-1.)
    settings = {'prior_std': old['prior_std'],
        'terminal_std': math.sqrt(recipe['kT']/recipe['restraint']) if recipe['reference_kernel'] == 'gaussian' else None,
        'max_drift_norm': recipe['max_drift_per_sqrt_dimension']*math.sqrt(dimension),
        'mean_parameterization': recipe['mean_parameterization'], 'noise_annealing_power': recipe['noise_annealing_power']}
    times = torch.linspace(0, 1, recipe['path_steps']+1, dtype=torch.float64)
    mapping = GaussianInnovationMap(field, dimension, times, recipe['noise'], **settings)
    ratios = torch.tensor(refit['evaluations']['after_fitted_width']['variance_ratios'], dtype=torch.float64, device=args.device)
    basis = centered_orthonormal_basis(n, device=args.device)
    report = {'complete': False, 'format': 'global_innovation_auxiliary_diagnostic_v1', 'scope': __doc__,
        'configuration': {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'condition': condition, 'source_recipe': recipe, 'research_protocol': protocol,
        'source_checkpoint_sha256': sha(checkpoint), 'reverse_checkpoint_sha256': sha(reverse_checkpoint),
        'source_results_sha256': sha(args.source_run/'results.json'), 'reverse_results_sha256': sha(args.backward_refit/'results.json'),
        'latent_dimension': mapping.latent_dimension, 'endpoint_dimension': dimension,
        'terminal_noise_std': mapping.terminal_noise_std, 'log_state_jacobian': mapping.log_state_jacobian,
        'metric_scales': [0., .01, .1, 1.], 'checks': [], 'posterior_diagnostics': [],
        'limitations': ['All auxiliaries share identical endpoints, energies and forward weights.',
            'Global Gaussian conditioning and Gauss-Newton approximations are established methods, not claimed novelty.',
            'Finite-iteration approximate posterior modes and Gaussian conditionals need not describe nonlinear or multimodal posterior structure.',
            'The global Gaussian is normalized even without convergence; its importance variance need not be finite.',
            'Neither a larger empirical ESS nor a successful small screen certifies target coverage.',
            'Dense endpoint Jacobians are an expensive diagnostic, not a scalable production architecture.']}
    write_json(output, report)
    states_out, latent_out, means_out, jacobians_out, joint_out, baseline_out = [], [], [], [], [], []
    auxiliary_logs = {str(scale): [] for scale in report['metric_scales']}
    for batch_index in range(args.particles//args.batch):
        generator = torch.Generator(device=args.device).manual_seed(args.seed+100003*batch_index)
        with torch.no_grad():
            x0 = torch.randn((args.batch, dimension), dtype=torch.float64, device=args.device, generator=generator)*old['prior_std']
            path = gaussian_training_path(x0, field, lambda z, t: torch.zeros_like(z), times, recipe['noise'], generator,
                retain_states=True, **settings)
            z = mapping.from_states(path.states)
            conditional, intermediate = mapping(z, retain_states=True)
            logjoint = innovation_log_joint(z, path.terminal, conditional, mapping.terminal_noise_std)
            initial, lf, _ = fixed_path_log_factors(path.states, field, lambda z, t: torch.zeros_like(z), times, recipe['noise'], **settings)
            density_error = float((logjoint-initial-lf-mapping.log_state_jacobian).abs().max())
            state_error = float((intermediate-path.states[:, :-1]).abs().max())
            if density_error > .05 or state_error > 1e-4:
                raise RuntimeError('Innovation coordinates do not reproduce original path measure')
            residual, normalization = fixed_backward_residuals(path.states, reverse, times, recipe['noise'], **settings)
            logb = backward_log_probability(residual, normalization, dimension, ratios)
            baseline = logb+mapping.log_state_jacobian
        if batch_index == 0:
            # Independent directional FD ladder on actual generated paths.
            value, jacobian = independent_map_jacobian(mapping, z[:2])
            directions = torch.randn(z[:2].shape, dtype=z.dtype, device=z.device,
                generator=torch.Generator(device=z.device).manual_seed(args.seed+1))
            directions = directions/directions.norm(dim=-1, keepdim=True)
            analytic = (jacobian@directions[..., None]).squeeze(-1)
            ladder = []
            for h in [.003, .001]:
                with torch.no_grad():
                    numerical = (mapping(z[:2]+h*directions)-mapping(z[:2]-h*directions))/(2*h)
                ladder.append({'h': h, 'relative_l2_error': float((numerical-analytic).norm()/analytic.norm().clamp_min(1e-12)),
                               'max_absolute_error': float((numerical-analytic).abs().max())})
            report['map_directional_fd'] = ladder
            if min(row['relative_l2_error'] for row in ladder) > .05:
                raise RuntimeError('Map Jacobian failed directional finite-difference gate')
        # The initialization uses the endpoint only; no actual generating noise.
        # A zero geometric start would coincide all atoms. Lift endpoint geometry
        # into the prior block, with all intermediate innovations initially zero.
        initial_mean = path.terminal.new_zeros(len(path.terminal), mapping.latent_dimension)
        initial_mean[:, :dimension] = path.terminal/settings['prior_std']
        posterior, diagnostic = fit_gauss_newton_auxiliary(mapping, path.terminal, mapping.latent_dimension,
            mapping.terminal_noise_std, iterations=args.iterations, initial_mean=initial_mean)
        for scale in report['metric_scales']:
            auxiliary_logs[str(scale)].append(posterior.log_prob(z, metric_scale=scale).detach().cpu())
        report['checks'].append({'batch': batch_index, 'state_max_error': state_error, 'log_density_max_error': density_error})
        report['posterior_diagnostics'].append(diagnostic)
        states_out.append(path.states.cpu()); latent_out.append(z.cpu()); means_out.append(posterior.mean.cpu())
        jacobians_out.append(posterior.jacobian.cpu()); joint_out.append(logjoint.cpu()); baseline_out.append(baseline.cpu())
        write_json(output, report)
        print(json.dumps({'particles': (batch_index+1)*args.batch, 'seconds': time.perf_counter()-start,
            'posterior_residual_max': max(diagnostic['endpoint_residual_norm'])}), flush=True)
    states = torch.cat(states_out)
    positions = torch.einsum('nk,bkd->bnd', basis.cpu(), states[:, -1].reshape(args.particles, n-1, 3))
    oracle_path = Path(recipe['oracle'])
    if sha(oracle_path) != old['oracle_sha256']:
        raise ValueError('Target potential changed')
    root = Path(__file__).resolve().parents[2]
    with EnergyOracle(Path(recipe['oracle_python']), root/'scripts/research/oracle_worker.py', oracle_path,
        numbers=condition['numbers'], charge=condition['charge'], spin_multiplicity=condition['spin_multiplicity'], batch_size=16) as oracle:
        energy, _ = oracle.evaluate(positions)
        report['oracle_evaluations'] = oracle.evaluated
    reduced = (energy-old['energy_zero_eV']+recipe['restraint']/2*positions.square().sum((1, 2)))/recipe['kT']
    logjoint = torch.cat(joint_out)
    logs = {'sequential_refit': torch.cat(baseline_out), **{'global_metric_'+key: torch.cat(value) for key, value in auxiliary_logs.items()}}
    works = {key: reduced+logjoint-value for key, value in logs.items()}
    report['evaluations'] = {key: {'weights': WeightedPaths(positions, -work, {}).summary(),
        'mean_work': float(work.mean()), 'work_std': float(work.std())} for key, work in works.items()}
    torch.save({'positions': positions, 'states': states, 'latent': torch.cat(latent_out),
        'auxiliary_mean': torch.cat(means_out), 'auxiliary_jacobian': torch.cat(jacobians_out),
        'log_joint_innovation': logjoint, 'auxiliary_log_probabilities': logs, 'works': works,
        'energy_eV': energy, 'condition': condition}, args.out/'comparison.pt')
    if not all(torch.equal(value.cpu(), frozen[key]) for key, value in forward.state_dict().items()):
        raise RuntimeError('Frozen forward model changed')
    report.update(complete=True, forward_state_unchanged=True, shared_endpoints=True, seconds=time.perf_counter()-start,
        comparison_sha256=sha(args.out/'comparison.pt'), oracle_sha256=sha(oracle_path),
        map_calls=mapping.map_calls, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None)
    write_json(output, report)
    print(json.dumps(report['evaluations']), flush=True)


if __name__ == '__main__':
    main()
