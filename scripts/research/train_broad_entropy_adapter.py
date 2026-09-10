#!/usr/bin/env python3
"""Matched entropy refinement of an audited finite-FM-plus-noise molecular source."""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import time

import torch

from cfm_mol.affine_species_adapter import AffineSpeciesCouplingAdapter
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.entropy_adapter_io import load_entropy_adapter
from cfm_mol.entropy_source import load_entropy_source
from cfm_mol.linear_entropy_adapter import LinearEntropyAdapter, endpoint_kl_change
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.path_work import external_energy
from cfm_mol.species_coupling_adapter import SpeciesCouplingAdapter
from molecular_tempered_pilot import sha, write_json


def summarize(x):
    return {'mean': float(x.mean()), 'sem': float(x.std()/len(x)**.5), 'samples': len(x)}


@contextmanager
def record_failure(oracle, report, output):
    try:
        yield
    except Exception as exc:
        report.update(complete=False, oracle_evaluations=oracle.evaluated,
            oracle_requested_evaluations=oracle.requested_evaluations,
            failure=f'{type(exc).__name__}: {exc}')
        write_json(output, report)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--condition-index', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--kind', choices=['convex', 'affine', 'typed'], required=True)
    p.add_argument('--replica', type=int, choices=[0, 1], default=0)
    p.add_argument('--engineering-smoke', action='store_true')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--eval-count', type=int, default=512)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.engineering_smoke:
        if args.steps != 2 or args.eval_count != 16:
            raise ValueError('Engineering source permits only the fixed2-update16-row smoke')
    elif args.steps != 1000 or args.eval_count != 512:
        raise ValueError('Production recipe is fixed at1000 updates and512 development rows')
    root = Path(__file__).resolve().parents[2]
    data = load_entropy_source(args.source_root, args.condition_index,
        protocol_path=root/'research/evidence/species_breadth_source_protocol_v2.json',
        manifest_path=root/'research/evidence/development_panel_v1.json', engineering=args.engineering_smoke)
    output = args.out/'results.json'
    if output.exists():
        raise FileExistsError(output)
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    condition, source, recipe = data['condition'], data['source'], data['recipe']
    x_train = data['training']['positions'].double()
    x_eval = data['development']['positions'][:args.eval_count].double()
    kT, restraint = source['physical_target_kT_eV'], source['restraint_eV_A2']
    init_seed, selection_seed = 9161+args.replica, 9141+args.replica
    torch.manual_seed(init_seed)
    if args.kind == 'typed':
        adapter = LinearEntropyAdapter(condition['numbers'], kind='typed')
        kind, lr_start, lr_end = 'typed', .01, .0001
    else:
        model_class = SpeciesCouplingAdapter if args.kind == 'convex' else AffineSpeciesCouplingAdapter
        adapter = model_class(condition['numbers'], charge=condition['charge'],
            spin_multiplicity=condition['spin_multiplicity'], kT=kT)
        kind, lr_start, lr_end = 'species_'+args.kind, .001, .00001
    adapter = adapter.to(args.device).double()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr_start, weight_decay=0.)
    selection = torch.Generator().manual_seed(selection_seed)
    # This constant is from generated-source energies, never the reference
    # energy carried in condition metadata. It changes neither forces nor KL.
    energy_zero = float(data['training']['energy_eV'].median())
    report = {'complete': False, 'scope': __doc__, 'kind': kind,
        'configuration': {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'source_kind': data['source_kind'], 'condition': condition, 'engineering_only': args.engineering_smoke,
        'source_checkpoint_sha256': data['source_checkpoint_sha256'],
        'source_protocol_sha256': data['source_protocol_sha256'], 'source_results_sha256': data['source_results_sha256'],
        'training_sha256': data['training_sha256'], 'evaluation_sha256': data['evaluation_sha256'],
        'oracle_sha256': source['oracle_sha256'], 'kT_eV': kT, 'restraint_eV_A2': restraint,
        'source_model_input_kT_eV': source['model_input_kT_eV'],
        'source_terminal_noise_std_A': source['source_terminal_noise_std_A'],
        'steps': args.steps, 'batch': 16, 'eval_count': args.eval_count, 'replica': args.replica,
        'init_seed': init_seed, 'selection_seed': selection_seed,
        'lr_start': lr_start, 'lr_end': lr_end, 'lr_schedule': 'cosine',
        'parameters': sum(p.numel() for p in adapter.parameters()),
        'energy_zero_eV': energy_zero, 'reference_geometry_used_to_initialize': False,
        'reference_energy_used_for_training': False, 'source_oracle_queries_additional': source['oracle_evaluations'],
        'history': [], 'weights': None,
        'limitations': ['The frozen finite-FM-plus-noise source has no qualified path density or importance weights.',
            'Relative endpoint-KL changes do not determine absolute KL or certify Boltzmann populations.',
            'Source production and shared pretraining costs remain additional to adapter query counts.',
            'Every source condition and failed optimization remains part of the development denominator.',
            'Two neural initializations use matched parent pools and streams; row SEM is conditional on the trained model.']}
    if args.kind == 'typed':
        report['maximum_pair_weight'] = .25
    else:
        report['adapter_configuration'] = adapter.configuration
        report['coupling_layers'] = adapter.layers
    write_json(output, report)
    oracle_path = Path(recipe['oracle'])
    if sha(oracle_path) != source['oracle_sha256']:
        raise ValueError('Physical oracle differs from the source target')
    with EnergyOracle(Path(recipe['oracle_python']), root/'scripts/research/oracle_worker.py', oracle_path,
            numbers=condition['numbers'], charge=condition['charge'],
            spin_multiplicity=condition['spin_multiplicity'], batch_size=16) as oracle, record_failure(oracle, report, output):
        initial_energy, _ = oracle.evaluate_chunked(x_eval)
        error = float((initial_energy-data['development']['energy_eV'][:args.eval_count]).abs().max())
        report['base_energy_replay_max_error_eV'] = error
        if error > .001:
            raise ValueError('Cached source and current physical energies differ')
        for step in range(args.steps):
            lr = lr_end+.5*(lr_start-lr_end)*(1+math.cos(math.pi*step/max(1, args.steps-1)))
            for group in optimizer.param_groups:
                group['lr'] = lr
            indices = torch.randint(len(x_train), (16,), generator=selection)
            x = x_train[indices].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            y, volume = adapter(x)
            energy, force = oracle.evaluate(y.detach())
            if step == 0:
                error = float((energy-data['training']['energy_eV'][indices]).abs().max())
                report['identity_training_energy_replay_max_error_eV'] = error
                if error > .001:
                    raise ValueError('Identity adapter does not replay the source energy')
            potential = external_energy(y, energy, force)+restraint/2*y.square().sum((1, 2))
            loss = ((potential-energy_zero)/kT-volume).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite entropy objective')
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            if step == 0 or (step+1) % 20 == 0:
                row = {'step': step+1, 'objective': float(loss.detach()), 'mean_log_volume': float(volume.detach().mean()),
                    'gradient_norm': float(gradient), 'lr': lr, 'seconds': time.perf_counter()-start}
                report['history'].append(row)
                report['oracle_evaluations_so_far'] = oracle.evaluated
                write_json(output, report)
                print(json.dumps(row), flush=True)
        with torch.no_grad():
            mapped = [adapter(x_eval[i:i+64].to(args.device)) for i in range(0, len(x_eval), 64)]
            y = torch.cat([r[0].cpu() for r in mapped])
            volume = torch.cat([(r[1].expand(len(r[0])) if r[1].ndim == 0 else r[1]).cpu() for r in mapped])
            if args.kind == 'typed':
                matrix, _, _ = adapter.matrices()
                restored = torch.linalg.solve(matrix, y[:32].to(args.device)).cpu()
                inverse_info = {'method': 'atom_matrix_linear_solve'}
            else:
                restored, inverse_volume, inverse_info = adapter.inverse(y[:32].to(args.device))
                restored = restored.cpu()
                torch.testing.assert_close(inverse_volume.cpu(), -volume[:len(restored)], atol=1e-9, rtol=1e-9)
            torch.testing.assert_close(restored, x_eval[:len(restored)], atol=1e-9, rtol=1e-9)
            report['trained_inverse_check'] = {**inverse_info, 'position_error_A': float((restored-x_eval[:len(restored)]).abs().max())}
        n = len(condition['numbers'])
        basis = centered_orthonormal_basis(n, device=args.device)
        errors = []
        for index in range(min(2, len(x_eval))):
            intrinsic = (basis.T@x_eval[index].to(args.device)).flatten()
            def transform(z):
                changed, _ = adapter((basis@z.reshape(n-1, 3))[None])
                return (basis.T@changed[0]).flatten()
            jacobian = torch.autograd.functional.jacobian(transform, intrinsic)
            sign, direct = torch.linalg.slogdet(jacobian)
            error = abs(float(direct)-float(volume[index]))
            if sign <= 0 or error > 1e-5:
                raise RuntimeError('Complete trained intrinsic determinant failed')
            errors.append(error)
        report['trained_full_jacobian_errors'] = errors
        final_energy, _ = oracle.evaluate_chunked(y)
        change = endpoint_kl_change(initial_energy, final_energy, x_eval, y, kT=kT, restraint=restraint, log_volume=volume)
        report['oracle_evaluations'] = oracle.evaluated
    if report['oracle_evaluations'] != 16*args.steps+2*args.eval_count:
        raise RuntimeError('Matched oracle-query budget differs')
    report.update(paired_endpoint_kl_change=summarize(change),
        paired_endpoint_kl_change_per_internal_dof=summarize(change/(3*(len(condition['numbers'])-1))),
        mean_energy_eV={'base': float(initial_energy.mean()), 'adapted': float(final_energy.mean())},
        log_volume={'mean': float(volume.mean()), 'std': float(volume.std()), 'min': float(volume.min()), 'max': float(volume.max())},
        maximum_COM_error_A=float(y.mean(1).abs().max()))
    torch.save({'positions': x_eval, 'energy_eV': initial_energy, 'condition': condition,
        'sample_ids': data['development']['sample_ids'][:args.eval_count]}, args.out/'base_samples.pt')
    torch.save({'positions': y, 'energy_eV': final_energy, 'condition': condition,
        'paired_endpoint_kl_change': change, 'log_volume': volume,
        'sample_ids': data['development']['sample_ids'][:args.eval_count]}, args.out/'adapted_samples.pt')
    state = {'state_dict': adapter.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'kind': kind,
        'condition': condition, 'source_checkpoint_sha256': data['source_checkpoint_sha256'],
        'source_kind': data['source_kind'], 'source_protocol_sha256': data['source_protocol_sha256']}
    if args.kind != 'typed':
        state['adapter_configuration'] = adapter.configuration
    torch.save(state, args.out/'adapter.ckpt')
    report.update(complete=True, seconds=time.perf_counter()-start,
        artifacts={name: sha(args.out/name) for name in ['base_samples.pt', 'adapted_samples.pt', 'adapter.ckpt']})
    write_json(output, report)
    # Exercise the public loader on this actual family checkpoint before
    # reporting qualification; an incomplete loader replay invalidates the run.
    try:
        loaded, _, _ = load_entropy_adapter(args.out, args.device)
        with torch.no_grad():
            replay, replay_volume = loaded(x_eval[:16].to(args.device))
        torch.testing.assert_close(replay.cpu(), y[:16], atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(replay_volume.expand(16).cpu() if replay_volume.ndim == 0 else replay_volume.cpu(),
            volume[:16], atol=1e-9, rtol=1e-9)
    except Exception:
        report['complete'] = False
        write_json(output, report)
        raise
    report['checkpoint_loader_replay_passed'] = True
    write_json(output, report)
    print(json.dumps(report['paired_endpoint_kl_change']), flush=True)


if __name__ == '__main__':
    main()
