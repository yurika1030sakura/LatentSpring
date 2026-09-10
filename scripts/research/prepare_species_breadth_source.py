#!/usr/bin/env python3
"""Frozen finite-FM plus COM-Gaussian sources for every prescribed development condition."""
import argparse
import json
import os
from pathlib import Path
import time

import dgl
import torch
from flowmol.model_utils.load import model_from_config, read_config_file
from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask

from cfm_mol.clamped_density import sample_clamped_flow
from cfm_mol.condition_systems import graph_from_condition, load_condition
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.radial_reference import prepare_research_backbone
from cfm_mol.smooth_geometry import patch_smooth_geometry
from molecular_tempered_pilot import sha, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['config', 'checkpoint', 'manifest', 'oracle', 'oracle-python', 'out']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--train-count', type=int, default=32)
    p.add_argument('--eval-count', type=int, default=32)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--seed', type=int, default=9181)
    p.add_argument('--condition-index', type=int)
    p.add_argument('--replay-diagnostic', action='store_true', help='Record prefix differences then stop without source qualification')
    p.add_argument('--deterministic-runtime', action='store_true')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.deterministic_runtime:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
    if min(args.train_count, args.eval_count, args.batch) < 1 or any(n % args.batch for n in [args.train_count, args.eval_count]):
        raise ValueError('Positive divisible stream counts required')
    output = args.out/'panel.json'
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(args.manifest.read_text())
    if not manifest['complete'] or manifest['role'] != 'new_development' or len(manifest['rows']) != 8:
        raise ValueError('Require prescribed eight-condition development manifest')
    root = Path(__file__).resolve().parents[2]
    frozen_path = root/('research/evidence/species_breadth_source_protocol_v2.json' if args.deterministic_runtime
        else 'research/evidence/species_breadth_source_protocol_v1.json')
    frozen = json.loads(frozen_path.read_text())
    if not frozen['frozen'] or frozen['reserved_outcomes_allowed']:
        raise ValueError('Invalid source protocol')
    for key in ['checkpoint', 'config', 'manifest', 'oracle']:
        if sha(getattr(args, key)) != frozen[key+'_sha256']:
            raise ValueError(f'{key} differs from committed source protocol')
    indices = list(range(8)) if args.condition_index is None else [args.condition_index]
    if any(i not in range(8) for i in indices):
        raise ValueError('Invalid condition index')
    cfg = read_config_file(args.config)
    cfg['mol_fm'].pop('bgfm', None)
    cfg['mol_fm']['prior_config']['x']['align'] = False
    if cfg['dataset']['max_atoms'] != 200 or cfg['mol_fm']['total_loss_weights']['e'] != 0:
        raise ValueError('Require bond-free max_atoms200')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    protocol = state['research_protocol']
    if protocol['requested_kT'] != frozen['model_input_kT_eV']:
        raise ValueError('Frozen model-input temperature differs')
    if not protocol.get('electronic_conditioning') or protocol['position_parameterization'] != 'displacement' or protocol['data_endpoint_time'] != 1.:
        raise ValueError('Require metadata-aware displacement checkpoint')
    model = model_from_config(cfg)
    prepare_research_backbone(model, protocol)
    model.load_state_dict(state['state_dict'], strict=True)
    patch_smooth_geometry(model, protocol.get('geometry_softening', 0.))
    model = model.to(args.device).float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    del state
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    report = {'complete': False, 'scope': __doc__, 'rows': [], 'requested_indices': indices,
        'source_protocol_sha256': sha(frozen_path),
        'checkpoint_sha256': sha(args.checkpoint), 'manifest_sha256': sha(args.manifest),
        'config_sha256': sha(args.config), 'oracle_sha256': sha(args.oracle),
        'configuration': {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'deterministic_runtime': args.deterministic_runtime,
        'model_input_kT_eV': protocol['requested_kT'], 'physical_target_kT_eV': .025851999786435,
        'restraint_eV_A2': .1, 'source_terminal_noise_std_A': .025,
        'source_sampler': '64-step midpoint displacement FM at T=1 plus independent COM Gaussian noise',
        'prior_std_A': protocol['prior_std'], 'reference_geometry_loaded': False,
        'weights': 'unavailable: no finite-path density has been defined for this source',
        'limitations': ['Source engineering and development data only; no adapter performance or calibration claim.',
            'Physical target temperature differs from the frozen pretrained neural temperature feature.',
            'Every prescribed condition is attempted; failed rows are retained and not replaced.']}
    write_json(output, report)
    for index in indices:
        directory = args.out/f'condition_{index:02d}'
        directory.mkdir(exist_ok=False)
        row = {'index': index, 'success': False, 'oracle_evaluations': 0}
        oracle = None
        try:
            condition = load_condition(args.manifest, index)
            condition['requested_kT_eV'] = protocol['requested_kT']
            condition['numbers'] = condition['atomic_numbers']
            base = graph_from_condition(condition, cfg['dataset']['atom_map'],
                n_bond_classes=5 if cfg['mol_fm'].get('explicit_aromaticity', False) else 4)
            n = base.num_nodes()
            graph = dgl.batch([base]*args.batch).to(args.device)
            nbi, _ = get_batch_idxs(graph)
            uem = get_upper_edge_mask(graph)
            basis = centered_orthonormal_basis(n, device=args.device)
            detail = {'complete': False, 'condition': condition, 'artifacts': {}, 'streams': {},
                'checkpoint_sha256': report['checkpoint_sha256'], 'manifest_sha256': report['manifest_sha256'],
                'oracle_sha256': report['oracle_sha256'], 'model_input_kT_eV': protocol['requested_kT'],
                'physical_target_kT_eV': report['physical_target_kT_eV'], 'restraint_eV_A2': .1,
                'source_terminal_noise_std_A': .025, 'reference_geometry_loaded': False,
                'source_sampler': report['source_sampler'], 'weights': report['weights']}
            write_json(directory/'results.json', detail)
            with EnergyOracle(args.oracle_python, root/'scripts/research/oracle_worker.py', args.oracle,
                numbers=condition['numbers'], charge=condition['charge'],
                spin_multiplicity=condition['spin_multiplicity'], batch_size=16) as oracle, torch.no_grad():
                for stream_index, (label, count) in enumerate([('training', args.train_count), ('development', args.eval_count)]):
                    samples, seeds, energies = [], [], []
                    maximum_force = 0.
                    chunk_dir = directory/label
                    chunk_dir.mkdir(exist_ok=False)
                    for batch_index in range(count//args.batch):
                        seed = 1000000000*args.seed+100003*condition['candidate_index']+100000003*stream_index+batch_index
                        generator = torch.Generator(device=args.device).manual_seed(seed)
                        z = torch.randn((args.batch, n-1, 3), generator=generator, device=args.device, dtype=torch.float64)
                        x0 = (torch.einsum('nk,bkd->bnd', basis, z)*protocol['prior_std']).reshape(args.batch*n,3).float()
                        if stream_index == 0 and batch_index == 0:
                            cpu_rng = torch.get_rng_state().clone()
                            cuda_rng = torch.cuda.get_rng_state().clone() if args.device == 'cuda' else None
                            original_x0 = x0.clone()
                        x = sample_clamped_flow(model, graph, nbi, uem, x0=x0,
                            n_ode_steps=64, terminal_time=1., parameterization='displacement')
                        if stream_index == 0 and batch_index == 0:
                            replay = sample_clamped_flow(model, graph, nbi, uem, x0=x0,
                                n_ode_steps=64, terminal_time=1., parameterization='displacement')
                            error = float((replay-x).abs().max())
                            detail['prefix_replay_max_error_A'] = error
                            detail['prefix_replay_rms_error_A'] = float((replay-x).square().mean().sqrt())
                            detail['prefix_input_changed'] = not torch.equal(original_x0, x0)
                            detail['prefix_cpu_rng_changed'] = not torch.equal(cpu_rng, torch.get_rng_state())
                            detail['prefix_cuda_rng_changed'] = cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state())
                            detail['additional_replay_neural_field_calls'] = 128*args.batch
                            write_json(directory/'results.json', detail)
                            if args.replay_diagnostic:
                                torch.save({'first': x.cpu(), 'replay': replay.cpu(), 'initial': x0.cpu()}, directory/'replay_diagnostic.pt')
                                raise RuntimeError('Diagnostic-only prefix recorded; no source qualified')
                            if error > 1e-7 or any(detail[k] for k in ['prefix_input_changed', 'prefix_cpu_rng_changed', 'prefix_cuda_rng_changed']):
                                raise RuntimeError(f'Frozen sampler prefix replay failed: maximum error {error:.9g} A')
                        x = x.reshape(args.batch, n, 3).double()
                        x = x-x.mean(1, keepdim=True)
                        noise = torch.randn((args.batch, n-1, 3), generator=generator, device=args.device, dtype=torch.float64)
                        x = x+.025*torch.einsum('nk,bkd->bnd', basis, noise)
                        if not torch.isfinite(x).all() or float(x.mean(1).abs().max()) > 1e-8:
                            raise FloatingPointError('Invalid source coordinates')
                        # Persist expensive source coordinates before physical
                        # evaluation. A failed RPC must not erase the source.
                        coordinate_path = chunk_dir/f'positions_{batch_index:05d}.pt'
                        torch.save({'positions': x.cpu(), 'condition': condition, 'seed': seed,
                            'sample_ids': list(range(batch_index*args.batch, (batch_index+1)*args.batch))}, coordinate_path)
                        energy, force = oracle.evaluate_chunked(x)
                        torch.save({'energy_eV': energy, 'force_eV_A': force,
                            'positions_sha256': sha(coordinate_path)}, chunk_dir/f'energies_{batch_index:05d}.pt')
                        maximum_force = max(maximum_force, float(force.norm(dim=-1).max()))
                        energies.append(energy)
                        samples.append(x.cpu())
                        seeds.append(seed)
                        detail['oracle_evaluations_so_far'] = oracle.evaluated
                        detail['oracle_requested_evaluations_so_far'] = oracle.requested_evaluations
                        write_json(directory/'results.json', detail)
                    x = torch.cat(samples)
                    energy = torch.cat(energies)
                    data = {'positions': x, 'energy_eV': energy, 'condition': condition,
                        'sample_ids': list(range(count)), 'stream': label, 'batch_seeds': seeds}
                    file = directory/f'{label}_samples.pt'
                    torch.save(data, file)
                    detail['artifacts'][file.name] = sha(file)
                    detail['streams'][label] = {'samples': count, 'batch_seeds': seeds,
                        'neural_field_calls_per_sample': 128, 'max_COM_error_A': float(x.mean(1).abs().max()),
                        'maximum_force_norm_eV_A': maximum_force,
                        'energy_eV': {'min': float(energy.min()), 'median': float(energy.median()), 'max': float(energy.max())}}
                    write_json(directory/'results.json', detail)
                detail.update(complete=True, oracle_evaluations=oracle.evaluated)
                write_json(directory/'results.json', detail)
                row.update(success=True, results_sha256=sha(directory/'results.json'), oracle_evaluations=oracle.evaluated)
        except Exception as exc:
            row.update(error=f'{type(exc).__name__}: {exc}', oracle_evaluations=0 if oracle is None else oracle.evaluated,
                oracle_requested_evaluations=0 if oracle is None else oracle.requested_evaluations,
                query_accounting_exact=oracle is not None and oracle.evaluated == oracle.requested_evaluations)
            write_json(directory/'failure.json', row)
        report['rows'].append(row)
        write_json(output, report)
        print(json.dumps(row), flush=True)
    report.update(complete=True, successful_conditions=sum(r['success'] for r in report['rows']),
        oracle_evaluations=sum(r['oracle_evaluations'] for r in report['rows']), seconds=time.perf_counter()-start)
    write_json(output, report)


if __name__ == '__main__':
    main()
