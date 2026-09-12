#!/usr/bin/env python3
"""Fresh geometry-only FM source for measured reuse benchmarks; no energy worker."""
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
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.radial_reference import prepare_research_backbone
from cfm_mol.smooth_geometry import patch_smooth_geometry
from scripts.research.evaluate_chemical_policy import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'checkpoint', 'manifest', 'out']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--original-source', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--condition-index', type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    pp = args.protocol or root/'research/evidence/geometry_only_source_protocol_v1.json'
    protocol = json.loads(pp.read_text())
    stream = protocol.get('stream', 'fresh_development')
    assert stream in {'fresh_development', 'fresh_training'}
    if stream == 'fresh_training':
        manifest = json.loads(args.manifest.read_text())
        assert manifest.get('intended_use') == 'proposal_training'
        assert manifest.get('excluded_evaluation_manifest_sha256')
    parent_path = root/'research/evidence/species_breadth_source_protocol_v2.json'
    assert sha(parent_path) == protocol['parent_source_protocol_sha256']
    parent = json.loads(parent_path.read_text())
    for name in ['config', 'checkpoint', 'manifest']:
        expected = protocol.get('manifest_sha256', parent['manifest_sha256']) if name == 'manifest' else parent[name+'_sha256']
        assert sha(getattr(args, name)) == expected
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    config = read_config_file(args.config)
    config['mol_fm'].pop('bgfm', None)
    config['mol_fm']['prior_config']['x']['align'] = False
    assert config['dataset']['max_atoms'] == 200 and config['mol_fm']['total_loss_weights']['e'] == 0
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    recipe = saved['research_protocol']
    assert recipe['electronic_conditioning'] and recipe['position_parameterization'] == 'displacement'
    assert recipe['data_endpoint_time'] == 1. and recipe['requested_kT'] == parent['model_input_kT_eV']
    model = model_from_config(config)
    prepare_research_backbone(model, recipe)
    model.load_state_dict(saved['state_dict'], strict=True)
    patch_smooth_geometry(model, recipe.get('geometry_softening', 0.))
    model = model.cuda().float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    del saved
    index = args.condition_index if args.condition_index is not None else protocol['condition_index']
    batch, count = protocol['batch'], protocol['count']
    assert count % batch == 0
    condition = load_condition(args.manifest, index)
    condition.update(requested_kT_eV=recipe['requested_kT'], numbers=condition['atomic_numbers'])
    original_hash = None
    if protocol.get('new_panel'):
        assert args.original_source is None and index in protocol['condition_indices']
        assert protocol['seed_namespace'] not in protocol['excluded_seed_namespaces']
        # Namespace stride exceeds the entire candidate/stream/batch seed range.
        assert 0 <= condition['candidate_index'] < 1000 and protocol['stream_number'] == 2
        assert count//batch < 1000 and condition['manifest_role'] == 'new_development'
        assert 'energy_eV' not in condition
        old_seeds = set()
    else:
        assert args.original_source is not None and index == protocol['condition_index']
        original_path = args.original_source/f'condition_{index:02d}/results.json'
        original = json.loads(original_path.read_text())
        assert original['complete'] and original['condition'] == condition
        old_seeds = {seed for stream in original['streams'].values() for seed in stream['batch_seeds']}
        original_hash = sha(original_path)
    seeds = [1000000000*protocol['seed_namespace']+100003*condition['candidate_index']+100000003*protocol['stream_number']+i for i in range(count//batch)]
    assert not old_seeds.intersection(seeds) and len(set(seeds)) == len(seeds)
    base = graph_from_condition(condition, config['dataset']['atom_map'],
        n_bond_classes=5 if config['mol_fm'].get('explicit_aromaticity', False) else 4)
    n = base.num_nodes()
    graph = dgl.batch([base]*batch).to('cuda')
    nbi, _ = get_batch_idxs(graph)
    upper = get_upper_edge_mask(graph)
    basis = centered_orthonormal_basis(n, device='cuda')
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out/'results.json'
    if output.exists():
        raise FileExistsError(output)
    chunks = args.out/'chunks'
    chunks.mkdir(exist_ok=False)
    report = dict(complete=False, protocol_sha256=sha(pp), parent_source_protocol_sha256=sha(parent_path),
        checkpoint_sha256=sha(args.checkpoint), config_sha256=sha(args.config), manifest_sha256=sha(args.manifest),
        condition=condition, stream=stream, count=count, batch=batch, batch_seeds=seeds,
        original_source_results_sha256=original_hash,
        condition_index=index, seed_namespace=protocol['seed_namespace'],
        no_seed_overlap_with_original_streams=True, source_sampler='64-step midpoint displacement FM at T=1 plus .025-A COM Gaussian noise',
        physical_queries=0, energy_labels_computed=False, reference_coordinates_loaded=False,
        scientific_submission_ready=False, source_density_available=False, completed_samples=0,
        runtime=dict(torch_version=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
            deterministic=True, allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32, allow_tf32_cudnn=torch.backends.cudnn.allow_tf32), chunks={})
    write(output, report)
    started = time.perf_counter()
    samples = []
    try:
        with torch.no_grad():
            for number, seed in enumerate(seeds):
                rng = torch.Generator(device='cuda').manual_seed(seed)
                noise = torch.randn((batch, n-1, 3), dtype=torch.float64, device='cuda', generator=rng)
                initial = (torch.einsum('nk,bkd->bnd', basis, noise)*recipe['prior_std']).reshape(batch*n, 3).float()
                original_initial = initial.clone()
                cpu_rng, gpu_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
                x = sample_clamped_flow(model, graph, nbi, upper, x0=initial, n_ode_steps=64, terminal_time=1., parameterization='displacement')
                if number == 0:
                    repeated = sample_clamped_flow(model, graph, nbi, upper, x0=initial, n_ode_steps=64, terminal_time=1., parameterization='displacement')
                    report['prefix_replay_max_error_A'] = float((x-repeated).abs().max())
                    assert report['prefix_replay_max_error_A'] == 0.
                assert torch.equal(initial, original_initial)
                assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(gpu_rng, torch.cuda.get_rng_state())
                x = x.reshape(batch, n, 3).double()
                x -= x.mean(1, keepdim=True)
                terminal = torch.randn((batch, n-1, 3), dtype=torch.float64, device='cuda', generator=rng)
                x += .025*torch.einsum('nk,bkd->bnd', basis, terminal)
                assert torch.isfinite(x).all() and float(x.mean(1).abs().max()) < 1e-8
                file = chunks/f'positions_{number:05d}.pt'
                saved_chunk = dict(positions=x.cpu(), condition=condition, seed=seed,
                    sample_ids=list(range(number*batch, (number+1)*batch)), stream=stream)
                torch.save(saved_chunk, file)
                samples.append(saved_chunk['positions'])
                report['chunks'][file.name] = sha(file)
                report['completed_samples'] = (number+1)*batch
                report['seconds'] = time.perf_counter()-started
                write(output, report)
                if number % 8 == 0:
                    print(json.dumps(dict(completed_samples=report['completed_samples'], seconds=report['seconds'], physical_queries=0)), flush=True)
        path = args.out/'samples.pt'
        torch.save(dict(positions=torch.cat(samples), condition=condition, sample_ids=list(range(count)),
            batch_seeds=seeds, stream=stream, protocol_sha256=sha(pp)), path)
        report.update(complete=True, samples_sha256=sha(path), generation_neural_field_calls=count*128,
            additional_replay_neural_field_calls=batch*128, seconds=time.perf_counter()-started,
            global_rng_unchanged=True, input_positions_unchanged=True,
            limitation='Fresh generated coordinates, with no physical labels or density/importance weights. Chemical support and measured sampling reuse must be assessed separately.')
        write(output, report)
    except Exception as exc:
        report.update(failure=f'{type(exc).__name__}: {exc}', seconds=time.perf_counter()-started)
        write(output, report)
        raise


if __name__ == '__main__':
    main()
