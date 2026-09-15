#!/usr/bin/env python3
"""Replay identical local candidates while removing their force displacement."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch
from flowmol.model_utils.load import read_config_file

from cfm_mol.clamped_fm import clamped_fm_loss
from cfm_mol.chemical_moves import covalent_radii
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.source_checkpoint import prior_from_checkpoint
from scripts.research.tree_prior_fm import restore_model, sample_source, evaluate
from scripts.research.run_tree_manifold import make_graph
from scripts.research.audit_generator_output_support import assess
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['project', 'protocol', 'out']:
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.protocol.read_text())
    ph = sha(args.protocol)
    assert spec['frozen'] and not args.out.exists()
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    for key in ['data', 'config', 'warm_checkpoint', 'selection_log', 'replay_checkpoint', 'condition_manifest']:
        assert sha(args.project/spec[key]) == spec[key+'_sha256']
    cfg = read_config_file(args.project/spec['config'])
    cfg['mol_fm'].pop('bgfm', None)
    assert cfg['dataset']['max_atoms'] == 200 and cfg['mol_fm']['total_loss_weights']['e'] == 0
    data = torch.load(args.project/spec['data'], weights_only=False, map_location='cpu')['training']
    warm = torch.load(args.project/spec['warm_checkpoint'], weights_only=False, map_location='cpu')
    replay = torch.load(args.project/spec['replay_checkpoint'], weights_only=False, map_location='cpu')
    selection = [json.loads(line) for line in (args.project/spec['selection_log']).read_text().splitlines()]
    assert len(selection) == spec['training_steps'] == 1000
    pools = []
    for reference in spec['teacher_files']:
        path = args.project/reference['path']
        assert sha(path) == reference['sha256']
        entry = torch.load(path, weights_only=False, map_location='cpu')
        if len(entry['raw_positions']):
            entry['unshifted'] = entry['record']['source'][entry['record']['eligible']]
            pools.append(entry)
    prior = prior_from_checkpoint(warm)
    torch.manual_seed(spec['training_seed'])
    model = restore_model(cfg, warm).train().requires_grad_(True)
    extra = list(model.vector_field.self_conditioning_residual_layer.parameters())+list(model.vector_field.to_edge_logits.parameters())
    extra_ids = {id(p) for p in extra}
    optimizer = torch.optim.AdamW([
        dict(params=[p for p in model.parameters() if id(p) not in extra_ids], lr=spec['fm_lr']),
        dict(params=extra, lr=spec['feedback_lr'])], weight_decay=1e-12)
    start = time.perf_counter()
    valid_before_shift, generated_targets = 0, 0
    for step, selected in enumerate(selection, 1):
        assert selected['step'] == step
        label = selected['selection']
        if label['kind'] == 'reference':
            row = data[label['row']]
            c, target = row['condition'], row['positions']
        else:
            row = pools[label['pool']]
            c = row['condition']
            j, particle = label['row'], label['particle']
            assert row['uniform_weights'][j, particle] > 0
            target = row['unshifted'][j, particle]
            valid_before_shift += assess(target[None], c, [0])['graph_supported']
            generated_targets += 1
        assert selected['composition'] == c['composition_hex']
        graph, node_batch, upper = make_graph(c, cfg)
        graph.ndata['x_1_true'] = target.cuda().float()
        graph.ndata['has_reference_geometry'] = torch.ones(c['n_atoms'], 1, dtype=torch.bool, device='cuda')
        x0, _ = sample_source(prior, c['numbers'], c['charge'], c['spin_multiplicity'], spec['training_seed']*3000017+step)
        objective = clamped_fm_loss(model, graph, node_batch, upper, terminal_time=1., parameterization='displacement',
            prior_positions=x0, pairing='typed_rotation', pairing_radii=covalent_radii(c['numbers'], device='cuda', dtype=torch.float32),
            generator=torch.Generator(device='cuda').manual_seed(spec['training_seed']*1000003+step),
            pairing_generator=torch.Generator(device='cuda').manual_seed(spec['training_seed']*2000003+step))
        if not torch.isfinite(objective):
            raise FloatingPointError('Nonfinite displacement-ablation loss')
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        with (args.out/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(dict(step=step, selection=label, composition=c['composition_hex'],
                loss=float(objective.detach()), gradient_norm=float(norm)))+'\n')
        if step % 100 == 0:
            print(json.dumps(dict(step=step, loss=float(objective.detach()))), flush=True)
    names = {n for n,_ in model.named_parameters()}
    trained = {n:v.detach().cpu() for n,v in model.state_dict().items()}
    torch.save(dict(state_dict=trained, research_protocol=warm['research_protocol'], source_prior=warm['source_prior'],
        ablation_protocol_sha256=ph, optimizer_state_dict=optimizer.state_dict()), args.out/'student.ckpt')
    merged = {}
    for name, value in warm['state_dict'].items():
        if name in names:
            merged[name] = value+(trained[name]-replay['state_dict'][name])
        else:
            assert torch.equal(value, trained[name]) and torch.equal(value, replay['state_dict'][name])
            merged[name] = value.clone()
    model.load_state_dict(merged, strict=True)
    checkpoint = args.out/'last.ckpt'
    recipe = dict(warm['research_protocol'], force_shift_ablation_protocol_sha256=ph)
    torch.save(dict(state_dict=merged, research_protocol=recipe, source_prior=warm['source_prior']), checkpoint)
    write(args.out/'training.json', dict(complete=True, steps=1000, seconds=time.perf_counter()-start,
        generated_targets=generated_targets, graph_valid_before_shift=valid_before_shift,
        selection_log_sha256=spec['selection_log_sha256'], checkpoint_sha256=sha(checkpoint), new_teacher_queries=0))
    out = args.out/'evaluation'
    out.mkdir()
    evaluate(model, prior, 'no_force_shift', cfg, spec, ph, out, sha(checkpoint), args.project/spec['condition_manifest'])
    del model, optimizer, warm, replay, trained, merged
    gc.collect()
    torch.cuda.empty_cache()
    worker = args.project/spec['oracle_worker']
    assert sha(worker) == spec['oracle_worker_sha256'] and sha(spec['oracle_checkpoint']) == spec['oracle_sha256']
    quality = args.out/'physical_eval'
    quality.mkdir()
    records, queries = [], 0
    for index in spec['conditions']:
        file = out/f'no_force_shift_c{index}.pt'
        saved = torch.load(file, weights_only=False, map_location='cpu')
        c, x = saved['condition'], saved['positions']
        n = len(x)
        with EnergyOracle(spec['oracle_interpreter'], worker, spec['oracle_checkpoint'], numbers=c['numbers'],
            charge=c['charge'], spin_multiplicity=c['spin_multiplicity'], device='cuda', batch_size=16) as oracle:
            assert oracle.handshake['base_precision_dtype'] == 'torch.float32' and not oracle.handshake['tf32']
            energy, force = oracle.evaluate_chunked(torch.cat([x,-x]), max_request=32)
            queries += oracle.evaluated
            assert oracle.evaluated == oracle.requested_evaluations
        target = quality/f'no_force_shift_c{index}.pt'
        torch.save(dict(positions=x, raw_energy_eV=energy, raw_force_eV_A=force,
            even_energy_eV=(energy[:n]+energy[n:])/2, even_force_eV_A=(force[:n]-force[n:])/2,
            source_sample_sha256=sha(file)), target)
        records.append(dict(method='no_force_shift', condition_index=index, artifact=target.name,
            artifact_sha256=sha(target), source_sample_sha256=sha(file)))
    assert queries == 1280
    write(quality/'results.json', dict(complete=True, rows=records, raw_queries=queries, protocol_sha256=ph))
    write(args.out/'complete.json', dict(complete=True, protocol_sha256=ph, raw_queries=queries, new_teacher_queries=0))


if __name__ == '__main__':
    main()
