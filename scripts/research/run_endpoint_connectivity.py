#!/usr/bin/env python3
"""Bounded raw-generation connectivity training with replay and local controls."""
import argparse
import gc
import json
import time
from pathlib import Path
from types import MethodType

import torch
from flowmol.model_utils.load import read_config_file
from cfm_mol.clamped_fm import clamped_fm_loss
from cfm_mol.clamped_density import center_by_graph
from cfm_mol.chemical_moves import covalent_radii
from cfm_mol.endpoint_connectivity import endpoint_support_loss
from cfm_mol.source_checkpoint import prior_from_checkpoint
from scripts.research.tree_prior_fm import restore_model, sample_source, evaluate
from scripts.research.run_tree_manifold import make_graph
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['project', 'protocol', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    spec = json.loads(a.protocol.read_text())
    ph = sha(a.protocol)
    assert spec['frozen'] and not a.out.exists()
    for key in ['config', 'data', 'warm_checkpoint', 'condition_manifest']:
        assert sha(a.project / spec[key]) == spec[key + '_sha256']
    cfg = read_config_file(a.project / spec['config'])
    cfg['mol_fm'].pop('bgfm', None)
    assert cfg['dataset']['max_atoms'] == 200 and cfg['mol_fm']['total_loss_weights']['e'] == 0
    warm = torch.load(a.project / spec['warm_checkpoint'], map_location='cpu', weights_only=False)
    data = torch.load(a.project / spec['data'], map_location='cpu', weights_only=False)
    panel = json.loads((a.project / spec['condition_manifest']).read_text())
    excluded = {r['composition_hex'] for r in panel['rows']}
    training = data['training'][:spec['training_steps']]
    assert len(training) == spec['training_steps']
    assert all(r['condition']['composition_hex'] not in excluded for r in training)
    a.out.mkdir(parents=True)
    evaluation = a.out / 'evaluation'
    evaluation.mkdir()
    prior = prior_from_checkpoint(warm)
    manifest = a.project / spec['condition_manifest']
    states = {'frozen': warm['state_dict']}
    parameter_names = None
    for method in spec['training_methods']:
        torch.manual_seed(spec['training_seed'])
        model = restore_model(cfg, warm).train()
        model.requires_grad_(True)
        parameter_names = {name for name, _ in model.named_parameters()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=spec['learning_rate'], weight_decay=1e-12)
        directory = a.out / method
        directory.mkdir()
        start = time.perf_counter()
        for step, row in enumerate(training, 1):
            c = row['condition']
            g, nbi, uem = make_graph(c, cfg)
            g.ndata['x_1_true'] = row['positions'].cuda().float()
            g.ndata['has_reference_geometry'] = torch.ones(c['n_atoms'], 1, dtype=torch.bool, device='cuda')
            radii = covalent_radii(c['numbers'], device='cuda', dtype=torch.float32)
            draw_seed = spec['training_seed'] * 3000017 + step
            x0, _ = sample_source(prior, c['numbers'], c['charge'], c['spin_multiplicity'], draw_seed)
            field = model.vector_field
            original = field.forward
            captured = {}

            def capture(self, graph, t, *args, **kwargs):
                output = original(graph, t, *args, **kwargs)
                x = center_by_graph(graph.ndata['x_t'], nbi, graph.batch_size)
                predicted = center_by_graph(output['x'], nbi, graph.batch_size)
                captured['endpoint'] = x + (1 - t[nbi, None]) * (predicted - x)
                captured['time'] = t
                return output

            field.forward = MethodType(capture, field)
            try:
                fm = clamped_fm_loss(model, g, nbi, uem, terminal_time=1., parameterization='displacement', prior_positions=x0,
                    pairing='typed_rotation', pairing_radii=radii,
                    generator=torch.Generator(device='cuda').manual_seed(spec['training_seed'] * 1000003 + step),
                    pairing_generator=torch.Generator(device='cuda').manual_seed(spec['training_seed'] * 2000003 + step))
            finally:
                field.forward = original
            t = captured['time'][0]
            auxiliary = fm.new_zeros(())
            if method != 'replay' and t >= spec['minimum_time']:
                auxiliary = t.square() * endpoint_support_loss(captured['endpoint'], radii,
                    mode=method, contact=spec['contact_margin'], exclusion=spec['exclusion_margin'])
            loss = fm + spec['support_weight'] * auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite connectivity objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            record = dict(step=step, processed_index=c['processed_index'], source_seed=draw_seed,
                time=float(t), fm_loss=float(fm.detach()), support_loss=float(auxiliary.detach()),
                total_loss=float(loss.detach()), gradient_norm=float(norm), seconds=time.perf_counter()-start)
            with (directory / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            if step % 100 == 0:
                print(json.dumps(dict(method=method, **record)), flush=True)
        states[method] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        recipe = dict(warm['research_protocol'], endpoint_connectivity_protocol_sha256=ph,
                      endpoint_connectivity_method=method)
        torch.save(dict(state_dict=states[method], research_protocol=recipe, source_prior=warm['source_prior'],
                        optimizer_state_dict=optimizer.state_dict(), global_step=spec['training_steps']), directory / 'last.ckpt')
        write(directory / 'training.json', dict(complete=True, steps=spec['training_steps'], seconds=time.perf_counter()-start,
              checkpoint_sha256=sha(directory/'last.ckpt'), primitive_training_forwards=2*spec['training_steps']))
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    for method in ['local', 'tree']:
        delta = {}
        for name, value in states['frozen'].items():
            if name in parameter_names:
                delta[name] = value + (states[method][name] - states['replay'][name])
            else:
                assert torch.equal(value, states[method][name]) and torch.equal(value, states['replay'][name])
                delta[name] = value.clone()
        states[method + '_delta'] = delta
        directory = a.out / (method + '_delta')
        directory.mkdir()
        torch.save(dict(state_dict=delta, research_protocol=dict(warm['research_protocol'],
            endpoint_connectivity_protocol_sha256=ph, endpoint_connectivity_method=method+'_delta', coefficient=1.),
            source_prior=warm['source_prior']), directory / 'last.ckpt')
    for method in spec['methods']:
        path = a.project/spec['warm_checkpoint'] if method == 'frozen' else a.out/method/'last.ckpt'
        saved = warm if method == 'frozen' else torch.load(path, map_location='cpu', weights_only=False)
        model = restore_model(cfg, saved)
        evaluate(model, prior, method, cfg, spec, ph, evaluation, sha(path), manifest)
        del model, saved
        gc.collect()
        torch.cuda.empty_cache()
    write(a.out/'complete.json', dict(complete=True, protocol_sha256=ph, methods=spec['methods'],
        new_training_steps=3*spec['training_steps'], generated_outputs=6*len(spec['conditions'])*spec['samples_per_condition'],
        new_physical_queries=0, output_coordinate_optimization=False, scientific_submission_ready=False))


if __name__ == '__main__':
    main()
