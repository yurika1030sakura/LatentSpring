#!/usr/bin/env python3
"""Remove self-conditioning, with both equal-update and equal-forward budgets."""
import argparse
import json
from pathlib import Path
import time

import torch
from flowmol.model_utils.load import read_config_file

from cfm_mol.clamped_fm import clamped_fm_loss
from cfm_mol.chemical_moves import covalent_radii
from scripts.research.tree_prior_fm import restore_model, load_prior, sample_source, evaluate
from scripts.research.run_tree_manifold import make_graph
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
    torch.set_num_threads(2)
    args.out.mkdir(parents=True)
    for key in ['data', 'config', 'warm_checkpoint', 'condition_manifest']:
        assert sha(args.project/spec[key]) == spec[key+'_sha256']
    cfg = read_config_file(args.project/spec['config'])
    cfg['mol_fm'].pop('bgfm', None)
    assert cfg['dataset']['max_atoms'] == 200 and cfg['mol_fm']['total_loss_weights']['e'] == 0
    data = torch.load(args.project/spec['data'], map_location='cpu', weights_only=False)['training']
    assert len(data) == 3000 and spec['training_steps'] == 6000
    panel = json.loads((args.project/spec['condition_manifest']).read_text())['rows']
    assert not {r['composition_hex'] for r in panel} & {r['condition']['composition_hex'] for r in data}
    warm = torch.load(args.project/spec['warm_checkpoint'], map_location='cpu', weights_only=False)
    torch.manual_seed(spec['fm_seed'])
    model = restore_model(cfg, warm).train()
    assert not hasattr(model.vector_field, '_geometry_sc_configuration')
    model._research_prior_kind = 'harmonic_tree'
    trainable = {n:p.requires_grad for n,p in model.named_parameters()}
    prior = load_prior('harmonic_tree', None, spec, ph)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec['fm_lr'], weight_decay=1e-12)
    training_seconds = 0.
    for step in range(1, spec['training_steps']+1):
        tick = time.perf_counter()
        row = data[(step-1) % len(data)]
        c = row['condition']
        graph, node_batch, upper = make_graph(c, cfg)
        graph.ndata['x_1_true'] = row['positions'].cuda().float()
        graph.ndata['has_reference_geometry'] = torch.ones(c['n_atoms'], 1, dtype=torch.bool, device='cuda')
        x0, _ = sample_source(prior, c['numbers'], c['charge'], c['spin_multiplicity'], spec['fm_seed']*3000017+step)
        objective = clamped_fm_loss(model, graph, node_batch, upper, terminal_time=1., parameterization='displacement',
            prior_positions=x0, pairing='typed_rotation', pairing_radii=covalent_radii(c['numbers'], device='cuda', dtype=torch.float32),
            generator=torch.Generator(device='cuda').manual_seed(spec['fm_seed']*1000003+step),
            pairing_generator=torch.Generator(device='cuda').manual_seed(spec['fm_seed']*2000003+step))
        if not torch.isfinite(objective):
            raise FloatingPointError('Nonfinite no-SC training loss')
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        training_seconds += time.perf_counter()-tick
        if step % 100 == 0:
            record = dict(step=step, loss=float(objective.detach()), gradient_norm=float(norm), seconds=training_seconds)
            with (args.out/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True)
        if step in spec['evaluation_steps']:
            directory = args.out/f'step{step}'
            directory.mkdir()
            recipe = dict(warm['research_protocol'], source_prior_kind='harmonic_tree',
                position_parameterization='displacement', sc_ablation_protocol_sha256=ph,
                sc_ablation_steps=step, fm_seed=spec['fm_seed'], data_sha256=spec['data_sha256'])
            assert 'geometry_self_conditioning' not in recipe
            checkpoint = directory/'last.ckpt'
            torch.save(dict(state_dict=model.state_dict(), research_protocol=recipe,
                source_prior=dict(configuration=prior.configuration, state_dict=prior.state_dict()),
                global_step=step, optimizer_state_dict=optimizer.state_dict()), checkpoint)
            write(directory/'training.json', dict(complete=True, steps=step,
                primitive_denoiser_training_forwards=step, training_seconds=training_seconds,
                self_conditioning=False, checkpoint_sha256=sha(checkpoint)))
            out = directory/'evaluation'
            out.mkdir()
            evaluate(model, prior, 'no_sc', cfg, spec, ph, out, sha(checkpoint), args.project/spec['condition_manifest'])
            model.train()
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(trainable[name])
    write(args.out/'complete.json', dict(complete=True, protocol_sha256=ph,
        evaluation_steps=spec['evaluation_steps'], new_molecular_oracle_calls=0,
        scope='The3000-update checkpoint matches optimizer updates;6000 matches the two-pass reference training-forward count. Both use128 inference calls.'))


if __name__ == '__main__':
    main()
