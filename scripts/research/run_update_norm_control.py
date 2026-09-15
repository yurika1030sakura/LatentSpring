#!/usr/bin/env python3
"""Test update direction against a direct physical update of the same norm."""
import argparse
import gc
import json
from pathlib import Path

import torch
from flowmol.model_utils.load import read_config_file

from cfm_mol.source_checkpoint import prior_from_checkpoint
from cfm_mol.energy_oracle import EnergyOracle
from scripts.research.tree_prior_fm import restore_model, evaluate
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
    states = {}
    for kind, reference in spec['checkpoints'].items():
        file = args.project/reference['path']
        assert sha(file) == reference['sha256']
        states[kind] = torch.load(file, map_location='cpu', weights_only=False)
    config = args.project/spec['config']
    assert sha(config) == spec['config_sha256']
    cfg = read_config_file(config)
    cfg['mol_fm'].pop('bgfm', None)
    assert cfg['dataset']['max_atoms'] == 200 and cfg['mol_fm']['total_loss_weights']['e'] == 0
    model = restore_model(cfg, states['frozen'])
    names = {name for name, _ in model.named_parameters()}
    base = states['frozen']['state_dict']
    physical = states['physical']['state_dict']
    paired = states['paired']['state_dict']
    norm2 = lambda a:sum(float((a[n]-base[n]).double().square().sum()) for n in sorted(names))
    direct_norm, paired_norm = norm2(physical)**.5, norm2(paired)**.5
    assert direct_norm > 0 and paired_norm > 0
    coefficient = paired_norm/direct_norm
    merged = {}
    for name, value in base.items():
        if name in names:
            merged[name] = value+coefficient*(physical[name]-value)
        else:
            assert torch.equal(value, physical[name]) and torch.equal(value, paired[name])
            merged[name] = value.clone()
    achieved = norm2(merged)**.5
    assert abs(achieved/paired_norm-1) < 1e-5
    model.load_state_dict(merged, strict=True)
    args.out.mkdir(parents=True)
    recipe = dict(states['frozen']['research_protocol'], norm_control_protocol_sha256=ph,
                  norm_control_coefficient=coefficient)
    checkpoint = args.out/'last.ckpt'
    torch.save(dict(state_dict=merged, research_protocol=recipe, source_prior=states['frozen']['source_prior']), checkpoint)
    write(args.out/'update.json', dict(complete=True, protocol_sha256=ph, coefficient=coefficient,
        direct_update_norm=direct_norm, paired_update_norm=paired_norm, achieved_update_norm=achieved,
        coefficient_uses_training_weights_only=True, checkpoint_sha256=sha(checkpoint), new_optimizer_steps=0))
    prior = prior_from_checkpoint(states['frozen'])
    manifest = args.project/spec['condition_manifest']
    assert sha(manifest) == spec['condition_manifest_sha256']
    out = args.out/'evaluation'
    out.mkdir()
    evaluate(model, prior, 'norm_direct', cfg, spec, ph, out, sha(checkpoint), manifest)
    del model, states, merged
    gc.collect()
    torch.cuda.empty_cache()
    worker = args.project/spec['oracle_worker']
    assert sha(worker) == spec['oracle_worker_sha256'] and sha(spec['oracle_checkpoint']) == spec['oracle_sha256']
    quality = args.out/'physical_eval'
    quality.mkdir()
    rows, queries = [], 0
    for index in spec['conditions']:
        file = out/f'norm_direct_c{index}.pt'
        saved = torch.load(file, weights_only=False, map_location='cpu')
        c, x = saved['condition'], saved['positions']
        n = len(x)
        with EnergyOracle(spec['oracle_interpreter'], worker, spec['oracle_checkpoint'], numbers=c['numbers'],
            charge=c['charge'], spin_multiplicity=c['spin_multiplicity'], device='cuda', batch_size=16) as oracle:
            assert oracle.handshake['base_precision_dtype'] == 'torch.float32' and not oracle.handshake['tf32']
            energy, force = oracle.evaluate_chunked(torch.cat([x, -x]), max_request=32)
            queries += oracle.evaluated
            assert oracle.evaluated == oracle.requested_evaluations
        target = quality/f'norm_direct_c{index}.pt'
        torch.save(dict(positions=x, raw_energy_eV=energy, raw_force_eV_A=force,
            even_energy_eV=(energy[:n]+energy[n:])/2, even_force_eV_A=(force[:n]-force[n:])/2,
            source_sample_sha256=sha(file)), target)
        rows.append(dict(method='norm_direct', condition_index=index, artifact=target.name,
                         artifact_sha256=sha(target), source_sample_sha256=sha(file)))
        print(json.dumps(dict(condition=index, queries=queries)), flush=True)
    assert queries == 2*len(spec['conditions'])*spec['samples_per_condition']
    write(quality/'results.json', dict(complete=True, rows=rows, protocol_sha256=ph, raw_queries=queries))
    write(args.out/'complete.json', dict(complete=True, protocol_sha256=ph, raw_queries=queries,
        new_training_steps=0, coefficient=coefficient))


if __name__ == '__main__':
    main()
