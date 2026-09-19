#!/usr/bin/env python3
"""Direct minima adaptation of a real LatentSpring checkpoint, with preflight.

Execute from the repository root as ``python -m scripts.research.train_weighted_minima``.
The original backbone/SC/source are restored strictly. A fixed-temperature
endpoint store is required. This script is additive and does not change archived
training/sampling. It never loads an energy model and never writes to GitHub.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from cfm_mol.weighted_endpoints import WeightedEndpointStore, file_sha256


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def preflight(args):
    modules = {k: importlib.util.find_spec(k) is not None
               for k in ('torch', 'dgl', 'flowmol', 'ase', 'rdkit')}
    checks = dict(modules=modules, cuda_available=torch.cuda.is_available(),
        requested_device=args.device, checkpoint_exists=args.checkpoint.is_file(),
        config_exists=args.config.is_file(), endpoints_exist=(args.endpoints/'manifest.json').is_file(),
        checkpoint_sha256=file_sha256(args.checkpoint) if args.checkpoint.is_file() else None,
        config_sha256=file_sha256(args.config) if args.config.is_file() else None)
    ready = (all(modules.values()) and checks['checkpoint_exists'] and checks['config_exists']
             and checks['endpoints_exist'] and (not args.device.startswith('cuda') or checks['cuda_available']))
    if checks['endpoints_exist']:
        store = WeightedEndpointStore.load(args.endpoints, failure_policy=args.failure_policy)
        temperatures = sorted({g.temperature_K for g in store.groups})
        checks.update(endpoint_manifest_sha256=file_sha256(args.endpoints/'manifest.json'),
                      endpoint_audit=store.audit(), teacher_temperatures_K=temperatures)
        ready = ready and len(temperatures) == 1
    checks.update(ready=ready, scope='Environment/inputs only; true does not certify model or teacher correctness')
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--endpoints', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--seed', type=int, default=260918)
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--feedback-learning-rate', type=float, default=3e-4)
    p.add_argument('--sigma-A', type=float, default=0.005)
    p.add_argument('--model-kT-eV', type=float, default=1.)
    p.add_argument('--failure-policy', choices=['error', 'condition_within_group'], default='error')
    p.add_argument('--allow-legacy-pickle', action='store_true',
        help='Only for a checkpoint you trust; otherwise use torch weights_only loading')
    args = p.parse_args()
    if (args.steps < 1 or args.batch_size < 1 or any(not math.isfinite(x) or x <= 0 for x in
            [args.learning_rate, args.feedback_learning_rate, args.model_kT_eV])
            or not math.isfinite(args.sigma_A) or args.sigma_A < 0):
        p.error('Invalid positive training parameters or smoothing width')
    args.out.mkdir(parents=True, exist_ok=False)
    audit = preflight(args); save_json(args.out/'preflight.json', audit)
    if args.preflight:
        print(json.dumps(audit, indent=2)); return
    if not audit['ready']:
        raise RuntimeError('Preflight failed; see preflight.json. No model training was started.')
    from flowmol.model_utils.load import read_config_file, model_from_config
    from cfm_mol.radial_reference import prepare_research_backbone
    from cfm_mol.smooth_geometry import patch_smooth_geometry
    from cfm_mol.source_checkpoint import prior_from_checkpoint
    from cfm_mol.weighted_endpoint_fm import dgl_batch, endpoint_fm_loss
    cfg = read_config_file(args.config); cfg['mol_fm'].pop('bgfm', None)
    if cfg['dataset']['max_atoms'] != 200 or cfg['mol_fm']['total_loss_weights']['e'] != 0:
        raise ValueError('Keep the original max_atoms=200, bond-free training contract')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=not args.allow_legacy_pickle)
    recipe = state['research_protocol']
    if recipe.get('position_parameterization') != 'displacement' or recipe.get('data_endpoint_time') != 1.:
        raise ValueError('This driver requires an existing time-one displacement-FM checkpoint')
    if recipe.get('latent_tree_context'):
        raise ValueError('Explicit tree-conditioned checkpoints are outside this composition-only integration')
    device = torch.device(args.device); torch.set_num_threads(2); torch.manual_seed(args.seed)
    model = model_from_config(cfg); prepare_research_backbone(model, recipe)
    model.load_state_dict(state['state_dict'], strict=True)
    patch_smooth_geometry(model, recipe.get('geometry_softening', 0.))
    model = model.to(device).float().train()
    # Deliberately DO NOT call requires_grad_(True): preserve frozen schedules/modules.
    trainable = [(n, v) for n, v in model.named_parameters() if v.requires_grad]
    if not trainable: raise ValueError('No trainable parameters after checkpoint restoration')
    extra = set()
    for key in ['self_conditioning_residual_layer', 'to_edge_logits']:
        module = getattr(model.vector_field, key, None)
        if module is not None: extra.update(id(v) for v in module.parameters() if v.requires_grad)
    groups = []
    for flag, rate in [(False, args.learning_rate), (True, args.feedback_learning_rate)]:
        values = [v for n, v in trainable if (id(v) in extra) == flag]
        if values: groups.append(dict(params=values, lr=rate))
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-12)
    store = WeightedEndpointStore.load(args.endpoints, failure_policy=args.failure_policy)
    rng = np.random.default_rng(args.seed)
    prior = prior_from_checkpoint(state)
    frozen = {n: v.detach().cpu().clone() for n, v in model.named_parameters() if not v.requires_grad}
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings.update(inputs=audit, trainable_parameter_names=[n for n, _ in trainable],
        trainable_parameters=sum(v.numel() for _, v in trainable),
        teacher_temperature_K=next(iter({g.temperature_K for g in store.groups})),
        mode='minima', replay=False, parameter_subtraction=False, raw_generation_terminal_noise_A=0.,
        inference_energy_queries=0, checkpoint_validation='strict state_dict load')
    save_json(args.out/'protocol.json', settings)
    start = time.perf_counter()
    with (args.out/'training.jsonl').open('w') as log:
        for step in range(1, args.steps+1):
            draws = store.draw(args.batch_size, rng, sigma_A=args.sigma_A)
            graph, nbi, uem = dgl_batch(draws, cfg, device, model_kT_eV=args.model_kT_eV)
            loss = endpoint_fm_loss(model, graph, nbi, uem, draws, prior, args.seed*3000017 + step*args.batch_size)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite FM loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([v for _, v in trainable], 1., error_if_nonfinite=True)
            optimizer.step()
            log.write(json.dumps(dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm),
                endpoint_ids=[[r.group_id, r.particle_id, r.basin_id] for r in draws]))+'\n')
            if step % 100 == 0: log.flush(); print(f'step={step} loss={float(loss.detach()):.6g}', flush=True)
    for n, v in model.named_parameters():
        if n in frozen and not torch.equal(v.detach().cpu(), frozen[n]):
            raise RuntimeError('A frozen parameter changed: '+n)
    updated = {**recipe, 'mass_preserving_minima': dict(protocol_sha256=file_sha256(args.out/'protocol.json'),
        endpoint_manifest_sha256=file_sha256(args.endpoints/'manifest.json'), sigma_A=args.sigma_A,
        teacher_temperature_K=settings['teacher_temperature_K'], model_kT_eV=args.model_kT_eV,
        failure_policy=args.failure_policy, terminal_noise_std_A=0., steps=args.steps)}
    checkpoint = args.out/'last.ckpt'
    torch.save(dict(state_dict=model.state_dict(), research_protocol=updated,
        source_prior=state.get('source_prior'), optimizer_state_dict=optimizer.state_dict(),
        global_step=args.steps, numpy_rng_state=rng.bit_generator.state), checkpoint)
    save_json(args.out/'complete.json', dict(complete=True, steps=args.steps,
        seconds=time.perf_counter()-start, checkpoint_sha256=file_sha256(checkpoint),
        scientific_performance_evaluated=False, published_quality_claim=False))
    print(str(checkpoint))


if __name__ == '__main__': main()
