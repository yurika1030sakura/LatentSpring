#!/usr/bin/env python3
"""Task-adapted official EDM: diffuse positions, condition on clean atom types.

This is a newly trained conditional baseline, not clamping the native joint
sampler and claiming its exact conditional law or published benchmark performance.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
from scripts.research.train_electronic_fm import sha
from scripts.research.evaluate_chemical_policy import write
from scripts.research.audit_generator_output_support import assess
from scripts.research.tree_prior_fm import geometry_counts


def center(x):
    return x - x.mean(1, keepdim=True)


def build(spec, device):
    upstream = Path(spec['upstream'])
    for path, digest in spec['upstream_sha256'].items():
        assert sha(upstream / path) == digest
    sys.path.insert(0, str(upstream))
    from qm9.models import get_model
    # Avoid an ambient namespace collision with this repository's configs/.
    from configs.datasets_config import get_dataset_info
    args = SimpleNamespace(**spec['upstream_args'])
    assert not args.conditioning and not args.include_charges
    info = get_dataset_info(args.dataset, args.remove_h)
    assert info['atomic_nb'] == spec['atomic_numbers']
    model, _, _ = get_model(args, device, info, None)
    assert sha(spec['warm_checkpoint']) == spec['warm_checkpoint_sha256']
    saved = torch.load(spec['warm_checkpoint'], map_location=device, weights_only=False)
    model.load_state_dict(saved, strict=True)
    model.to(device)

    def reject_nonfinite(module, inputs, output):
        if any(not torch.isfinite(value).all() for value in output):
            raise FloatingPointError('Nonfinite upstream EGNN output before its legacy NaN fallback')

    model.dynamics.egnn.register_forward_hook(reject_nonfinite)
    return model


def features(numbers, batch, spec, device):
    indices = torch.tensor([spec['atomic_numbers'].index(z) for z in numbers], device=device)
    h = torch.nn.functional.one_hot(indices, len(spec['atomic_numbers'])).float()
    h = h[None].expand(batch, -1, -1) / spec['upstream_args']['normalize_factors'][1]
    n = len(numbers)
    node_mask = torch.ones(batch, n, 1, device=device)
    edge_mask = (~torch.eye(n, dtype=torch.bool, device=device))[None].expand(batch, -1, -1).reshape(-1, 1).float()
    return h, node_mask, edge_mask


def noise_prediction(model, x, t, h, node_mask, edge_mask):
    out = model.phi(torch.cat([x, h], -1), t, node_mask, edge_mask, None)[..., :3]
    if not torch.isfinite(out).all():
        raise FloatingPointError('Nonfinite conditional noise prediction')
    return center(out)


@torch.no_grad()
def sample(model, numbers, spec, seed, calls, batch):
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    shape = (batch, len(numbers), 3)
    h, node_mask, edge_mask = features(numbers, batch, spec, device)
    x = center(torch.randn(shape, device=device, generator=rng))
    initial = x.clone()
    # Include the final p(x|z_0) evaluation in the requested primitive call count.
    ticks = np.rint(np.linspace(0, model.T, calls)).astype(int)
    assert len(np.unique(ticks)) == calls and ticks[0] == 0 and ticks[-1] == model.T
    for low, high in reversed(list(zip(ticks[:-1], ticks[1:]))):
        s = x.new_full((batch, 1), int(low) / model.T)
        t = x.new_full((batch, 1), int(high) / model.T)
        gs, gt = model.gamma(s), model.gamma(t)
        variance, std, ratio = model.sigma_and_alpha_t_given_s(gt, gs, x)
        ss, st = model.sigma(gs, x), model.sigma(gt, x)
        epsilon = noise_prediction(model, x, t, h, node_mask, edge_mask)
        mean = x / ratio - variance / (ratio * st) * epsilon
        x = center(mean + std * ss / st * center(torch.randn(shape, device=device, generator=rng)))
        if not torch.isfinite(x).all():
            raise FloatingPointError('Nonfinite conditional diffusion state')
    t = x.new_zeros((batch, 1))
    gamma = model.gamma(t)
    alpha, sigma = model.alpha(gamma, x), model.sigma(gamma, x)
    epsilon = noise_prediction(model, x, t, h, node_mask, edge_mask)
    result = center((x - sigma * epsilon) / alpha + sigma / alpha * center(torch.randn(shape, device=device, generator=rng)))
    return result * model.norm_values[0], initial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['project', 'protocol', 'out']:
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.protocol.read_text())
    ph = sha(args.protocol)
    assert spec['frozen'] and not args.out.exists()
    torch.set_num_threads(2)
    torch.manual_seed(spec['training_seed'])
    for key in ['data', 'condition_manifest']:
        assert sha(args.project/spec[key]) == spec[key+'_sha256']
    data = torch.load(args.project/spec['data'], map_location='cpu', weights_only=False)
    panel = json.loads((args.project/spec['condition_manifest']).read_text())['rows']
    excluded = {r['composition_hex'] for r in panel}
    training = data['training']
    assert len(training) * spec.get('training_passes', 1) == spec['training_steps']
    assert all(r['condition']['composition_hex'] not in excluded for r in training)
    assert all(r['condition']['charge'] == 0 and r['condition']['spin_multiplicity'] == 1 for r in training)
    model = build(spec, 'cuda').train()
    args.out.mkdir(parents=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec['learning_rate'], amsgrad=True, weight_decay=1e-12)
    start_step = 0
    prior_seconds = 0.
    if spec.get('resume'):
        reference = spec['resume']
        for key in ['checkpoint','metrics','training_report']:
            assert sha(args.project/reference[key]) == reference[key+'_sha256']
        prior = torch.load(args.project/reference['checkpoint'], map_location='cuda', weights_only=False)
        model.load_state_dict(prior['state_dict'], strict=True)
        optimizer.load_state_dict(prior['optimizer_state_dict'])
        old_metrics = (args.project/reference['metrics']).read_text()
        old_rows = [json.loads(line) for line in old_metrics.splitlines()]
        start_step = reference['completed_steps']
        assert len(old_rows) == start_step and old_rows[-1]['step'] == start_step
        (args.out/'metrics.jsonl').write_text(old_metrics)
        prior_seconds = json.loads((args.project/reference['training_report']).read_text())['seconds']
    start = time.perf_counter()
    for step in range(start_step+1, spec['training_steps']+1):
        row = training[(step-1) % len(training)]
        c = row['condition']
        clean = center(row['positions'][None].cuda().float()) / model.norm_values[0]
        rng = torch.Generator(device='cuda').manual_seed(spec['training_seed'] * 1000003 + step)
        t = torch.randint(0, model.T+1, (1, 1), device='cuda', generator=rng).float()/model.T
        epsilon = center(torch.randn(clean.shape, device='cuda', generator=rng))
        gamma = model.gamma(t)
        x = model.alpha(gamma, clean)*clean + model.sigma(gamma, clean)*epsilon
        h, node_mask, edge_mask = features(c['numbers'], 1, spec, 'cuda')
        prediction = noise_prediction(model, x, t, h, node_mask, edge_mask)
        loss = (prediction-epsilon).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite conditional EDM loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        record = dict(step=step, processed_index=c['processed_index'], time=float(t), loss=float(loss.detach()),
                      gradient_norm=float(norm), seconds=time.perf_counter()-start)
        with (args.out/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        if step % 100 == 0:
            print(json.dumps(record), flush=True)
    checkpoint = args.out/'last.ckpt'
    torch.save(dict(state_dict=model.state_dict(), protocol_sha256=ph, optimizer_state_dict=optimizer.state_dict()), checkpoint)
    write(args.out/'training.json', dict(complete=True, steps=spec['training_steps'], seconds=prior_seconds+time.perf_counter()-start,
        new_steps=spec['training_steps']-start_step, new_seconds=time.perf_counter()-start,
        parameter_count=sum(p.numel() for p in model.parameters()), checkpoint_sha256=sha(checkpoint),
        primitive_denoiser_training_forwards=spec['training_steps']))
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    model.eval().requires_grad_(False)
    out = args.out/'evaluation'
    out.mkdir()
    for calls in spec['inference_calls']:
        method = f'edm_{calls}'
        rows = []
        for i, c in enumerate(panel):
            c = dict(c, numbers=c['atomic_numbers'])
            assert c['charge'] == 0 and c['spin_multiplicity'] == 1
            xs, initials, seeds = [], [], []
            start = time.perf_counter()
            for begin in range(0, spec['samples_per_condition'], spec['evaluation_batch']):
                seed = spec['evaluation_seed']*1000003+i*100003+begin
                x, initial = sample(model, c['atomic_numbers'], spec, seed, calls, spec['evaluation_batch'])
                xs.append(x.cpu().double()); initials.append(initial.cpu().double()); seeds.append(seed)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            x = torch.cat(xs)
            file = out/f'{method}_c{i}.pt'
            torch.save(dict(positions=x, initial_positions=torch.cat(initials), batch_seeds=seeds, condition=c,
                            primitive_denoiser_calls_per_sample=calls), file)
            row = dict(method=method, condition_index=i, condition=c, sample_sha256=sha(file),
                checkpoint_sha256=sha(checkpoint), generation_seconds=seconds,
                final_geometry=geometry_counts(x, c['atomic_numbers']), **assess(x, c, list(range(len(x)))))
            rows.append(row)
            print(json.dumps({k:row[k] for k in ['method','condition_index','graph_supported','geometrically_supported']}), flush=True)
        write(out/f'{method}_results.json', dict(complete=True, protocol_sha256=ph, rows=rows,
            scope=spec['scope'], checkpoint_sha256=sha(checkpoint)))
    write(args.out/'complete.json', dict(complete=True, protocol_sha256=ph, new_training_steps=spec['training_steps']-start_step,
        new_neural_outputs=len(panel)*spec['samples_per_condition']*len(spec['inference_calls']),
        new_physical_queries=0, scientific_submission_ready=False))


if __name__ == '__main__':
    main()
