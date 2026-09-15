#!/usr/bin/env python3
"""Train one prespecified, unpretrained generator with common data and capacity."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import datetime

import numpy as np
import torch

from cfm_mol.matched_egnn import HarmonicSource, initialize, loss, sample, state_hash
from scripts.research.audit_generator_output_support import assess
from scripts.research.tree_prior_fm import geometry_counts
from scripts.research.train_electronic_fm import sha


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def batches(rows, steps, batch, seed):
    """Select size buckets in proportion to their counts: uniform row marginals."""
    sizes = np.asarray([r['condition']['n_atoms'] for r in rows])
    unique, counts = np.unique(sizes, return_counts=True)
    buckets = {n:np.flatnonzero(sizes == n) for n in unique}
    rng = np.random.default_rng(seed)
    selected_sizes = rng.choice(unique, size=steps, p=counts/counts.sum())
    return np.stack([rng.choice(buckets[n], size=batch, replace=True) for n in selected_sizes])


@torch.no_grad()
def evaluate(model, source, spec, kind, rows, seed, count, batch, calls, out, label):
    output = []
    model.eval()
    model_digest = state_hash(model)
    report_file = out/f'{label}_results.json'
    if report_file.exists():
        report = json.loads(report_file.read_text())
        assert report['complete'] and report['model_state_sha256'] == model_digest
        for row in report['rows']:
            assert sha(out/f'{label}_c{row["condition_index"]}.pt') == row['sample_sha256']
        return report
    for index, c in enumerate(rows):
        c = dict(c, numbers=c['atomic_numbers'])
        file = out/f'{label}_c{index}.pt'
        if file.exists():
            saved = torch.load(file, map_location='cpu', weights_only=False)
            assert saved['model_state_sha256'] == model_digest and saved['condition'] == c
            assert saved['calls_per_sample'] == calls and saved['evaluation_seed'] == seed
            x, seconds = saved['positions'], saved['generation_seconds']
            assert len(x) == count
        else:
            positions, initial = [], []
            start = time.perf_counter()
            for begin in range(0, count, batch):
                size = min(batch, count-begin)
                x, x0 = sample(model, c['numbers'], kind, spec, source,
                    seed*1000003+index*100003+begin, size, calls)
                if not torch.isfinite(x).all():
                    raise FloatingPointError('Nonfinite generated coordinates')
                positions.append(x.cpu().double())
                initial.append(x0.cpu().double())
            torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            x = torch.cat(positions)
            temporary = file.with_suffix('.tmp')
            torch.save(dict(positions=x, initial_positions=torch.cat(initial), condition=c,
                calls_per_sample=calls, evaluation_seed=seed, model_state_sha256=model_digest,
                generation_seconds=seconds), temporary)
            temporary.replace(file)
        result = dict(condition_index=index, condition=c, sample_sha256=sha(file), generation_seconds=seconds,
            final_geometry=geometry_counts(x, c['numbers']), **assess(x, c, list(range(len(x)))))
        output.append(result)
        print(json.dumps(dict(label=label, index=index, graph=result['graph_supported'], n=c['n_atoms'])), flush=True)
    report = dict(complete=True, kind=kind, calls_per_sample=calls, rows=output, model_state_sha256=model_digest,
        attempted=sum(r['attempted'] for r in output), graph_supported=sum(r['graph_supported'] for r in output),
        new_physical_queries=0)
    write(report_file, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['project', 'protocol', 'out']:
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    spec = json.loads(args.protocol.read_text())
    assert spec['frozen']
    if args.out.exists() and not args.resume:
        raise FileExistsError(args.out)
    if (args.out/'complete.json').exists():
        assert json.loads((args.out/'complete.json').read_text())['protocol_sha256'] == sha(args.protocol)
        return
    torch.set_num_threads(2)
    for key in ['data', 'condition_manifest']:
        assert sha(args.project/spec[key]) == spec[key+'_sha256']
    data = torch.load(args.project/spec['data'], weights_only=False, map_location='cpu')
    panel = json.loads((args.project/spec['condition_manifest']).read_text())['rows']
    all_train = {r['condition']['composition_hex'] for r in data['training']}
    all_val = {r['condition']['composition_hex'] for r in data['validation']}
    assert not all_train & all_val
    assert not (all_train | all_val) & {r['composition_hex'] for r in panel}
    assert len(data['training']) == spec['training_rows']
    for rows in data.values():
        assert all(r['condition']['charge'] == 0 and r['condition']['spin_multiplicity'] == 1 for r in rows)
    args.out.mkdir(parents=True, exist_ok=args.resume)
    model = initialize(spec, 'cuda').train()
    assert model.norm_values[0] == 1., 'Harmonic scales are in Angstrom'
    initial_hash = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec['learning_rate'], amsgrad=True, weight_decay=1e-12)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    source = HarmonicSource(spec['edge_log_width'])
    schedule = batches(data['training'], spec['training_steps'], spec['batch_size'], spec['batch_seed'])
    if (args.out/'batch_indices.npy').exists():
        np.testing.assert_array_equal(np.load(args.out/'batch_indices.npy'), schedule)
    else:
        np.save(args.out/'batch_indices.npy', schedule)
    initialization = dict(state_sha256=initial_hash,
        parameter_count=sum(p.numel() for p in model.parameters()), pretrained=False,
        protocol_sha256=sha(args.protocol), batch_schedule_sha256=sha(args.out/'batch_indices.npy'))
    if (args.out/'initialization.json').exists():
        assert json.loads((args.out/'initialization.json').read_text()) == initialization
    else:
        write(args.out/'initialization.json', initialization)
    start_step, training_seconds = 0, 0.
    if (args.out/'last.ckpt').exists():
        saved = torch.load(args.out/'last.ckpt', map_location='cuda', weights_only=False)
        assert saved['protocol_sha256'] == sha(args.protocol) and saved['initial_state_sha256'] == initial_hash
        model.load_state_dict(saved['state_dict'], strict=True)
        ema.load_state_dict(saved['ema_state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        start_step, training_seconds = saved['global_step'], saved['training_seconds']
        del saved
    attempts = args.out/'attempts'
    attempts.mkdir(exist_ok=True)
    attempt = len(list(attempts.glob('*.json')))
    write(attempts/f'{attempt:03d}.json', dict(start_step=start_step,
        at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        discarded_prefix_compute_retained_in_previous_metrics=True))
    metrics_file = args.out/('metrics.jsonl' if attempt == 0 else f'metrics_attempt{attempt:03d}.jsonl')
    validation_conditions, seen = [], set()
    for row in data['validation']:
        c = row['condition']
        if c['composition_hex'] not in seen:
            validation_conditions.append(c)
            seen.add(c['composition_hex'])
        if len(validation_conditions) == spec['validation_compositions']:
            break
    assert len(validation_conditions) == spec['validation_compositions']
    validation = args.out/'validation'
    validation.mkdir(exist_ok=True)
    start = time.perf_counter()
    if start_step in spec['validation_steps'] and not (validation/f'step{start_step}_results.json').exists():
        evaluate(ema, source, spec, spec['kind'], validation_conditions,
            spec['validation_seed'], spec['validation_samples'], spec['evaluation_batch'],
            spec['inference_calls'][0], validation, f'step{start_step}')
    for step, indices in enumerate(schedule[start_step:], start_step+1):
        tick = time.perf_counter()
        rows = [data['training'][int(i)] for i in indices]
        clean = torch.stack([r['positions'] for r in rows]).float().cuda()
        numbers = torch.tensor([r['condition']['numbers'] for r in rows], device='cuda')
        objective = loss(model, clean, numbers, spec['kind'], spec, source, spec['noise_seed']*1000003+step)
        if not torch.isfinite(objective):
            raise FloatingPointError('Nonfinite training objective')
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        decay = min(spec['ema_decay'], (1+step)/(10+step))
        with torch.no_grad():
            for average, value in zip(ema.parameters(), model.parameters()):
                average.lerp_(value, 1-decay)
        torch.cuda.synchronize()
        training_seconds += time.perf_counter()-tick
        if step == 1 or step % 250 == 0:
            record = dict(step=step, loss=float(objective.detach()), gradient_norm=float(norm),
                elapsed_seconds=time.perf_counter()-start, training_seconds=training_seconds,
                examples_seen=step*spec['batch_size'])
            with metrics_file.open('a') as f:
                f.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True)
        if step % spec['checkpoint_every'] == 0 or step == spec['training_steps']:
            temporary = args.out/'last.tmp'
            torch.save(dict(state_dict=model.state_dict(), ema_state_dict=ema.state_dict(),
                optimizer_state_dict=optimizer.state_dict(), global_step=step, protocol=spec,
                protocol_sha256=sha(args.protocol), initial_state_sha256=initial_hash,
                training_seconds=training_seconds), temporary)
            temporary.replace(args.out/'last.ckpt')
        if step in spec['validation_steps']:
            evaluate(ema, source, spec, spec['kind'], validation_conditions,
                spec['validation_seed'], spec['validation_samples'], spec['evaluation_batch'],
                spec['inference_calls'][0], validation, f'step{step}')
            model.train()
    write(args.out/'training.json', dict(complete=True, steps=spec['training_steps'],
        training_seconds=training_seconds, elapsed_seconds=time.perf_counter()-start,
        examples_seen=spec['training_steps']*spec['batch_size'], initial_state_sha256=initial_hash,
        checkpoint_sha256=sha(args.out/'last.ckpt'), parameter_count=sum(p.numel() for p in model.parameters()),
        primitive_training_calls=spec['training_steps']*spec['batch_size'],
        validation_used_for_model_selection=False, final_model='EMA at fixed final update'))
    final = args.out/'evaluation'
    final.mkdir(exist_ok=True)
    for calls in spec['inference_calls']:
        evaluate(ema, source, spec, spec['kind'], panel, spec['evaluation_seed'],
            spec['samples_per_condition'], spec['evaluation_batch'], calls, final, f'{spec["kind"]}_{calls}')
    write(args.out/'complete.json', dict(complete=True, protocol_sha256=sha(args.protocol),
        new_evaluation_outputs=len(panel)*spec['samples_per_condition']*len(spec['inference_calls']),
        validation_outputs=len(validation_conditions)*spec['validation_samples']*len(spec['validation_steps']),
        new_physical_queries=0))


if __name__ == '__main__':
    main()
