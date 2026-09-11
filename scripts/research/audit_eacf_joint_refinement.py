#!/usr/bin/env python3
"""Replay all saved EACF parents and independently recompute the joint KL statistic.

Run in the isolated JAX environment. This makes no physical oracle queries and
does not estimate the unknown physical marginal density or target coverage.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from eacf.flow.aug_flow_dist import FullGraphSample
from cfm_mol.eacf_reference import build_reference_flow

jax.config.update('jax_enable_x64', True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report_path = args.run / 'results.json'
    report = json.loads(report_path.read_text())
    if not report['complete'] or report['engineering_only']:
        raise ValueError('Require completed production result')
    for name, digest in report['artifacts'].items():
        if sha(args.run / name) != digest:
            raise ValueError(f'Changed artifact: {name}')
    submission = json.loads((args.run / 'submission.json').read_text())
    snapshot = Path(submission['snapshot'])
    for name, digest in report['source_sha256'].items():
        if sha(snapshot / name) != digest:
            raise ValueError(f'Changed source snapshot: {name}')
    manifest_path = args.source / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if sha(manifest_path) != report['source_export_sha256']:
        raise ValueError('Source manifest mismatch')
    if sha(args.source / 'arrays.npz') != manifest['arrays_sha256']:
        raise ValueError('Source array mismatch')
    source = np.load(args.source / 'arrays.npz')
    samples = np.load(args.run / 'samples.npz')
    count = report['evaluation_parents']
    if count != 512 or report['oracle_evaluations'] != 16 * report['steps'] + 4 * count:
        raise ValueError('Production denominator or query count mismatch')
    for key in samples.files:
        if samples[key].dtype.kind in 'fc' and not np.isfinite(samples[key]).all():
            raise ValueError(f'Nonfinite array: {key}')
    np.testing.assert_array_equal(samples['parent_ids'], source['development_parent_ids'])
    np.testing.assert_array_equal(samples['inversion_signs'], source['evaluation_signs'])
    np.testing.assert_array_equal(samples['base_positions'],
        source['development_positions'] * source['evaluation_signs'][:, None, None])
    with (args.run / 'adapter.pkl').open('rb') as handle:
        saved = pickle.load(handle)
    if saved['recipe'] != report['recipe'] or saved['condition'] != report['condition']:
        raise ValueError('Checkpoint recipe or condition mismatch')
    recipe = saved['recipe']
    if not recipe['aux_conditioned_on_x'] or recipe['n_aug'] != 1:
        raise ValueError('Analytic auxiliary check requires the declared conditional Gaussian')
    aux_errors = {}
    for prefix, xkey, akey in [('base', 'base_positions', 'base_auxiliary'),
                             ('adapted', 'positions', 'auxiliary')]:
        delta = (samples[akey] - samples[xkey]) / recipe['aux_scale']
        value = -.5 * (delta ** 2).sum((1, 2)) - delta.shape[1] * 3 * np.log(
            recipe['aux_scale'] * np.sqrt(2 * np.pi))
        stored = samples[prefix + '_aux_log_prob']
        np.testing.assert_allclose(value, stored, atol=1e-9, rtol=1e-10)
        aux_errors[prefix] = float(np.max(np.abs(value - stored)))
    change = (samples['base_log_target'] - samples['adapted_log_target']
              + samples['base_aux_log_prob'] - samples['adapted_aux_log_prob']
              - samples['log_volume'])
    np.testing.assert_allclose(change, samples['paired_joint_kl_change'], atol=1e-9, rtol=0)
    mean, sem = float(change.mean()), float(change.std(ddof=1) / np.sqrt(count))
    np.testing.assert_allclose([mean, sem], [report['paired_joint_kl_change']['mean'],
        report['paired_joint_kl_change']['row_sem']], atol=1e-9, rtol=0)
    flow = build_reference_flow(recipe)
    forward = jax.jit(flow.bijector_forward_and_log_det_with_extra_apply)
    inverse = jax.jit(flow.bijector_inverse_and_log_det_with_extra_apply)
    errors = dict(forward_positions=0., forward_log_volume=0., inverse_positions=0., inverse_log_volume=0.)
    for begin in range(0, count, 32):
        end = min(begin + 32, count)
        x = samples['base_positions'][begin:end]
        a = samples['base_auxiliary'][begin:end]
        features = np.broadcast_to(np.asarray(report['condition']['numbers'])[None, :, None, None],
                                   (len(x), len(x[0]), 1, 1))
        original = FullGraphSample(jnp.asarray(np.stack([x, a], axis=-2)), jnp.asarray(features))
        mapped, volume, _ = forward(saved['theta'], original)
        reconstructed, inverse_volume, _ = inverse(saved['theta'], mapped)
        expected = np.stack([samples['positions'][begin:end], samples['auxiliary'][begin:end]], axis=-2)
        pairs = [('forward_positions', mapped.positions, expected),
                 ('forward_log_volume', volume, samples['log_volume'][begin:end]),
                 ('inverse_positions', reconstructed.positions, original.positions),
                 ('inverse_log_volume', volume + inverse_volume, np.zeros(len(x)))]
        for name, actual, expected in pairs:
            np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-8)
            errors[name] = max(errors[name], float(np.max(np.abs(np.asarray(actual) - expected))))
    result = dict(complete=True, run=str(args.run), source_commit=submission['source_commit'],
        results_sha256=sha(report_path), artifacts=report['artifacts'], parents_replayed=count,
        auxiliary_gaussian_errors=aux_errors, replay_errors=errors,
        paired_joint_kl_change=dict(mean=mean, row_sem=sem),
        original_oracle_evaluations=report['oracle_evaluations'], additional_oracle_evaluations=0,
        scientific_submission_ready=False,
        limitations=['Joint KL change upper-bounds physical marginal change in expectation.',
                     'No independent physical energy re-query or mode-coverage certification.',
                     'Row SEM is not training-seed uncertainty.'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
