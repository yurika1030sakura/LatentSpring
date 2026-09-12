#!/usr/bin/env python3
"""Isolated NumPy/JSON EACF forward/inverse service; no physical oracle calls."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from eacf.flow.aug_flow_dist import FullGraphSample
from cfm_mol.eacf_reference import build_reference_flow

jax.config.update('jax_enable_x64', True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def emit(value):
    print('BGFM_EACF_JSON '+json.dumps(value, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run/'results.json').read_text())
    assert report['complete'] and not report['engineering_only']
    checkpoint = args.run/'adapter.pkl'
    assert sha(checkpoint) == report['artifacts']['adapter.pkl']
    with checkpoint.open('rb') as handle:
        saved = pickle.load(handle)
    assert saved['recipe'] == report['recipe'] and saved['condition'] == report['condition']
    recipe, condition = saved['recipe'], saved['condition']
    assert recipe['n_aug'] == 1 and recipe['aux_conditioned_on_x']
    flow = build_reference_flow(recipe)
    theta = jax.tree_util.tree_map(jnp.asarray, saved['theta'])
    n = recipe['nodes']
    def one(theta, item, numbers):
        x, a, direction = item
        sample = FullGraphSample(positions=jnp.stack([x, a], -2)[None], features=numbers[None, :, None, None])
        def apply_forward(sample):
            out, volume, _ = flow.bijector_forward_and_log_det_with_extra_apply(theta, sample)
            return out.positions[0], volume[0]
        def apply_inverse(sample):
            out, volume, _ = flow.bijector_inverse_and_log_det_with_extra_apply(theta, sample)
            return out.positions[0], volume[0]
        return jax.lax.cond(direction == 0, apply_forward, apply_inverse, sample)
    # lax.map retains per-row conditional execution; vmap(cond) could evaluate
    # both directions and quietly double baseline work.
    mapped = jax.jit(lambda theta, x, a, d, numbers: jax.lax.map(lambda item: one(theta, item, numbers), (x, a, d)))
    center = np.eye(n)-np.ones((n, n))/n
    basis = jnp.asarray(np.linalg.qr(center[:, :n-1])[0])
    dim_x = 3*(n-1)
    def chart_map(z, direction, numbers):
        x = basis@z[:dim_x].reshape(n-1, 3)
        a = z[dim_x:].reshape(n, 3)
        out, volume = one(theta, (x, a, direction), numbers)
        return jnp.concatenate([(basis.T@out[:, 0]).flatten(), out[:, 1].flatten()]), volume
    chart_jacobian = jax.jit(jax.jacrev(lambda z, d, numbers: chart_map(z, d, numbers)[0], argnums=0))
    emit(dict(ready=True, condition=condition, recipe=recipe, checkpoint_sha256=sha(checkpoint),
        physical_dimension=dim_x, auxiliary_dimension=3*n, auxiliary_scale=recipe['aux_scale'],
        jax_version=jax.__version__, backend=jax.default_backend(), dispatch='sequential lax.map with scalar direction cond'))
    for line in sys.stdin:
        request = json.loads(line)
        started = time.perf_counter()
        try:
            x = np.asarray(request['positions'], dtype=np.float64)
            a = np.asarray(request['auxiliary'], dtype=np.float64)
            d = np.asarray(request['directions'], dtype=np.int32)
            assert x.ndim == 3 and x.shape[1:] == (n, 3) and a.shape == x.shape and d.shape == (len(x),)
            assert np.isfinite(x).all() and np.isfinite(a).all() and np.max(np.abs(x.mean(1))) < 1e-8
            assert np.isin(d, [0, 1]).all()
            order = np.asarray(request.get('order', list(range(n))), dtype=int)
            assert sorted(order.tolist()) == list(range(n))
            numbers = jnp.asarray(np.asarray(condition['numbers'])[order])
            out, volume = mapped(theta, jnp.asarray(x), jnp.asarray(a), jnp.asarray(d), numbers)
            out, volume = np.asarray(out), np.asarray(volume)
            if not np.isfinite(out).all() or not np.isfinite(volume).all():
                raise FloatingPointError('Nonfinite actual EACF transform')
            result = dict(ok=True, positions=out[:, :, 0].tolist(), auxiliary=out[:, :, 1].tolist(),
                log_volume=volume.tolist(), transformations=len(x))
            if request.get('check_inverse'):
                back, bv = mapped(theta, jnp.asarray(out[:, :, 0]), jnp.asarray(out[:, :, 1]), jnp.asarray(1-d), numbers)
                expected = np.stack([x, a], -2)
                result.update(inverse_error=float(np.max(np.abs(np.asarray(back)-expected))),
                    inverse_volume_error=float(np.max(np.abs(np.asarray(bv)+volume))), validation_transformations=len(x))
            if request.get('check_jacobian'):
                z = jnp.concatenate([(basis.T@jnp.asarray(x[0])).flatten(), jnp.asarray(a[0]).flatten()])
                jac = np.asarray(chart_jacobian(z, jnp.asarray(d[0]), numbers))
                logdet = float(np.linalg.slogdet(jac)[1])
                result.update(intrinsic_dimension=len(z), numerical_log_volume=logdet,
                    intrinsic_log_volume_error=abs(logdet-float(volume[0])))
            result['seconds'] = time.perf_counter()-started
            emit(result)
        except Exception as exc:
            emit(dict(ok=False, error=f'{type(exc).__name__}: {exc}', seconds=time.perf_counter()-started))


if __name__ == '__main__':
    main()
