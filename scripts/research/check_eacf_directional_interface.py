#!/usr/bin/env python3
"""Qualify real isolated EACF maps for direction-augmented MH, without energy calls."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from cfm_mol.eacf_transport import EACFTransport
from scripts.research.evaluate_chemical_policy import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'table', 'transport-python', 'upstream', 'out']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--check-jacobian', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out/'results.json'
    if output.exists():
        raise FileExistsError(output)
    saved_report = json.loads((args.run/'results.json').read_text())
    recipe = saved_report['recipe']
    shape = tuple(recipe[k] if not isinstance(recipe[k], list) else tuple(recipe[k]) for k in
                  ['n_layers', 'n_blocks', 'mlp_units', 'n_invariant_feat_hidden', 'embedding_dim'])
    profiles = {(12, 3, (64, 64), 128, 32): 'published', (2, 1, (12,), 12, 8): 'compact'}
    if shape not in profiles or saved_report.get('architecture_profile', profiles[shape]) != profiles[shape]:
        raise ValueError('Unknown or inconsistent frozen architecture profile')
    for name, digest in saved_report['artifacts'].items():
        assert sha(args.run/name) == digest
    archived = np.load(args.run/'samples.npz')
    table_report = json.loads((args.table/'results.json').read_text())
    assert table_report['complete'] and sha(args.table/'development.pt') == table_report['artifacts']['development']
    warm = torch.load(args.table/'development.pt', map_location='cpu', weights_only=False)
    assert warm['condition'] == saved_report['condition'] and warm['stream'] == 'development'
    report = dict(complete=False, profile=profiles[shape], replica=saved_report['replica'],
        checkpoint_sha256=sha(args.run/'adapter.pkl'), development_sha256=sha(args.table/'development.pt'),
        original_results_sha256=sha(args.run/'results.json'), new_physical_queries=0,
        scientific_submission_ready=False, checks=[])
    traces = []
    write(output, report)
    start = time.perf_counter()
    transport = None
    try:
        transport = EACFTransport(args.transport_python, root/'scripts/research/eacf_transport_worker.py',
            args.run, args.upstream, log_path=args.out/'worker.log')
        report['runtime'] = transport.handshake
        assert transport.handshake['checkpoint_sha256'] == report['checkpoint_sha256']
        n = len(saved_report['condition']['numbers'])
        assert transport.handshake['physical_dimension'] == 3*(n-1)
        assert transport.handshake['auxiliary_dimension'] == 3*n
        def call(label, x, a, d, **kw):
            response = transport.transform(x, a, d, **kw)
            traces.append(dict(label=label, positions=x, auxiliary=a, directions=d, options=kw, response=response))
            if kw.get('check_inverse'):
                assert response['inverse_error'] < 1e-7 and response['inverse_volume_error'] < 1e-7
            report['checks'].append(dict(label=label, seconds=response['seconds'],
                inverse_error=response.get('inverse_error'), inverse_volume_error=response.get('inverse_volume_error')))
            write(output, report)
            return response
        x = torch.from_numpy(archived['base_positions'][:4].copy())
        a = torch.from_numpy(archived['base_auxiliary'][:4].copy())
        forward = call('archived_forward', x, a, torch.zeros(4, dtype=torch.long), check_inverse=True)
        for key, array in [('positions', 'positions'), ('auxiliary', 'auxiliary'), ('log_volume', 'log_volume')]:
            torch.testing.assert_close(forward[key], torch.from_numpy(archived[array][:4].copy()), atol=1e-7, rtol=1e-8)
        backward = call('archived_inverse', forward['positions'], forward['auxiliary'], torch.ones(4, dtype=torch.long))
        torch.testing.assert_close(backward['positions'], x, atol=1e-7, rtol=1e-8)
        torch.testing.assert_close(backward['auxiliary'], a, atol=1e-7, rtol=1e-8)
        torch.testing.assert_close(backward['log_volume']+forward['log_volume'], torch.zeros(4, dtype=x.dtype), atol=1e-7, rtol=0)
        x = torch.stack([warm['states'][i]['positions'] for i in warm['warm_state_ids']])
        rng = torch.Generator().manual_seed(2405+saved_report['replica'])
        a = x+transport.aux_scale*torch.randn(x.shape, dtype=x.dtype, generator=rng)
        directions = torch.tensor([0, 1, 0, 1])
        current = call('mixed_directions_on_warm_states', x, a, directions, check_inverse=True)
        rotation = torch.linalg.qr(torch.randn(3, 3, dtype=x.dtype, generator=rng))[0]
        rotation[:, 0] *= -torch.linalg.det(rotation)
        reflected = call('orthogonal_reflection', x@rotation, a@rotation, directions)
        torch.testing.assert_close(reflected['positions'], current['positions']@rotation, atol=1e-7, rtol=1e-8)
        torch.testing.assert_close(reflected['auxiliary'], current['auxiliary']@rotation, atol=1e-7, rtol=1e-8)
        torch.testing.assert_close(reflected['log_volume'], current['log_volume'], atol=1e-7, rtol=1e-8)
        order = list(range(n-1, -1, -1))
        permuted = call('joint_atom_permutation', x[:, order], a[:, order], directions, order=order)
        torch.testing.assert_close(permuted['positions'], current['positions'][:, order], atol=1e-7, rtol=1e-8)
        torch.testing.assert_close(permuted['auxiliary'], current['auxiliary'][:, order], atol=1e-7, rtol=1e-8)
        torch.testing.assert_close(permuted['log_volume'], current['log_volume'], atol=1e-7, rtol=1e-8)
        if args.check_jacobian:
            jac = call('full_intrinsic_joint_jacobian', x[:1], a[:1], torch.zeros(1, dtype=torch.long), check_jacobian=True)
            assert jac['intrinsic_dimension'] == 6*n-3 and jac['intrinsic_log_volume_error'] < 1e-6
            report['intrinsic_log_volume_error'] = jac['intrinsic_log_volume_error']
        torch.save(traces, args.out/'trace.pt')
        report.update(complete=True, trace_sha256=sha(args.out/'trace.pt'), seconds=time.perf_counter()-start,
            requested_transformations=transport.requested_transformations, completed_transformations=transport.completed_transformations,
            validation_transformations=transport.validation_transformations, full_jacobian_checked=args.check_jacobian,
            limitation='Real-interface numerical qualification on saved and warm coordinates; no molecular efficiency or equilibrium claim.')
        write(output, report)
    except Exception as exc:
        torch.save(traces, args.out/'failed_trace.pt')
        report.update(failure=f'{type(exc).__name__}: {exc}',
            requested_transformations=transport.requested_transformations if transport else 0,
            completed_transformations=transport.completed_transformations if transport else 0)
        write(output, report)
        raise
    finally:
        if transport is not None:
            transport.close()


if __name__ == '__main__':
    main()
