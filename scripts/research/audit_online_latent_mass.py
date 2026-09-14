#!/usr/bin/env python3
"""Replay every online estimator and a prespecified subset of adaptive training."""
import argparse
import json
import math
from pathlib import Path
import torch
from cfm_mol.latent_mass_coupling import log_weight
from scripts.research.latent_mass_calibration import draw
from scripts.research.online_latent_mass import run_history
from scripts.research.audit_masked_angular import sha
from scripts.research.evaluate_chemical_policy import write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['protocol', 'run', 'out']:
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--chunk', type=int, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    torch.set_num_threads(2)
    protocol = json.loads(a.protocol.read_text())
    source = a.run / f's{a.chunk}'
    report = json.loads((source / 'results.json').read_text())
    assert report['complete'] and report['chunk'] == a.chunk
    assert report['protocol_sha256'] == sha(a.protocol)
    assert report['trace_sha256'] == sha(source / 'trace.pt')
    data = torch.load(source / 'trace.pt', map_location='cpu', weights_only=False)
    expected = set()
    for si, scenario in enumerate(protocol['scenarios']):
        for history in range(a.chunk * protocol['histories_per_chunk'], (a.chunk + 1) * protocol['histories_per_chunk']):
            seed = protocol['base_seed'] + 1000000 * si + 10000 * history
            expected.update((scenario, seed, method) for method in protocol['methods'])
    assert len(data) == len(expected) == len(report['rows'])
    assert {(r['scenario'], r['seed'], r['method']) for r in data} == expected
    calls = 0
    replayed = []
    for row, compact in zip(data, report['rows']):
        assert {k: v for k, v in row.items() if k != 'batches'} == compact
        n = protocol['pairs_per_batch']
        assert len(row['batches']) * n == protocol['total_pairs']
        labels = []
        for stage, batch in enumerate(row['batches']):
            assert batch['stage'] == stage and batch['learned_from_previous_pairs'] == stage * n
            assert batch['seed'] == row['seed'] + 1001 + stage * 100
            g = torch.Generator().manual_seed(batch['seed'])
            z, eps = draw(n, g), draw(n, g)
            torch.testing.assert_close(z, batch['input_z'], atol=0, rtol=0)
            if row['method'] == 'independent':
                torch.testing.assert_close(eps, batch['second_z'], atol=0, rtol=0)
            elif row['method'] == 'identity' or stage == 0:
                torch.testing.assert_close(z, batch['second_z'], atol=0, rtol=0)
            weights = torch.stack([log_weight(z, 0, row['scenario']), log_weight(batch['second_z'], 1, row['scenario'])], 1)
            torch.testing.assert_close(weights, batch['log_weights'], atol=1e-12, rtol=0)
            labels.append(weights)
            completed = (stage + 1) * n
            logz = torch.logsumexp(torch.cat(labels), 0) - math.log(completed)
            cp = row['checkpoints'][str(completed)]
            assert cp['target_calls'] == 2 * completed
            torch.testing.assert_close(logz, logz.new_tensor(cp['log_normalizers']), atol=1e-12, rtol=0)
            assert abs(float((logz[1] - logz[0]).sigmoid()) - cp['component_mass']) < 1e-12
            calls += 2 * n
        # First history in each chunk/case, fixed independently of performance.
        si = protocol['scenarios'].index(row['scenario'])
        first_seed = protocol['base_seed'] + 1000000 * si + 10000 * a.chunk * protocol['histories_per_chunk']
        if row['seed'] == first_seed:
            fresh = run_history(row['method'], row['scenario'], row['seed'], protocol)
            for old, new in zip(row['batches'], fresh['batches']):
                for field in ['input_z', 'second_z', 'log_weights']:
                    torch.testing.assert_close(old[field], new[field], atol=1e-10, rtol=1e-10)
            replayed.append(dict(scenario=row['scenario'], seed=row['seed'], method=row['method']))
    assert calls == report['total_analytic_target_calls']
    write(a.out, dict(complete=True, chunk=a.chunk, protocol_sha256=sha(a.protocol),
        source_results_sha256=sha(source / 'results.json'), source_trace_sha256=sha(source / 'trace.pt'),
        histories_checked=len(data), analytic_weights_checked=calls, all_prefix_estimators_replayed=True,
        full_training_and_adaptation_replayed=replayed, new_molecular_oracle_calls=0,
        scope='All saved weights, fresh base draws, prefix estimates and costs checked; complete optimizer/coupling replay for the first history of every chunk, scenario and method. Remaining training histories are not claimed independently replayed.'))
    print(json.dumps(dict(chunk=a.chunk, histories=len(data), checked=calls, full_replays=len(replayed))), flush=True)


if __name__ == '__main__':
    main()
