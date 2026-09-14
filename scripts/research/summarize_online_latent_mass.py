#!/usr/bin/env python3
"""Summarize all-cost calibration errors across independent adaptive histories."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from cfm_mol.latent_mass_coupling import exact_targets
from scripts.research.audit_masked_angular import sha
from scripts.research.evaluate_chemical_policy import write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['protocol', 'run', 'audit', 'out', 'csv']:
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    if a.out.exists() or a.csv.exists():
        raise FileExistsError('Preserve earlier summaries')
    protocol = json.loads(a.protocol.read_text())
    rows, sources, audited_calls, full_replays = [], {}, 0, 0
    for chunk in range(protocol['chunks']):
        rp, ap = a.run / f's{chunk}/results.json', a.audit / f's{chunk}.json'
        r, audit = json.loads(rp.read_text()), json.loads(ap.read_text())
        assert r['complete'] and audit['complete']
        assert r['protocol_sha256'] == audit['protocol_sha256'] == sha(a.protocol)
        assert audit['source_results_sha256'] == sha(rp)
        assert audit['source_trace_sha256'] == r['trace_sha256'] == sha(a.run / f's{chunk}/trace.pt')
        assert audit['all_prefix_estimators_replayed']
        audited_calls += audit['analytic_weights_checked']
        full_replays += len(audit['full_training_and_adaptation_replayed'])
        sources[str(chunk)] = dict(results_sha256=sha(rp), audit_sha256=sha(ap), trace_sha256=r['trace_sha256'])
        rows.extend(r['rows'])
    truth = exact_targets()
    target_mass = truth['component_mass'][1]
    target_logratio = np.log(truth['normalizers'][1] / truth['normalizers'][0])
    output, flat = [], []
    for scenario in protocol['scenarios']:
        by_method = {m: sorted([r for r in rows if r['scenario'] == scenario and r['method'] == m], key=lambda r: r['seed']) for m in protocol['methods']}
        seeds = [r['seed'] for r in by_method['independent']]
        assert len(seeds) == len(set(seeds)) == protocol['chunks'] * protocol['histories_per_chunk']
        assert all([r['seed'] for r in v] == seeds for v in by_method.values())
        bootstrap = np.random.default_rng(77101).integers(len(seeds), size=(10000, len(seeds)))
        for n in range(protocol['pairs_per_batch'], protocol['total_pairs'] + 1, protocol['pairs_per_batch']):
            methods, errors = {}, {}
            for m, group in by_method.items():
                cp = [r['checkpoints'][str(n)] for r in group]
                assert all(c['target_calls'] == 2 * n for c in cp)
                mass = np.array([c['component_mass'] for c in cp])
                logz = np.array([c['log_normalizers'] for c in cp])
                err = mass - target_mass
                errors[m] = err
                rmse = float(np.sqrt(np.mean(err ** 2)))
                ci = np.quantile(np.sqrt(np.mean(err[bootstrap] ** 2, axis=1)), [.025, .975]).tolist()
                methods[m] = dict(mass_RMSE=rmse, mass_bias=float(err.mean()), mass_RMSE_bootstrap95=ci,
                    log_ratio_RMSE=float(np.sqrt(np.mean((logz[:, 1] - logz[:, 0] - target_logratio) ** 2))),
                    mean_normalizers=np.exp(logz).mean(0).tolist(), target_calls_per_history=2 * n,
                    full_256_pair_seconds_mean=float(np.mean([r['seconds'] for r in group])))
                flat.append(dict(scenario=scenario, method=m, pairs=n, target_calls=2*n, histories=len(seeds),
                    mass_RMSE=rmse, mass_bias=float(err.mean()), rmse_low=ci[0], rmse_high=ci[1],
                    full_256_pair_seconds_mean=methods[m]['full_256_pair_seconds_mean']))
            comparisons = {}
            for control in ['independent', 'identity', 'constant', 'gaussian', 'binned']:
                delta = errors['nonlinear'] ** 2 - errors[control] ** 2
                comparisons['nonlinear minus ' + control] = dict(paired_MSE_difference=float(delta.mean()),
                    paired_history_bootstrap95=np.quantile(delta[bootstrap].mean(1), [.025, .975]).tolist(),
                    RMSE_ratio=methods['nonlinear']['mass_RMSE'] / methods[control]['mass_RMSE'])
            output.append(dict(scenario=scenario, pairs=n, histories=len(seeds), methods=methods, comparisons=comparisons))
    write(a.out, dict(complete=True, protocol_sha256=sha(a.protocol), sources=sources, rows=output, exact=truth,
        histories_checked=len(rows), analytic_target_calls=audited_calls, full_training_histories_replayed=full_replays,
        every_weight_and_prefix_replayed=True, new_molecular_oracle_calls=0, scientific_submission_ready=False,
        uncertainty='Descriptive paired bootstrap over independent complete adaptive histories on fixed constructed targets; 32 histories per case, no multiplicity adjustment or molecular generalization.',
        limits=['All learning queries are counted and retained. Compute time is reported separately; no matched-wall-time superiority follows.',
                'The target deliberately contains a radius-dependent twist. Generic coupling optimization and Gaussian-preserving rearrangement have direct prior art.',
                'Reference shapes remain fixed and misspecified. Better component mass estimation need not improve joint reverse KL.',
                'Normalizers are conditionally unbiased in exact arithmetic; ratios, masses and logs generally have finite-sample bias.',
                'Only 48 of 384 complete optimization histories are replayed; all saved target weights and prefix estimates are checked.']))
    a.csv.parent.mkdir(parents=True, exist_ok=True)
    with a.csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps([r for r in output if r['pairs'] == protocol['total_pairs']]))


if __name__ == '__main__':
    main()
