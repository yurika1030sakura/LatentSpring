#!/usr/bin/env python3
"""Summarize complete audited finite-catalogue molecular policy expectations."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.research.audit_masked_angular import sha
from scripts.research.evaluate_chemical_policy import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'audit', 'out', 'protocol'): parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); protocol = json.loads(args.protocol.read_text())
    parents, provenance = [], {}; total = trace_calls = checks = reused = 0
    for index in [0, 1, 2, 3, 5, 7]:
        path = args.run/f'condition_{index:02d}'; ap = args.audit/f'condition_{index:02d}/results.json'
        report = json.loads((path/'results.json').read_text()); audit = json.loads(ap.read_text())
        assert report['complete'] and audit['complete'] and audit['full_replay']
        assert report['protocol_sha256'] == audit['protocol_sha256'] == sha(args.protocol)
        assert sha(path/'results.json') == audit['source_results_sha256']
        assert sha(path/'trace.pt') == report['trace_sha256'] == audit['trace_sha256']
        assert report.get('physical_calls_in_trace', report['new_raw_queries']) == audit['producer_raw_queries'] == protocol['counts'][str(index)]['raw_queries']
        total += report['new_raw_queries']; checks += audit['independent_ratio_checks']
        trace_calls += audit['producer_raw_queries']; reused += report.get('reused_raw_queries', 0)
        parents.extend(dict(index=index, **row) for row in report['rows'])
        provenance[str(index)] = dict(results_sha256=sha(path/'results.json'), trace_sha256=sha(path/'trace.pt'), audit_sha256=sha(ap))
    assert len(parents) == 18 and total == protocol['maximum_new_raw_queries']
    assert trace_calls == protocol.get('maximum_trace_raw_queries', protocol['maximum_new_raw_queries'])
    methods = list(parents[0]['methods'])
    means = {m: {key: float(np.mean([p['methods'][m][key] for p in parents]))
                  for key in parents[0]['methods'][m]} for m in methods}
    values = {m: np.array([p['methods'][m]['utility_eV'] for p in parents]) for m in methods}
    for variant in ['linear', 'graph', 'geometry', 'graph_residual', 'geometry_residual']:
        if variant+'_s0' in values: values[variant] = (values[variant+'_s0']+values[variant+'_s1'])/2
    rng = np.random.default_rng(28691)
    boot = rng.integers(0, len(parents), size=(10000, len(parents)))
    comparisons = {}
    pairs = [('geometry','uniform'), ('geometry','force'), ('geometry','linear'), ('geometry','graph'),
             ('linear','uniform'), ('linear','force'), ('graph','uniform')]
    if 'geometry_residual' in values:
        pairs += [('geometry_residual', m) for m in ('linear', 'graph_residual', 'geometry', 'uniform', 'force')]
        pairs += [('graph_residual', 'linear')]
    for left, right in pairs:
        difference = values[left]-values[right]
        comparisons[left+' minus '+right] = dict(mean_utility_difference_eV=float(difference.mean()),
            descriptive_parent_bootstrap95=np.quantile(difference[boot].mean(1), [.025,.975]).tolist(),
            parents_positive=int((difference > 0).sum()), parent_differences_eV=difference.tolist())
    if args.out.exists(): raise FileExistsError(args.out)
    write(args.out, dict(complete=True, protocol_sha256=sha(args.protocol), sources=provenance, parents=parents,
        mean_one_step=means, comparisons=comparisons, new_raw_queries=total, evaluation_queries_reused=reused,
        physical_calls_in_trace=trace_calls, independent_ratio_checks=checks,
        training_queries_reused=966, scientific_submission_ready=False,
        scope=protocol['interpretation'], uncertainty='Descriptive parent bootstrap over repeatedly used development parents; model-seed outcomes averaged within each parent, not counted as independent parents.'))


if __name__ == '__main__': main()
