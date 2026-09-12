#!/usr/bin/env python3
"""Summarize audited direction-MH pilots and frozen controls without new queries."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    x = torch.tensor(values, dtype=torch.float64)
    return dict(count=len(x), minimum=float(x.min()), median=float(x.median()), maximum=float(x.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.project
    pp = root/'research/evidence/eacf_directional_protocol_v1.json'
    protocol = json.loads(pp.read_text())
    rows = []
    for profile in protocol['profiles']:
        for rep in protocol['replicas']:
            run = root/f'runs/eacf_directional_eval_v1/{profile}_s{rep}'
            ap = root/f'runs/eacf_directional_audit_v1/{profile}_s{rep}/results.json'
            audit = json.loads(ap.read_text())
            report = json.loads((run/'results.json').read_text())
            assert audit['complete'] and report['complete']
            assert sha(run/'results.json') == audit['results_sha256']
            assert report['protocol_sha256'] == sha(pp)
            assert sha(run/'trace.pt') == audit['trace_sha256'] == report['trace_sha256']
            assert audit['full_producer_and_real_flow_replay'] and audit['all_random_streams_replayed']
            trace = torch.load(run/'trace.pt', map_location='cpu', weights_only=False)
            selected = [r for r in trace['transitions'] if r['kind'] == 'directional_flow' and r['valid']]
            terms = dict(log_acceptance_ratio=[], auxiliary_and_volume=[], negative_potential_change_over_kT=[])
            for r in selected:
                correction = float(r['reverse_auxiliary_log_prob']-r['forward_auxiliary_log_prob']+r['log_volume'])
                terms['log_acceptance_ratio'].append(r['log_acceptance_ratio'])
                terms['auxiliary_and_volume'].append(correction)
                terms['negative_potential_change_over_kT'].append(r['log_acceptance_ratio']-correction)
            assert len(selected) == audit['moves']['directional_flow']['valid']
            rows.append(dict(method='eacf_'+profile, replica=rep, audit_path=str(ap.relative_to(root)),
                audit_sha256=sha(ap), results_sha256=sha(run/'results.json'),
                reference_first_hit_steps=audit['reference_connectivity_first_hit_step'],
                nonlocal_moves=audit['moves']['directional_flow'],
                new_raw_queries=report['new_raw_queries'], training_raw_queries=report['inherited_training_raw_queries'],
                source_raw_queries=report['inherited_source_raw_queries'], warm_raw_queries=report['inherited_development_warm_raw_queries'],
                total_raw_queries=audit['total_raw_queries'], prior_validation_raw_queries=report['prior_generator_validation_raw_queries'],
                seconds=report['seconds'], first_transport_seconds=report['first_transport_seconds'],
                transport_seconds=report['transport_seconds'], valid_proposal_terms={k:stats(v) for k,v in terms.items()}))
    cp = root/'research/evidence/normalized_site_audit_v1.json'
    controls = json.loads(cp.read_text())
    assert controls['complete'] and controls['evaluation_protocol_sha256'] == protocol['shared_evaluation_protocol_sha256']
    for a in controls['rows']:
        if a['method'] not in protocol['reused_control_sha256']:
            continue
        path = root/f"runs/normalized_site_eval_v1/{a['method']}_s{a['replica']}/results.json"
        r = json.loads(path.read_text())
        assert sha(path) == protocol['reused_control_sha256'][a['method']][a['replica']]
        assert a['full_producer_replay'] and a['all_random_streams_replayed'] and r['complete']
        rows.append(dict(method=a['method'], replica=a['replica'], audit_path=str(cp.relative_to(root)),
            audit_sha256=sha(cp), results_sha256=sha(path),
            reference_first_hit_steps=a['reference_connectivity_first_hit_step'],
            nonlocal_moves={k:a['moves']['joint_exchange'][k] for k in ['attempts','valid','accepted']},
            new_raw_queries=r['new_raw_queries'], training_raw_queries=r['inherited_training_raw_queries'],
            source_raw_queries=r['inherited_source_raw_queries'], warm_raw_queries=r['inherited_development_warm_raw_queries'],
            total_raw_queries=a['total_raw_queries'], seconds=r['seconds']))
    assert len(rows) == 8
    out = dict(complete=True, protocol_sha256=sha(pp), rows=rows,
        new_physical_queries_in_eacf_experiment=sum(r['new_raw_queries'] for r in rows if r['method'].startswith('eacf_')),
        new_physical_queries_in_summary=0, scientific_submission_ready=False,
        scope='One repeatedly inspected ten-atom composition, four common starts, two seeds; identical target, 256 microsteps and common local schedule, unequal actual cost.',
        baseline_limit='These are frozen EACF refinement checkpoints repurposed as direction-MH proposals. They were not trained for this MH acceptance objective and are not native EACF/FAB samplers. Zero observed acceptance does not establish an architectural impossibility or general generator superiority.',
        positive_signal='Learned vectors reach the diagnostic reference graph from all four starts in both seeds; the frozen EACF proposal pilots do not. This motivates fresh-start validation, not a generalization claim.',
        unresolved=['Independent composition evidence for the learned component','Measured reuse and total-cost advantage over the physical site control','Distribution calibration for any equilibrium claim','Distinct method contribution relative to learned molecular MH and directional-mixture prior work'])
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2)+'\n')
    for row in rows:
        print(row['method'], row['replica'], row['reference_first_hit_steps'], row['nonlocal_moves'], row['total_raw_queries'])


if __name__ == '__main__':
    main()
