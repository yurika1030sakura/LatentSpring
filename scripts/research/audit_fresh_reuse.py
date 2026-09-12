#!/usr/bin/env python3
"""Replay fresh preparation/budgeted trajectories and independently check joint MH."""
import argparse
import json
import math
from pathlib import Path
import torch
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.joint_chemical_geometry import distinct_anchor_actions
from scripts.research.audit_joint_chemical import independent_defensive_q, independent_log_q
from scripts.research.audit_masked_angular import ReplayOracle, equal, sha
from scripts.research.evaluate_normalized_site import load_model
from scripts.research.fresh_reuse import validate_inputs, prepare, run_budget


def check_ratios(saved, target, model, method, shared):
    checked = 0
    numbers = torch.tensor(target.numbers, dtype=torch.long)
    electronic = torch.tensor([target.condition['charge'], target.condition['spin_multiplicity'], target.kT], dtype=torch.float64)
    counts = [2]*len(saved['parent_ids'])
    for round_index, rows in enumerate(saved['transitions']):
        indices = saved['rounds'][round_index]['active_indices']
        old_ids = saved['history_state_ids'][round_index]
        new_ids = saved['history_state_ids'][round_index+1]
        for index, row in zip(indices, rows):
            old = target.states[old_ids[index]]
            assert row['old_state_id'] == old['state_id']
            counts[index] += 2*int(row['valid'])
            assert new_ids[index] == (row['new_state_id'] if row['accepted'] else row['old_state_id'])
            if row['kind'] != 'joint_exchange' or not row['valid']:
                continue
            new = target.states[row['new_state_id']]
            x, y = old['positions'], new['positions']
            if method == 'learned_vector':
                qf = independent_defensive_q(x, y, old['graph']['bond_orders'], numbers, electronic,
                    target.radii, row['action'], row['order'], model, shared, row['forward'])
                qr = independent_defensive_q(y, x, new['graph']['bond_orders'], numbers, electronic,
                    target.radii, row['inverse_action'], row['order'], model, shared, row['reverse'])
            else:
                qf = independent_log_q(x, y, old['graph']['bond_orders'], numbers, electronic,
                    target.radii, row['action'], row['order'], 'site', None, shared, row['forward'])
                qr = independent_log_q(y, x, new['graph']['bond_orders'], numbers, electronic,
                    target.radii, row['inverse_action'], row['order'], 'site', None, shared, row['reverse'])
            nf = len(distinct_anchor_actions(numbers, old['graph']['bond_orders']))
            nr = len(distinct_anchor_actions(numbers, new['graph']['bond_orders']))
            ratio = -float(new['potential_eV']-old['potential_eV'])/target.kT+qr-qf+math.log(nf/nr)
            assert abs(ratio-row['log_acceptance_ratio']) < 1e-7
            assert row['accepted'] == (row['log_uniform'] < min(0., ratio))
            checked += 1
        assert counts == saved['query_count_history'][round_index+1]
    assert counts == saved['final_queries_per_parent']
    return checked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['project','run','out']:
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    pp = root/'research/evidence/fresh_reuse_protocol_v1.json'
    protocol = json.loads(pp.read_text())
    data, source_audit, physical, shared = validate_inputs(root, args.project, protocol)
    report = json.loads((args.run/'results.json').read_text())
    assert report['complete'] and report['protocol_sha256'] == sha(pp)
    assert report['condition'] == data['condition']
    assert report['new_raw_queries'] == report['requested_raw_queries']
    assert report['source_audit_sha256'] == sha(args.project/protocol['source_audit'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    audits = []
    checked = 0
    if report['phase'] == 'prepare':
        assert sha(args.run/'trace.pt') == report['trace_sha256']
        saved = torch.load(args.run/'trace.pt', map_location='cpu', weights_only=False)
        oracle = ReplayOracle(saved['query_trace'])
        target = ChemicalTarget(oracle, data['condition'], physical['kT_eV'], physical['restraint_eV_A2'])
        actual = {}
        prepare(target, data, source_audit, protocol, actual)
        actual.update(condition=data['condition'], states=target.states, query_trace=target.query_trace)
        equal(actual, saved)
        assert oracle.index == len(oracle.queries) and oracle.evaluated == report['new_raw_queries']
        independent_counts = [2+2*sum(int(rows[i]['valid']) for rows in saved['transitions']) for i in range(len(saved['parent_ids']))]
        assert independent_counts == saved['final_queries_per_parent'] == report['raw_queries_per_parent']
        audits.append(dict(file='trace.pt', raw_queries=oracle.evaluated, maximum_position_error_A=oracle.maximum_position_error))
    else:
        prep = args.project/protocol['preparation_run']
        assert sha(prep/'results.json') == report['preparation_results_sha256']
        assert sha(prep/'trace.pt') == report['preparation_trace_sha256']
        warm = torch.load(prep/'trace.pt', map_location='cpu', weights_only=False)
        model, metadata = load_model(args.project/'runs/normalized_site_train_v1', report['method'], report['replica'],
            shared, protocol['training_artifact_sha256'])
        assert report['policy_sha256'] == (metadata['checkpoint_sha256'] if metadata else None)
        assert report['one_time_training_raw_queries'] == (protocol['one_time_training_raw_queries'] if metadata else 0)
        offset = 0
        flattened = []
        for batch, chunk in enumerate(report['chunks']):
            path = args.run/chunk['file']
            assert sha(path) == chunk['sha256']
            saved = torch.load(path, map_location='cpu', weights_only=False)
            assert saved['raw_query_offset'] == offset
            begin = batch*protocol['batch']
            end = min(len(warm['parent_ids']), begin+protocol['batch'])
            assert saved['parent_ids'] == chunk['parent_ids'] == warm['parent_ids'][begin:end]
            flattened.extend(chunk['parent_ids'])
            oracle = ReplayOracle(saved['query_trace'])
            oracle.evaluated = offset
            target = ChemicalTarget(oracle, data['condition'], physical['kT_eV'], physical['restraint_eV_A2'])
            actual = dict(condition=data['condition'], raw_query_offset=offset)
            run_budget(target, warm['warm_positions'][begin:end], saved['parent_ids'], model,
                report['method'], report['replica'], batch, protocol, shared, actual)
            actual.update(states=target.states, query_trace=target.query_trace)
            equal(actual, saved)
            for local_index, state_id in enumerate(saved['history_state_ids'][0]):
                original = warm['states'][warm['history_state_ids'][-1][begin+local_index]]
                observed = saved['states'][state_id]
                torch.testing.assert_close(observed['positions'], original['positions'], atol=0, rtol=0)
                torch.testing.assert_close(observed['energy_eV'], original['energy_eV'], atol=1e-4, rtol=0)
                torch.testing.assert_close(observed['force_eV_A'], original['force_eV_A'], atol=1e-4, rtol=0)
            assert oracle.index == len(oracle.queries) and oracle.evaluated-offset == chunk['raw_queries']
            checked += check_ratios(saved, target, model, report['method'], shared)
            audits.append(dict(file=chunk['file'], trace_sha256=sha(path), parent_ids=chunk['parent_ids'],
                raw_queries=chunk['raw_queries'], maximum_position_error_A=oracle.maximum_position_error))
            offset = oracle.evaluated
        assert offset == report['new_raw_queries']
        assert flattened == report['parent_ids'] == warm['parent_ids']
    result = dict(complete=True, phase=report['phase'], method=report['method'], replica=report['replica'],
        results_sha256=sha(args.run/'results.json'), protocol_sha256=sha(pp),
        full_producer_replay=True, all_random_streams_replayed=True,
        independent_joint_MH_ratios=checked, independent_per_parent_query_accounting=True,
        raw_queries=report['new_raw_queries'], new_physical_queries=0, chunks=audits,
        scientific_submission_ready=False)
    if report['phase'] == 'prepare':
        result['trace_sha256'] = report['trace_sha256']
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'chunks'}))


if __name__ == '__main__':
    main()
