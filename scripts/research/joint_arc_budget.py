#!/usr/bin/env python3
"""Bounded TRAINING-only full-chain comparison of legacy and constrained proposals."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.joint_chemical_geometry import distinct_anchor_actions
from scripts.research.audit_joint_arc_support import independent_arc_q
from scripts.research.audit_joint_chemical import independent_log_q
from scripts.research.audit_masked_angular import ReplayOracle, equal, sha
from scripts.research.evaluate_chemical_policy import write
from scripts.research.fresh_reuse import run_budget, parent_query_caps


def validate_chain_cost_contract(project, protocol, preparation_audit):
    """Independently recover shared preparation and model data costs before queries."""
    selected = [(int(i), pid) for i, ids in protocol['parent_ids_by_condition'].items() for pid in ids]
    assert len(selected) == len(set(selected)) == 48
    ordered = sorted(selected, key=lambda pair: hashlib.sha256(
        f"{protocol['cost_assignment_seed']}|{pair[0]}|{pair[1]}".encode()).hexdigest())
    extra = set(ordered[:39]); total = 0
    for index, pid in selected:
        cap = protocol['query_caps_by_parent']['arc_site'][str(index)][str(pid)]
        assert cap == (438 if (index, pid) in extra else 436)
        total += cap
    shared = full = 0
    for row in preparation_audit['rows']:
        if row['zero_support']: continue
        path = project/protocol['preparation_run']/f"condition_{row['index']:02d}/results.json"
        assert sha(path) == row['results_sha256']
        data = json.loads(path.read_text())
        for pid, cost in zip(data['parent_ids'], data['raw_queries_per_parent']):
            full += cost
            if (row['index'], pid) in selected: shared += cost
    probe = json.loads((project/'runs/multicomposition_angular_audit_v1/results.json').read_text())
    assert probe['complete']
    labels = probe['raw_queries_in_probes']
    for run, indices in [('arc_oracle_feasibility', [1,2,3,5]), ('arc_internal_validation', [0,1,2,3,5,7])]:
        for index in indices:
            path = project/f'runs/{run}_audit_v1/condition_{index:02d}/results.json'
            audit = json.loads(path.read_text()); assert audit['complete'] and audit['full_replay']
            labels += audit['raw_queries']
    overhead = full-shared+labels
    assert full == protocol['common_preparation_raw_queries_all_96_parents']
    assert shared == protocol['common_selected_preparation_raw_queries'] == 6088
    assert overhead == protocol['model_specific_raw_overhead_per_method_replica'] == 14862
    assert total == protocol['primary_total_raw_calls_per_method_replica'] == 48*128+overhead
    assert len(protocol['replicas'])*(total+2*48*128) == protocol['maximum_total_new_raw_queries']


def inputs(root, project, index, protocol_path=None):
    pp = protocol_path or root/'research/evidence/joint_arc_budget_protocol_v1.json'
    protocol = json.loads(pp.read_text())
    assert protocol['frozen'] and index in protocol['condition_indices']
    for name in ['preparation_audit', 'joint_support_audit']:
        path = project/protocol[name]
        assert sha(path) == protocol[name+'_sha256']
        assert json.loads(path.read_text())['complete']
    ap = project/protocol['preparation_audit']
    audit = json.loads(ap.read_text())
    assert audit['full_producer_replay'] and audit['all_eight_conditions_retained']
    row = next(r for r in audit['rows'] if r['index'] == index)
    directory = project/protocol['preparation_run']/f'condition_{index:02d}'
    assert sha(directory/'results.json') == row['results_sha256']
    assert sha(directory/'trace.pt') == row['trace_sha256']
    header = json.loads((directory/'results.json').read_text())
    warm = torch.load(directory/'trace.pt', map_location='cpu', weights_only=False)
    assert header['complete'] and header['stream'] == 'fresh_training'
    assert warm['condition'] == header['condition']
    split_path = root/'research/evidence/multicomposition_angular_probe_protocol_v1.json'
    assert sha(split_path) == protocol['split_protocol_sha256']
    split = json.loads(split_path.read_text())['condition_splits'][str(index)]
    ids = protocol['parent_ids_by_condition'][str(index)]
    if protocol.get('selection_role') == 'internal_withheld_parent_or_composition':
        assert ids == split['withheld_parent_ids']+split['withheld_composition_parent_ids']
        assert not set(ids) & set(split['fit_parent_ids'])
        ap = project/protocol['learning_audit']
        assert sha(ap) == protocol['learning_audit_sha256']
        audit = json.loads(ap.read_text())
        assert audit['complete'] and audit['full_data_rebuilt'] and audit['all_split_masks_rebuilt']
        assert audit['protocol_sha256'] == protocol['training_protocol_sha256']
        validate_chain_cost_contract(project, protocol, json.loads((project/protocol['preparation_audit']).read_text()))
        for name, info in protocol['models'].items():
            assert next(r for r in audit['rows'] if r['name'] == name)['model_sha256'] == info['sha256']
    else:
        assert ids == split['fit_parent_ids'] and len(ids) == 12
        assert not set(ids) & set(split['withheld_parent_ids'])
    indices = [warm['parent_ids'].index(i) for i in ids]
    positions = warm['warm_positions'][indices]
    physical_path = root/'research/evidence/parity_training_protocol_v1.json'
    assert sha(physical_path) == protocol['physical_protocol_sha256']
    return pp, protocol, header, warm, positions, ids, json.loads(physical_path.read_text())


def load_scalar_model(project, protocol, method, replica):
    if method not in protocol.get('model_methods', {}):
        return None, None
    from cfm_mol.conditional_arc_energy import ConditionalArcEnergy
    objective = protocol['model_methods'][method]
    info = protocol['models'][f'{objective}_s{replica}']
    path = project/info['path']
    if sha(path) != info['sha256']:
        raise ValueError('Scalar model differs from the frozen checkpoint')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    assert checkpoint['objective'] == objective and checkpoint['replica'] == replica
    assert checkpoint['protocol_sha256'] == protocol['training_protocol_sha256']
    model = ConditionalArcEnergy(**checkpoint['configuration']).double()
    model.load_state_dict(checkpoint['state_dict']); model.eval(); model.requires_grad_(False)
    return model, info['sha256']


def diagnostics(saved, readouts):
    endpoints = []
    for i, parent in enumerate(saved['parent_ids']):
        first = saved['states'][saved['history_state_ids'][0][i]]
        for readout in readouts:
            cap = saved['query_caps_per_parent'][i] if readout == 'final' else readout
            hit = next((j for j, counts in enumerate(saved['query_count_history']) if counts[i] == cap), None)
            row = dict(parent=parent, raw_queries=cap, reached=hit is not None)
            if 'final' in readouts: row['readout'] = readout
            if hit is not None:
                end = saved['states'][saved['history_state_ids'][hit][i]]
                smiles = [saved['states'][ids[i]]['graph']['connectivity_smiles'] for ids in saved['history_state_ids'][:hit+1]]
                row.update(initial_potential_eV=float(first['potential_eV']), potential_eV=float(end['potential_eV']),
                    potential_change_eV=float(end['potential_eV']-first['potential_eV']),
                    initial_smiles=first['graph']['connectivity_smiles'], final_smiles=end['graph']['connectivity_smiles'],
                    unique_visited_smiles=len(set(smiles)), final_graph_changed=smiles[-1] != smiles[0])
            endpoints.append(row)
    counts = {}
    for rows in saved['transitions']:
        for row in rows:
            stats = counts.setdefault(row['kind'], dict(attempted=0, scored=0, accepted=0))
            stats['attempted'] += 1
            stats['scored'] += int(row['valid'])
            stats['accepted'] += int(row['accepted'])
    return dict(endpoints=endpoints, transition_counts=counts,
        all_caps_reached=all(saved['query_cap_reached']), raw_queries_per_parent=saved['final_queries_per_parent'])


def audit_ratios(saved, target, decoder, shared):
    """All joint coordinate densities by independent quadrature; all query ledgers."""
    checked = 0
    counts = [2]*len(saved['parent_ids'])
    numbers = torch.tensor(target.numbers)
    electronic = torch.tensor([target.condition['charge'], target.condition['spin_multiplicity'], target.kT], dtype=torch.float64)
    # Reconstruct every state energy directly from the paired raw observations.
    potentials = []
    for state in target.states:
        query = saved['query_trace'][state['query_batch']]; j = state['query_row']
        raw = .5*(float(query['raw_energy_eV'][j])+float(query['inverted_energy_eV'][j]))
        value = raw+target.restraint/2*float(state['positions'].square().sum())
        assert abs(value-float(state['potential_eV'])) < 1e-7
        potentials.append(value)
    for step, rows in enumerate(saved['transitions']):
        for index, row in zip(saved['rounds'][step]['active_indices'], rows):
            counts[index] += 2*int(row['valid'])
            old_id = saved['history_state_ids'][step][index]
            assert row['old_state_id'] == old_id
            assert saved['history_state_ids'][step+1][index] == (row['new_state_id'] if row['accepted'] else old_id)
            if row['kind'] != 'joint_exchange' or not row['valid']:
                continue
            old = target.states[old_id]; new = target.states[row['new_state_id']]
            if decoder.startswith('arc_'):
                qf, qr = independent_arc_q(row['forward']), independent_arc_q(row['reverse'])
            else:
                qf = independent_log_q(old['positions'], new['positions'], old['graph']['bond_orders'], numbers,
                    electronic, target.radii, row['action'], row['order'], 'site', None, shared, row['forward'])
                qr = independent_log_q(new['positions'], old['positions'], new['graph']['bond_orders'], numbers,
                    electronic, target.radii, row['inverse_action'], row['order'], 'site', None, shared, row['reverse'])
            nf = len(distinct_anchor_actions(numbers, old['graph']['bond_orders']))
            nr = len(distinct_anchor_actions(numbers, new['graph']['bond_orders']))
            ratio = -(potentials[new['state_id']]-potentials[old_id])/target.kT+qr-qf+math.log(nf/nr)
            assert abs(ratio-row['log_acceptance_ratio']) < 1e-7
            assert row['accepted'] == (row['log_uniform'] < min(0., ratio))
            checked += 1
        assert counts == saved['query_count_history'][step+1]
    assert counts == saved['final_queries_per_parent']
    return checked


def verify_recovery_prefix(run, source):
    if 'recovery' not in source: return False
    info = source['recovery']; previous = Path(info['source'])
    assert sha(previous/'results.json') == info['source_results_sha256']
    old = json.loads((previous/'results.json').read_text())
    assert old['failure'] == info['source_failure'] == 'ValueError: Orthogonal circle frame required'
    assert old['new_raw_queries'] == old['requested_raw_queries'] == info['previous_physical_calls']
    partial_path = previous/'learned_work_s0_failed_trace.pt'
    assert sha(partial_path) == info['source_failed_trace_sha256']
    partial = torch.load(partial_path, map_location='cpu', weights_only=False)
    row = next(r for r in source['arms'] if r['name'] == 'learned_work_s0')
    continued = torch.load(run/row['file'], map_location='cpu', weights_only=False)
    equal(continued['query_trace'][:len(partial['query_trace'])], partial['query_trace'])
    cached = partial['query_trace'][-1]['raw_queries_after']-partial['raw_query_offset']
    assert cached == info['cached_partial_calls'] == info['expected_cached_partial_calls']
    assert info['all_cached_requests_replayed']
    for old_row,new_row in zip(old['arms'],source['arms']):
        assert old_row['name'] == new_row['name'] and old_row['trace_sha256'] == new_row['trace_sha256']
    assert source['new_raw_queries'] == info['previous_physical_calls']+info['additional_physical_calls']
    assert info['additional_physical_calls'] == info['additional_requested_calls']
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['evaluate', 'audit'], required=True)
    for name in ['project', 'out']: p.add_argument('--'+name, type=Path, required=True)
    for name in ['oracle-python', 'oracle-checkpoint', 'run']: p.add_argument('--'+name, type=Path)
    p.add_argument('--index', type=int, required=True)
    p.add_argument('--protocol', type=Path)
    args = p.parse_args(); root = Path(__file__).resolve().parents[2]
    pp, protocol, header, warm, positions, ids, physical = inputs(root, args.project, args.index, args.protocol)
    shared = protocol['shared']
    args.out.mkdir(parents=True, exist_ok=True); output = args.out/'results.json'
    if output.exists(): raise FileExistsError(output)
    report = dict(complete=False, phase=args.phase, index=args.index, condition=header['condition'],
        protocol_sha256=sha(pp), parent_ids=ids, arms=[], new_raw_queries=0, scientific_submission_ready=False,
        scope=protocol.get('scope', 'FIT parents only, full-chain fixed-query readouts; no new learned model or equilibrium claim.'))
    write(output, report)
    oracle = None
    started = time.monotonic()
    try:
        if args.phase == 'evaluate':
            assert sha(args.oracle_checkpoint) == physical['raw_oracle_sha256']
            oracle = EnergyOracle(args.oracle_python, root/'scripts/research/oracle_worker.py', args.oracle_checkpoint,
                numbers=header['condition']['numbers'], charge=header['condition']['charge'],
                spin_multiplicity=header['condition']['spin_multiplicity'], device='cuda', batch_size=32)
            assert oracle.handshake['base_precision_dtype'] == 'torch.float32' and not oracle.handshake['tf32']
            report['oracle_runtime'] = oracle.handshake
            report['oracle_startup_seconds'] = time.monotonic()-started
        else:
            source = json.loads((args.run/'results.json').read_text())
            assert source['complete'] and source['phase'] == 'evaluate' and source['protocol_sha256'] == sha(pp)
            assert source['parent_ids'] == ids and source['condition'] == header['condition']
            assert len(source['arms']) == len(protocol['methods'])*len(protocol['replicas'])
            report['producer_results_sha256'] = sha(args.run/'results.json')
            if 'recovery' in source: report['recovery_prefix_verified'] = verify_recovery_prefix(args.run, source)
        for replica in protocol['replicas']:
            for method in protocol['methods']:
                name = f'{method}_s{replica}'
                model_tick = time.monotonic()
                model, model_hash = load_scalar_model(args.project, protocol, method, replica)
                model_loading_seconds = time.monotonic()-model_tick
                if args.phase == 'audit':
                    expected = source['arms'][len(report['arms'])]
                    assert expected['name'] == name
                    if 'model_methods' in protocol: assert expected['model_sha256'] == model_hash
                    path = args.run/expected['file']; assert sha(path) == expected['trace_sha256']
                    saved = torch.load(path, map_location='cpu', weights_only=False)
                    oracle = ReplayOracle(saved['query_trace']); oracle.evaluated = saved['raw_query_offset']
                offset = oracle.evaluated
                target = ChemicalTarget(oracle, header['condition'], physical['kT_eV'], physical['restraint_eV_A2'])
                progress = dict(raw_query_offset=offset)
                tick = time.monotonic()
                try:
                    run_budget(target, positions, ids, model, method, replica, args.index, protocol, shared, progress)
                except Exception:
                    progress.update(states=target.states, query_trace=target.query_trace)
                    torch.save(progress, args.out/f'{name}_failed_trace.pt')
                    raise
                progress.update(states=target.states, query_trace=target.query_trace)
                cost = oracle.evaluated-offset
                assert cost == sum(progress['final_queries_per_parent']) <= len(ids)*protocol['query_caps'][method]
                stats = diagnostics(progress, protocol['readout_raw_queries'])
                if args.phase == 'evaluate':
                    path = args.out/f'{name}_trace.pt'; torch.save(progress, path)
                    row = dict(name=name, method=method, replica=replica, file=path.name, trace_sha256=sha(path),
                        raw_query_offset=offset, raw_queries=cost, sampling_seconds=time.monotonic()-tick, **stats)
                    if 'model_methods' in protocol: row.update(model_sha256=model_hash, model_loading_seconds=model_loading_seconds)
                else:
                    equal(progress, saved); equal(stats, {k: expected[k] for k in stats})
                    assert oracle.index == len(oracle.queries) and cost == expected['raw_queries']
                    assert offset == expected['raw_query_offset'] == sum(r['raw_queries'] for r in report['arms'])
                    checked = audit_ratios(saved, target, protocol['joint_decoders'][method], shared)
                    row = dict(name=name, raw_queries=cost, trace_sha256=sha(path), full_replay=True,
                        independent_joint_MH_checks=checked, independent_per_parent_ledger=True)
                report['arms'].append(row)
                report['new_raw_queries'] = oracle.evaluated if args.phase == 'evaluate' else 0
                write(output, report)
                print(json.dumps(dict(index=args.index, arm=name, phase=args.phase, raw_queries=cost,
                    caps_reached=stats['all_caps_reached'])), flush=True)
        total = sum(r['raw_queries'] for r in report['arms'])
        maximum = len(protocol['replicas'])*sum(sum(parent_query_caps(protocol, method, args.index, ids)) for method in protocol['methods'])
        assert total <= maximum
        if args.phase == 'evaluate':
            assert oracle.evaluated == oracle.requested_evaluations == total
            report['requested_raw_queries'] = oracle.requested_evaluations
        else:
            assert source['new_raw_queries'] == source['requested_raw_queries'] == total
            report.update(full_producer_replay=True, all_random_streams_replayed=True, raw_queries_in_producer=total)
        report.update(complete=True, elapsed_seconds=time.monotonic()-started)
        write(output, report)
    except Exception as exc:
        report.update(failure=f'{type(exc).__name__}: {exc}', elapsed_seconds=time.monotonic()-started)
        if args.phase == 'evaluate' and oracle is not None:
            report.update(new_raw_queries=oracle.evaluated, requested_raw_queries=oracle.requested_evaluations)
        write(output, report)
        raise
    finally:
        if args.phase == 'evaluate' and oracle is not None: oracle.close()


if __name__ == '__main__': main()
