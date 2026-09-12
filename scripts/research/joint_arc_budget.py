#!/usr/bin/env python3
"""Bounded TRAINING-only full-chain comparison of legacy and constrained proposals."""
import argparse
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
from scripts.research.fresh_reuse import run_budget


def inputs(root, project, index):
    pp = root/'research/evidence/joint_arc_budget_protocol_v1.json'
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
    assert ids == split['fit_parent_ids'] and len(ids) == 12
    assert not set(ids) & set(split['withheld_parent_ids'])
    indices = [warm['parent_ids'].index(i) for i in ids]
    positions = warm['warm_positions'][indices]
    physical_path = root/'research/evidence/parity_training_protocol_v1.json'
    assert sha(physical_path) == protocol['physical_protocol_sha256']
    return pp, protocol, header, warm, positions, ids, json.loads(physical_path.read_text())


def diagnostics(saved, readouts):
    endpoints = []
    for i, parent in enumerate(saved['parent_ids']):
        first = saved['states'][saved['history_state_ids'][0][i]]
        for cap in readouts:
            hit = next((j for j, counts in enumerate(saved['query_count_history']) if counts[i] == cap), None)
            row = dict(parent=parent, raw_queries=cap, reached=hit is not None)
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['evaluate', 'audit'], required=True)
    for name in ['project', 'out']: p.add_argument('--'+name, type=Path, required=True)
    for name in ['oracle-python', 'oracle-checkpoint', 'run']: p.add_argument('--'+name, type=Path)
    p.add_argument('--index', type=int, required=True)
    args = p.parse_args(); root = Path(__file__).resolve().parents[2]
    pp, protocol, header, warm, positions, ids, physical = inputs(root, args.project, args.index)
    shared = protocol['shared']
    args.out.mkdir(parents=True, exist_ok=True); output = args.out/'results.json'
    if output.exists(): raise FileExistsError(output)
    report = dict(complete=False, phase=args.phase, index=args.index, condition=header['condition'],
        protocol_sha256=sha(pp), parent_ids=ids, arms=[], new_raw_queries=0, scientific_submission_ready=False,
        scope='FIT parents only, full-chain fixed-query readouts; no new learned model or equilibrium claim.')
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
        for replica in protocol['replicas']:
            for method in protocol['methods']:
                name = f'{method}_s{replica}'
                if args.phase == 'audit':
                    expected = source['arms'][len(report['arms'])]
                    assert expected['name'] == name
                    path = args.run/expected['file']; assert sha(path) == expected['trace_sha256']
                    saved = torch.load(path, map_location='cpu', weights_only=False)
                    oracle = ReplayOracle(saved['query_trace']); oracle.evaluated = saved['raw_query_offset']
                offset = oracle.evaluated
                target = ChemicalTarget(oracle, header['condition'], physical['kT_eV'], physical['restraint_eV_A2'])
                progress = dict(raw_query_offset=offset)
                tick = time.monotonic()
                try:
                    run_budget(target, positions, ids, None, method, replica, args.index, protocol, shared, progress)
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
        assert total <= protocol['maximum_total_new_raw_queries']//len(protocol['condition_indices'])
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
