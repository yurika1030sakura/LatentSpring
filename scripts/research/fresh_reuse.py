#!/usr/bin/env python3
"""Frozen fresh-parent preparation and per-parent physical-query-budget pilots."""
import argparse
import json
from pathlib import Path
import time
import torch
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.parity_refinement import randomize_inversion
from cfm_mol.terminal_rotation import uniform_internal_transition
from cfm_mol.joint_chemical_geometry import joint_chemical_transition
from scripts.research.evaluate_chemical_policy import sha, write
from scripts.research.evaluate_normalized_site import load_model


def update_query_counts(counts, indices, rows, oracle_delta, cap):
    """Charge all supported proposals, including MH rejections, to their parent."""
    if len(indices) != len(rows) or len(set(indices)) != len(indices):
        raise ValueError('One transition per distinct active parent required')
    charged = 0
    for index, row in zip(indices, rows):
        if row['accepted'] and not row['valid']:
            raise ValueError('Unsupported proposal cannot be accepted')
        cost = 2*int(row['valid'])
        counts[index] += cost
        charged += cost
        if counts[index] > cap:
            raise ValueError('Per-parent physical-query cap exceeded')
    if charged != oracle_delta:
        raise ValueError('Paired oracle calls differ from parent ledger')


def run_budget(target, positions, parent_ids, model, method, replica, batch_index, protocol, shared, progress):
    """Finite-cost experiment; query-index stopping is NOT an equilibrium claim."""
    cap = protocol['query_caps'][method]
    if cap < 2 or cap % 2 or len(positions) != len(parent_ids):
        raise ValueError('Matched starts and an even cap including initialization required')
    rng = torch.Generator().manual_seed(protocol['evaluation_seeds'][replica]+100003*batch_index)
    srng = torch.Generator().manual_seed(protocol['scale_seeds'][replica]+100003*batch_index)
    states = target.evaluate([target.coordinate_state(x) for x in positions], phase='initial')
    counts = [2]*len(states)
    progress.update(parent_ids=list(parent_ids), transitions=[], rounds=[],
        history_state_ids=[[s['state_id'] for s in states]], query_count_history=[list(counts)])
    start_count = target.oracle.evaluated-2*len(states)
    for step in range(protocol['maximum_microsteps']):
        indices = [i for i, count in enumerate(counts) if count < cap]
        if not indices:
            break
        current = [states[i] for i in indices]
        choices = torch.randint(len(shared['local_scales']), (len(indices),), generator=srng)
        scales = torch.tensor(shared['local_scales'], dtype=torch.float64)[choices]*target.kT**.5
        kind = shared['schedule'][step % len(shared['schedule'])]
        phase = f'evaluation_{step}'
        before = target.oracle.evaluated
        if kind == 'local':
            proposed, rows = target.transition(current, policy=None, generator=rng, proposal_std=scales, phase=phase, local_only=True)
            for row in rows:
                row['kind'] = kind
        elif kind == 'force_rotation':
            proposed, rows = uniform_internal_transition(target, current, kind=kind, generator=rng, phase=phase)
        else:
            proposed, rows = joint_chemical_transition(target, current,
                kind='site' if method == 'site' else 'defensive_site', generator=rng, phase=phase,
                model=model, radial_width=shared['radial_width'], site_concentration=shared['site_concentration'])
        update_query_counts(counts, indices, rows, target.oracle.evaluated-before, cap)
        for index, new in zip(indices, proposed):
            states[index] = new
        progress['rounds'].append(dict(step=step, active_indices=indices, scale_choices=choices.tolist(), kind=kind))
        progress['transitions'].append(rows)
        progress['history_state_ids'].append([s['state_id'] for s in states])
        progress['query_count_history'].append(list(counts))
        assert sum(counts) == target.oracle.evaluated-start_count
    progress.update(generator_state=rng.get_state(), scale_generator_state=srng.get_state(),
        final_queries_per_parent=counts, query_cap_reached=[q == cap for q in counts])
    return states


def prepare(target, data, source_audit, protocol, progress):
    rng = torch.Generator().manual_seed(protocol['warm_seed'])
    positions, signs = randomize_inversion(data['positions'], generator=rng)
    ids = source_audit['supported_parent_ids'][:protocol['maximum_parents']]
    if len(ids) < protocol['minimum_parents']:
        raise ValueError('Too few fresh supported parents for the frozen pilot')
    states = [target.coordinate_state(positions[i]) for i in ids]
    target.evaluate(states, phase='initial')
    counts = [2]*len(ids)
    progress.update(parent_ids=ids, inversion_signs=signs, transitions=[],
        history_state_ids=[[s['state_id'] for s in states]], query_count_history=[list(counts)])
    for step in range(protocol['warm_steps']):
        before = target.oracle.evaluated
        states, rows = target.transition(states, policy=None, generator=rng,
            proposal_std=protocol['warm_local_scale']*target.kT**.5, phase=f'warm_{step}', local_only=True)
        update_query_counts(counts, list(range(len(ids))), rows, target.oracle.evaluated-before,
            2*(protocol['warm_steps']+1))
        progress['transitions'].append(rows)
        progress['history_state_ids'].append([s['state_id'] for s in states])
        progress['query_count_history'].append(list(counts))
    progress.update(generator_state=rng.get_state(), final_queries_per_parent=counts,
        warm_positions=torch.stack([s['positions'] for s in states]))
    return states


def validate_inputs(root, project, protocol):
    shared_path = root/'research/evidence/normalized_site_evaluation_protocol_v1.json'
    physical_path = root/'research/evidence/parity_training_protocol_v1.json'
    assert sha(shared_path) == protocol['shared_protocol_sha256']
    assert sha(physical_path) == protocol['physical_protocol_sha256']
    source = project/protocol['source_run']
    audit_path = project/protocol['source_audit']
    audit = json.loads(audit_path.read_text())
    if 'derived_from_audit' in audit:
        parent_path = project/audit['derived_from_audit']
        assert sha(parent_path) == audit['derived_from_audit_sha256'] == protocol['parent_source_audit_sha256']
        parent_audit = json.loads(parent_path.read_text())
        assert parent_audit['complete'] and parent_audit['all_chunk_rows_verified'] and parent_audit['all_six_conditions_retained']
        source_row = parent_audit['rows'][audit['condition_index']]
        assert source_row['index'] == audit['condition_index'] == protocol['condition_index']
        for key in ['source_results_sha256','attempted','chemically_supported','validator_errors','supported_parent_ids']:
            assert audit[key] == source_row[key]
        assert audit['source_samples_sha256'] == source_row['samples_sha256']
        assert audit['protocol_sha256'] == parent_audit['protocol_sha256']
    report = json.loads((source/'results.json').read_text())
    submission = json.loads((source/'submission.json').read_text())
    assert submission['source_commit'] == protocol['source_commit']
    assert report['complete'] and audit['complete']
    assert report['protocol_sha256'] == audit['protocol_sha256'] == protocol['source_protocol_sha256']
    assert sha(source/'results.json') == audit['source_results_sha256']
    assert sha(source/'samples.pt') == audit['source_samples_sha256'] == report['samples_sha256']
    assert audit['all_chunk_rows_verified'] and audit['no_seed_overlap_with_original_streams']
    assert audit['attempted'] == protocol['source_attempts'] and audit['physical_queries'] == 0
    data = torch.load(source/'samples.pt', map_location='cpu', weights_only=False)
    assert data['condition'] == report['condition'] and data['stream'] == 'fresh_development'
    shared = json.loads(shared_path.read_text())
    if 'proposal_site_concentration' in protocol:
        concentration = float(protocol['proposal_site_concentration'])
        if not 0 < concentration < float('inf'):
            raise ValueError('Positive finite prospective site concentration required')
        shared['site_concentration'] = concentration
    return data, audit, json.loads(physical_path.read_text()), shared


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['prepare','evaluate'], required=True)
    for name in ['project','out','oracle-python','oracle-checkpoint']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--method', choices=['site','learned_vector'])
    p.add_argument('--replica', type=int, choices=[0,1])
    p.add_argument('--protocol', type=Path)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    pp = args.protocol or root/'research/evidence/fresh_reuse_protocol_v1.json'
    protocol = json.loads(pp.read_text())
    data, source_audit, physical, shared = validate_inputs(root, args.project, protocol)
    assert sha(args.oracle_checkpoint) == physical['raw_oracle_sha256']
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out/'results.json'
    if output.exists():
        raise FileExistsError(output)
    model = metadata = warm = None
    if args.phase == 'evaluate':
        if args.method is None or args.replica is None:
            raise ValueError('Evaluation method and replica required')
        prep = args.project/protocol['preparation_run']
        prep_report = json.loads((prep/'results.json').read_text())
        prep_audit = json.loads((args.project/protocol['preparation_audit']).read_text())
        assert prep_report['complete'] and prep_audit['complete'] and prep_audit['full_producer_replay']
        assert sha(prep/'results.json') == prep_audit['results_sha256']
        assert sha(prep/'trace.pt') == prep_report['trace_sha256'] == prep_audit['trace_sha256']
        assert prep_report['protocol_sha256'] == protocol.get('preparation_protocol_sha256', sha(pp))
        warm = torch.load(prep/'trace.pt', map_location='cpu', weights_only=False)
        assert warm['condition'] == data['condition'] and warm['parent_ids'] == source_audit['supported_parent_ids'][:protocol['maximum_parents']]
        model, metadata = load_model(args.project/'runs/normalized_site_train_v1', args.method, args.replica,
            shared, protocol['training_artifact_sha256'])
    report = dict(complete=False, phase=args.phase, method=args.method, replica=args.replica,
        condition=data['condition'], protocol_sha256=sha(pp), source_attempts=source_audit['attempted'],
        source_supported=source_audit['chemically_supported'], source_validator_errors=source_audit['validator_errors'],
        source_results_sha256=source_audit['source_results_sha256'], source_samples_sha256=source_audit['source_samples_sha256'],
        source_audit_sha256=sha(args.project/protocol['source_audit']),
        policy_sha256=metadata['checkpoint_sha256'] if metadata else None,
        proposal_site_concentration=shared['site_concentration'],
        one_time_training_raw_queries=protocol['one_time_training_raw_queries'] if metadata else 0,
        reference_coordinates_loaded=False, fresh_coordinates_used_for_training=False,
        scientific_submission_ready=False, chunks=[], new_raw_queries=0)
    if warm is not None:
        report.update(preparation_results_sha256=sha(prep/'results.json'),
            preparation_trace_sha256=sha(prep/'trace.pt'), preparation_raw_queries=prep_report['new_raw_queries'])
    write(output, report)
    oracle = target = None
    progress = {}
    started = time.perf_counter()
    try:
        oracle = EnergyOracle(args.oracle_python, root/'scripts/research/oracle_worker.py', args.oracle_checkpoint,
            numbers=data['condition']['numbers'], charge=data['condition']['charge'],
            spin_multiplicity=data['condition']['spin_multiplicity'], device='cuda', batch_size=32)
        assert oracle.handshake.get('base_precision_dtype') == 'torch.float32' and not oracle.handshake.get('tf32')
        report['oracle_runtime'] = oracle.handshake
        if args.phase == 'prepare':
            target = ChemicalTarget(oracle, data['condition'], physical['kT_eV'], physical['restraint_eV_A2'])
            prepare(target, data, source_audit, protocol, progress)
            progress.update(condition=data['condition'], states=target.states, query_trace=target.query_trace)
            torch.save(progress, args.out/'trace.pt')
            report.update(trace_sha256=sha(args.out/'trace.pt'), parent_ids=progress['parent_ids'],
                raw_queries_per_parent=progress['final_queries_per_parent'])
        else:
            parent_ids = warm['parent_ids']
            report['parent_ids'] = parent_ids
            for batch, begin in enumerate(range(0, len(parent_ids), protocol['batch'])):
                end = min(len(parent_ids), begin+protocol['batch'])
                before = oracle.evaluated
                target = ChemicalTarget(oracle, data['condition'], physical['kT_eV'], physical['restraint_eV_A2'])
                progress = dict(condition=data['condition'], raw_query_offset=before)
                t0 = time.perf_counter()
                run_budget(target, warm['warm_positions'][begin:end], parent_ids[begin:end], model,
                    args.method, args.replica, batch, protocol, shared, progress)
                progress.update(states=target.states, query_trace=target.query_trace)
                path = args.out/f'trace_{batch:03d}.pt'
                torch.save(progress, path)
                report['chunks'].append(dict(file=path.name, sha256=sha(path), parent_ids=progress['parent_ids'],
                    raw_queries=oracle.evaluated-before, raw_queries_per_parent=progress['final_queries_per_parent'],
                    query_cap_reached=progress['query_cap_reached'], microsteps=len(progress['rounds']),
                    seconds=time.perf_counter()-t0))
                report.update(new_raw_queries=oracle.evaluated, requested_raw_queries=oracle.requested_evaluations,
                    seconds=time.perf_counter()-started)
                write(output, report)
                print(json.dumps(report['chunks'][-1]), flush=True)
        assert oracle.evaluated == oracle.requested_evaluations
        assert oracle.evaluated <= protocol['maximum_total_new_raw_queries']
        report.update(complete=True, new_raw_queries=oracle.evaluated, requested_raw_queries=oracle.requested_evaluations,
            seconds=time.perf_counter()-started,
            limitation='Fresh random parents of one development composition. Per-query stopping measures finite-cost output, not a stationary ensemble. Unsupported source attempts, capped trajectories and validator failures remain explicit.')
        write(output, report)
    except Exception as exc:
        if target is not None:
            progress.update(states=target.states, query_trace=target.query_trace)
            torch.save(progress, args.out/'failed_trace.pt')
        report.update(failure=f'{type(exc).__name__}: {exc}', new_raw_queries=oracle.evaluated if oracle else 0,
            requested_raw_queries=oracle.requested_evaluations if oracle else 0, seconds=time.perf_counter()-started)
        write(output, report)
        raise
    finally:
        if oracle is not None:
            oracle.close()


if __name__ == '__main__':
    main()
