#!/usr/bin/env python3
"""Frozen EACF directional-MH comparison on the existing matched development starts."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.eacf_transport import EACFTransport
from cfm_mol.directional_flow_mh import directional_flow_transition
from cfm_mol.terminal_rotation import uniform_internal_transition
from scripts.research.evaluate_chemical_policy import sha, write


class RecordedTransport:
    def __init__(self, transport):
        self.transport = transport
        self.aux_scale = transport.aux_scale
        self.traces = []

    def transform(self, x, a, d):
        row = dict(positions=x.clone(), auxiliary=a.clone(), directions=d.clone())
        self.traces.append(row)
        response = self.transport.transform(x, a, d)
        row['response'] = response
        return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['flows', 'table', 'controls', 'interfaces', 'out', 'transport-python', 'upstream', 'oracle-python', 'oracle-checkpoint']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--profile', choices=['published', 'compact'], required=True)
    parser.add_argument('--replica', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    pp = root/'research/evidence/eacf_directional_protocol_v1.json'
    protocol = json.loads(pp.read_text())
    shared = root/'research/evidence/normalized_site_evaluation_protocol_v1.json'
    assert sha(shared) == protocol['shared_evaluation_protocol_sha256']
    shared = json.loads(shared.read_text())
    physical_path = root/'research/evidence/parity_training_protocol_v1.json'
    assert sha(physical_path) == shared['physical_protocol_sha256']
    physical = json.loads(physical_path.read_text())
    assert sha(args.oracle_checkpoint) == physical['raw_oracle_sha256']
    stem = 'eacf_joint' if args.profile == 'published' else 'eacf_compact'
    flow_dir = args.flows/f'{stem}_condition_00_s{args.replica}_v1'
    flow_report = json.loads((flow_dir/'results.json').read_text())
    assert flow_report['complete'] and not flow_report['engineering_only']
    assert sha(flow_dir/'adapter.pkl') == protocol['checkpoint_sha256'][args.profile][args.replica]
    check_path = args.interfaces/f'{args.profile}_s{args.replica}/results.json'
    check = json.loads(check_path.read_text())
    assert check['complete'] and sha(check_path) == protocol['interface_results_sha256'][args.profile][args.replica]
    assert check['checkpoint_sha256'] == sha(flow_dir/'adapter.pkl')
    jac_path = args.interfaces/f'{args.profile}_s0/results.json'
    jac = json.loads(jac_path.read_text())
    assert jac['complete'] and jac['full_jacobian_checked'] and jac['intrinsic_log_volume_error'] < 1e-6
    header = json.loads((args.table/'results.json').read_text())
    assert header['complete'] and sha(args.table/'results.json') == shared['table_results_sha256']
    assert sha(args.table/'development.pt') == header['artifacts']['development']
    data = torch.load(args.table/'development.pt', map_location='cpu', weights_only=False)
    assert data['stream'] == 'development' and data['condition'] == flow_report['condition']
    for method in ['site', 'learned_vector']:
        reference_path = args.controls/f'{method}_s{args.replica}/results.json'
        reference = json.loads(reference_path.read_text())
        assert reference['complete'] and sha(reference_path) == protocol['reused_control_sha256'][method][args.replica]
        assert reference['condition'] == data['condition'] and reference['development_parent_ids'] == data['source_parent_ids']
        assert reference['evaluation_protocol_sha256'] == protocol['shared_evaluation_protocol_sha256']
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out/'results.json'
    if output.exists():
        raise FileExistsError(output)
    training_calls = flow_report['steps']*flow_report['batch']
    assert training_calls == protocol['inherited_training_raw_queries']
    previous_validation = flow_report['oracle_evaluations']-training_calls
    warm_cost = header['streams']['development']['raw_queries']
    source_cost = header['inherited_source_raw_queries']
    report = dict(complete=False, method='eacf_'+args.profile, replica=args.replica, protocol_sha256=sha(pp),
        shared_evaluation_protocol_sha256=protocol['shared_evaluation_protocol_sha256'], condition=data['condition'],
        development_parent_ids=data['source_parent_ids'], development_sha256=sha(args.table/'development.pt'),
        checkpoint_sha256=sha(flow_dir/'adapter.pkl'), interface_results_sha256=sha(check_path),
        inherited_training_raw_queries=training_calls, prior_generator_validation_raw_queries=previous_validation,
        inherited_source_raw_queries=source_cost, inherited_development_warm_raw_queries=warm_cost,
        scientific_submission_ready=False, history=[])
    write(output, report)
    generator = torch.Generator().manual_seed(shared['evaluation_seeds'][args.replica])
    scale_generator = torch.Generator().manual_seed(shared['scale_seed']+args.replica)
    transport = oracle = target = recorded = None
    transitions, history_ids, choices_history = [], [], []
    started = time.perf_counter()
    try:
        transport = EACFTransport(args.transport_python, root/'scripts/research/eacf_transport_worker.py', flow_dir,
                                  args.upstream, log_path=args.out/'transport.log')
        assert transport.handshake['condition'] == data['condition']
        assert transport.handshake['checkpoint_sha256'] == report['checkpoint_sha256']
        report['transport_runtime'] = transport.handshake
        recorded = RecordedTransport(transport)
        oracle = EnergyOracle(args.oracle_python, root/'scripts/research/oracle_worker.py', args.oracle_checkpoint,
            numbers=data['condition']['numbers'], charge=data['condition']['charge'],
            spin_multiplicity=data['condition']['spin_multiplicity'], device='cuda', batch_size=32)
        assert not oracle.handshake.get('tf32') and oracle.handshake.get('base_precision_dtype') == 'torch.float32'
        report['oracle_runtime'] = oracle.handshake
        target = ChemicalTarget(oracle, data['condition'], data['kT_eV'], data['restraint_eV_A2'])
        initial = [data['states'][i] for i in data['warm_state_ids']]
        states = target.evaluate([target.coordinate_state(s['positions']) for s in initial], phase='initial')
        for before, after in zip(initial, states):
            torch.testing.assert_close(before['energy_eV'], after['energy_eV'], atol=1e-4, rtol=0)
            torch.testing.assert_close(before['force_eV_A'], after['force_eV_A'], atol=1e-4, rtol=0)
        for step in range(shared['steps']+1):
            row = dict(step=step, energy_eV=[float(s['energy_eV']) for s in states],
                potential_eV=[float(s['potential_eV']) for s in states], smiles=[s['graph']['connectivity_smiles'] for s in states],
                new_raw_queries=oracle.evaluated, total_raw_queries=oracle.evaluated+training_calls+source_cost+warm_cost,
                total_including_prior_validation=oracle.evaluated+training_calls+source_cost+warm_cost+previous_validation)
            report['history'].append(row)
            history_ids.append([s['state_id'] for s in states])
            if step % 32 == 0 or step == shared['steps']:
                print(json.dumps(row), flush=True)
                write(output, report)
            if step == shared['steps']:
                break
            choices = torch.randint(len(shared['local_scales']), (len(states),), generator=scale_generator)
            choices_history.append(choices.tolist())
            scales = torch.tensor(shared['local_scales'], dtype=torch.float64)[choices]*target.kT**.5
            kind, phase = shared['schedule'][step % 4], f'evaluation_{step}'
            if kind == 'local':
                states, rows = target.transition(states, policy=None, generator=generator, proposal_std=scales, phase=phase, local_only=True)
                for record in rows:
                    record['kind'] = 'local'
            elif kind == 'force_rotation':
                states, rows = uniform_internal_transition(target, states, kind=kind, generator=generator, phase=phase)
            else:
                states, rows = directional_flow_transition(target, states, recorded, generator=generator, phase=phase)
            transitions.extend(rows)
        assert oracle.evaluated <= shared['maximum_raw_queries_per_arm']
        artifact = dict(states=target.states, query_trace=target.query_trace, transitions=transitions,
            history_state_ids=history_ids, scale_choice_history=choices_history,
            generator_state=generator.get_state(), scale_generator_state=scale_generator.get_state(),
            transport_traces=recorded.traces)
        torch.save(artifact, args.out/'trace.pt')
        arrays = {key: np.stack([row[key].numpy() for row in recorded.traces]) for key in ['positions', 'auxiliary', 'directions']}
        arrays.update({key: np.stack([row['response'][source].numpy() for row in recorded.traces])
                       for key, source in [('mapped_positions', 'positions'), ('mapped_auxiliary', 'auxiliary'), ('log_volume', 'log_volume')]})
        np.savez(args.out/'flow_trace.npz', **arrays)
        report.update(complete=True, new_raw_queries=oracle.evaluated, requested_raw_queries=oracle.requested_evaluations,
            trace_sha256=sha(args.out/'trace.pt'), flow_trace_sha256=sha(args.out/'flow_trace.npz'), seconds=time.perf_counter()-started,
            requested_transformations=transport.requested_transformations, completed_transformations=transport.completed_transformations,
            transport_seconds=transport.seconds, first_transport_seconds=recorded.traces[0]['response']['seconds'],
            previous_validation_excluded_from_standard_training_inference_total=True,
            limitation='One repeatedly inspected composition and four starts. Same schedule/maximum-query cap is not equal actual cost or an equilibrium certificate. Full and compact architectures retain their original training.')
        write(output, report)
    except Exception as exc:
        if target is not None:
            torch.save(dict(states=target.states, query_trace=target.query_trace, transitions=transitions,
                transport_traces=recorded.traces if recorded else [], generator_state=generator.get_state()), args.out/'failed_trace.pt')
        report.update(failure=f'{type(exc).__name__}: {exc}', new_raw_queries=oracle.evaluated if oracle else 0,
            requested_raw_queries=oracle.requested_evaluations if oracle else 0,
            requested_transformations=transport.requested_transformations if transport else 0,
            completed_transformations=transport.completed_transformations if transport else 0)
        write(output, report)
        raise
    finally:
        if oracle is not None:
            oracle.close()
        if transport is not None:
            transport.close()


if __name__ == '__main__':
    main()
