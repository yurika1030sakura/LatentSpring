#!/usr/bin/env python3
"""Frozen normalized site-mixture evaluation with physical and initialization controls."""
import argparse
import json
from pathlib import Path
import time
import torch
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.terminal_rotation import uniform_internal_transition
from cfm_mol.normalized_site_guide import NormalizedSiteGuide
from cfm_mol.joint_chemical_geometry import joint_chemical_transition
from scripts.research.evaluate_chemical_policy import sha, write


def load_model(directory, method, replica, protocol, training_hash):
    if method == 'site':
        return None, None
    name = 'vector' if method == 'learned_vector' else 'mixture'
    directory = directory/f'{name}_s{replica}'
    metadata = json.loads((directory/'results.json').read_text())
    if (not metadata['complete'] or metadata['training_sha256'] != training_hash
            or metadata['protocol_sha256'] != protocol['normalized_training_protocol_sha256']
            or sha(directory/'model.pt') != metadata['checkpoint_sha256']
            or metadata['checkpoint_sha256'] != protocol['frozen_model_sha256'][name][replica]
            or metadata['development_coordinates_loaded'] or metadata['reference_coordinates_loaded']):
        raise ValueError('Normalized guide provenance failed')
    checkpoint = torch.load(directory/'model.pt', map_location='cpu', weights_only=False)
    model = NormalizedSiteGuide(**checkpoint['configuration']).double()
    field = 'initial_state_dict' if method == 'initial_mixture' else 'state_dict'
    model.load_state_dict(checkpoint[field])
    return model.eval(), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ['table', 'models', 'out', 'oracle-python', 'oracle-checkpoint']:
        parser.add_argument('--'+arg, type=Path, required=True)
    parser.add_argument('--method', choices=['site', 'initial_mixture', 'learned_vector', 'learned_mixture'], required=True)
    parser.add_argument('--replica', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    pp = root/'research/evidence/normalized_site_evaluation_protocol_v1.json'
    protocol = json.loads(pp.read_text())
    training_protocol = root/'research/evidence/chemical_policy_protocol_v1.json'
    physical_path = root/'research/evidence/parity_training_protocol_v1.json'
    physical = json.loads(physical_path.read_text())
    if sha(training_protocol) != protocol['training_protocol_sha256'] or sha(physical_path) != protocol['physical_protocol_sha256']:
        raise ValueError('Changed upstream protocol')
    if sha(args.oracle_checkpoint) != physical['raw_oracle_sha256']:
        raise ValueError('Changed oracle weights')
    header = json.loads((args.table/'results.json').read_text())
    if (not header['complete'] or sha(args.table/'results.json') != protocol['table_results_sha256']
            or header['protocol_sha256'] != sha(training_protocol)
            or sha(args.table/'development.pt') != header['artifacts']['development']):
        raise ValueError('Development provenance failed')
    data = torch.load(args.table/'development.pt', map_location='cpu', weights_only=False)
    if data['stream'] != 'development' or data['kT_eV'] != physical['kT_eV'] or data['restraint_eV_A2'] != physical['restraint_eV_A2']:
        raise ValueError('Changed stream or target')
    model, trained = load_model(args.models, args.method, args.replica, protocol, header['artifacts']['training'])
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out/'results.json'
    if output.exists():
        raise FileExistsError(output)
    learned = args.method.startswith('learned_')
    training_cost = trained['inherited_training_raw_queries'] if learned else 0
    source_cost = header['inherited_source_raw_queries'] if learned else 512
    warm_cost = header['streams']['development']['raw_queries']
    report = dict(complete=False, method=args.method, replica=args.replica,
        evaluation_protocol_sha256=sha(pp), development_sha256=sha(args.table/'development.pt'),
        condition=data['condition'], development_parent_ids=data['source_parent_ids'],
        policy_sha256=trained['checkpoint_sha256'] if trained else None,
        weights_kind='initial' if args.method == 'initial_mixture' else 'final' if learned else 'physical',
        inherited_training_raw_queries=training_cost, inherited_source_raw_queries=source_cost,
        inherited_development_warm_raw_queries=warm_cost, source_generation_denominators=header['source_generation_denominators'],
        reference_coordinates_loaded=False, development_coordinates_used_for_training=False,
        scientific_submission_ready=False, history=[])
    write(output, report)
    generator = torch.Generator().manual_seed(protocol['evaluation_seeds'][args.replica])
    scale_generator = torch.Generator().manual_seed(protocol['scale_seed']+args.replica)
    target = oracle = None
    transitions, history_ids, scales_history = [], [], []
    start = time.perf_counter()
    try:
        oracle = EnergyOracle(args.oracle_python, root/'scripts/research/oracle_worker.py', args.oracle_checkpoint,
            numbers=data['condition']['numbers'], charge=data['condition']['charge'],
            spin_multiplicity=data['condition']['spin_multiplicity'], device='cuda', batch_size=32)
        if oracle.handshake.get('tf32') or oracle.handshake.get('base_precision_dtype') != 'torch.float32':
            raise ValueError('Unqualified physical oracle precision')
        report['oracle_runtime'] = oracle.handshake
        target = ChemicalTarget(oracle, data['condition'], data['kT_eV'], data['restraint_eV_A2'])
        initial = [data['states'][i] for i in data['warm_state_ids']]
        states = target.evaluate([target.coordinate_state(s['positions']) for s in initial], phase='initial')
        for old, new in zip(initial, states):
            torch.testing.assert_close(old['energy_eV'], new['energy_eV'], atol=1e-4, rtol=0)
            torch.testing.assert_close(old['force_eV_A'], new['force_eV_A'], atol=1e-4, rtol=0)
        for step in range(protocol['steps']+1):
            row = dict(step=step, energy_eV=[float(s['energy_eV']) for s in states],
                potential_eV=[float(s['potential_eV']) for s in states], smiles=[s['graph']['connectivity_smiles'] for s in states],
                new_raw_queries=oracle.evaluated, total_raw_queries=oracle.evaluated+training_cost+source_cost+warm_cost)
            report['history'].append(row)
            history_ids.append([s['state_id'] for s in states])
            if step % 32 == 0 or step == protocol['steps']:
                print(json.dumps(row), flush=True)
                write(output, report)
            if step == protocol['steps']:
                break
            choices = torch.randint(len(protocol['local_scales']), (len(states),), generator=scale_generator)
            scales_history.append(choices.tolist())
            scales = torch.tensor(protocol['local_scales'], dtype=torch.float64)[choices]*target.kT**.5
            kind = protocol['schedule'][step % len(protocol['schedule'])]
            phase = f'evaluation_{step}'
            if kind == 'local':
                states, rows = target.transition(states, policy=None, generator=generator, proposal_std=scales, phase=phase, local_only=True)
                for record in rows:
                    record['kind'] = kind
            elif kind == 'force_rotation':
                states, rows = uniform_internal_transition(target, states, kind=kind, generator=generator, phase=phase)
            else:
                states, rows = joint_chemical_transition(target, states, kind='site' if args.method == 'site' else 'defensive_site', generator=generator,
                    phase=phase, model=model, radial_width=protocol['radial_width'], site_concentration=protocol['site_concentration'])
            transitions.extend(rows)
            if oracle.evaluated > protocol['maximum_raw_queries_per_arm']:
                raise RuntimeError('Prospective query cap exceeded')
        artifact = dict(states=target.states, query_trace=target.query_trace, transitions=transitions,
            history_state_ids=history_ids, generator_state=generator.get_state(),
            scale_choice_history=scales_history, scale_generator_state=scale_generator.get_state())
        torch.save(artifact, args.out/'trace.pt')
        report.update(complete=True, new_raw_queries=oracle.evaluated, requested_raw_queries=oracle.requested_evaluations,
            seconds=time.perf_counter()-start, trace_sha256=sha(args.out/'trace.pt'),
            proposals=len(transitions), invalid_proposals=sum(not r['valid'] for r in transitions),
            accepted_proposals=sum(r['accepted'] for r in transitions),
            limitation='One repeatedly inspected development composition, four generated parents, two seeds; first hits and energies are not equilibrium validation or independent generalization.')
        write(output, report)
    except Exception as exc:
        if target is not None:
            torch.save(dict(states=target.states, query_trace=target.query_trace, transitions=transitions,
                history_state_ids=history_ids, generator_state=generator.get_state(),
                scale_choice_history=scales_history, scale_generator_state=scale_generator.get_state()), args.out/'failed_trace.pt')
        report.update(failure=f'{type(exc).__name__}: {exc}', new_raw_queries=oracle.evaluated if oracle else 0,
                      requested_raw_queries=oracle.requested_evaluations if oracle else 0)
        write(output, report)
        raise
    finally:
        if oracle is not None:
            oracle.close()


if __name__ == '__main__':
    main()
