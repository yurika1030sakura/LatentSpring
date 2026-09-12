#!/usr/bin/env python3
"""Bounded physical fragment pilot on separately selected development compositions."""
import argparse
import json
from pathlib import Path
import time
import torch
from cfm_mol.entropy_source import load_entropy_source
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.terminal_rotation import uniform_internal_transition
from cfm_mol.fragment_sampler import fragment_transition
from scripts.research.evaluate_chemical_policy import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'out', 'oracle-python', 'oracle-checkpoint']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--stage', choices=['warm', 'evaluate'], required=True)
    parser.add_argument('--condition-index', type=int, required=True)
    parser.add_argument('--method', choices=['local_only', 'all_singletons', 'fragments4'])
    parser.add_argument('--replica', type=int, choices=[0, 1], default=0)
    parser.add_argument('--warm', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    pp = root/'research/evidence/fragment_physical_protocol_v1.json'
    protocol = json.loads(pp.read_text())
    assert args.condition_index in protocol['condition_indices']
    physical_path = root/'research/evidence/parity_training_protocol_v1.json'
    physical = json.loads(physical_path.read_text())
    assert sha(physical_path) == protocol['physical_protocol_sha256']
    assert sha(args.oracle_checkpoint) == physical['raw_oracle_sha256']
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out/'results.json'
    if output.exists():
        raise FileExistsError(output)
    if args.stage == 'warm':
        source = load_entropy_source(args.source, args.condition_index,
            protocol_path=root/'research/evidence/species_breadth_source_protocol_v2.json',
            manifest_path=root/'research/evidence/development_panel_v1.json')
        census_path = root/'research/evidence/chemical_source_panel_audit_v2.json'
        assert sha(census_path) == protocol['census_sha256']
        census = json.loads(census_path.read_text())['conditions'][args.condition_index]
        parents = [r['parent_id'] for r in census['streams']['development']['supported_parents']][:protocol['chains']]
        assert len(parents) == protocol['chains']
        condition = source['condition']
        positions = source['development']['positions'][parents]
        generator = torch.Generator().manual_seed(protocol['warm_seed']+args.condition_index)
        signs = 2*torch.randint(2, (len(parents),), generator=generator)-1
        positions = positions*signs[:, None, None]
        warm_cost = 0
        provenance = dict(source_results_sha256=source['source_results_sha256'],
            development_source_sha256=source['evaluation_sha256'], inversion_signs=signs.tolist(),
            source_attempts=512, chemically_supported_development=census['streams']['development']['chemically_supported'])
    else:
        if args.method is None or args.warm is None:
            raise ValueError('Evaluation requires method and immutable warm input')
        warm = json.loads((args.warm/'results.json').read_text())
        assert warm['complete'] and warm['protocol_sha256'] == sha(pp)
        assert warm['condition_index'] == args.condition_index
        assert sha(args.warm/'trace.pt') == warm['trace_sha256']
        saved = torch.load(args.warm/'trace.pt', map_location='cpu', weights_only=False)
        condition, parents = warm['condition'], warm['parent_ids']
        initial = [saved['states'][i] for i in saved['history_state_ids'][-1]]
        positions = torch.stack([s['positions'] for s in initial])
        warm_cost = warm['new_raw_queries']
        provenance = dict(warm_results_sha256=sha(args.warm/'results.json'), warm_trace_sha256=warm['trace_sha256'],
            source_results_sha256=warm['source_results_sha256'], development_source_sha256=warm['development_source_sha256'],
            source_attempts=warm['source_attempts'], chemically_supported_development=warm['chemically_supported_development'])
        generator = torch.Generator().manual_seed(protocol['evaluation_seeds'][args.replica]+100*args.condition_index)
    report = dict(complete=False, protocol_sha256=sha(pp), stage=args.stage, method=args.method,
        replica=args.replica, condition_index=args.condition_index, condition=condition, parent_ids=parents,
        stream='development', training_coordinates_used=False, reference_coordinates_used=False,
        source_provenance_loader_reads_both_streams=args.stage == 'warm', only_development_positions_used=True,
        inherited_source_raw_queries=512, inherited_warm_raw_queries=warm_cost,
        scientific_submission_ready=False, history=[], **provenance)
    write(output, report)
    scale_generator = torch.Generator().manual_seed(protocol['scale_seed']+100*args.condition_index+(0 if args.stage == 'warm' else 10+args.replica))
    oracle = target = None
    transitions, ids, scale_choices = [], [], []
    start = time.perf_counter()
    try:
        oracle = EnergyOracle(args.oracle_python, root/'scripts/research/oracle_worker.py', args.oracle_checkpoint,
            numbers=condition['numbers'], charge=condition['charge'], spin_multiplicity=condition['spin_multiplicity'],
            device='cuda', batch_size=32)
        if oracle.handshake.get('tf32') or oracle.handshake.get('base_precision_dtype') != 'torch.float32':
            raise ValueError('Unqualified physical precision')
        report['oracle_runtime'] = oracle.handshake
        target = ChemicalTarget(oracle, condition, physical['kT_eV'], physical['restraint_eV_A2'])
        states = target.evaluate([target.coordinate_state(x) for x in positions], phase='initial')
        if args.stage == 'evaluate':
            for a, b in zip(initial, states):
                torch.testing.assert_close(a['energy_eV'], b['energy_eV'], atol=1e-4, rtol=0)
                torch.testing.assert_close(a['force_eV_A'], b['force_eV_A'], atol=1e-4, rtol=0)
        steps = protocol['warm_steps'] if args.stage == 'warm' else protocol['evaluation_steps']
        for step in range(steps+1):
            row = dict(step=step, energy_eV=[float(s['energy_eV']) for s in states],
                potential_eV=[float(s['potential_eV']) for s in states], smiles=[s['graph']['connectivity_smiles'] for s in states],
                new_raw_queries=oracle.evaluated, total_raw_queries=oracle.evaluated+512+warm_cost)
            report['history'].append(row)
            ids.append([s['state_id'] for s in states])
            if step % 32 == 0 or step == steps:
                print(json.dumps(row), flush=True)
                write(output, report)
            if step == steps:
                break
            choices = torch.randint(len(protocol['local_scales']), (len(states),), generator=scale_generator)
            scale_choices.append(choices.tolist())
            scales = torch.tensor(protocol['local_scales'], dtype=torch.float64)[choices]*target.kT**.5
            kind = 'local' if args.stage == 'warm' else protocol['schedule'][step % 4]
            if args.method == 'local_only' and kind == 'fragment_exchange':
                kind = 'local'
            phase = f'{args.stage}_{step}'
            if kind == 'local':
                states, rows = target.transition(states, policy=None, generator=generator, proposal_std=scales, phase=phase, local_only=True)
                for r in rows:
                    r['kind'] = kind
            elif kind == 'force_rotation':
                states, rows = uniform_internal_transition(target, states, kind=kind, generator=generator, phase=phase)
            else:
                states, rows = fragment_transition(target, states, method=args.method, generator=generator, phase=phase,
                    max_fragment_atoms=protocol['max_fragment_atoms'], radial_width=protocol['radial_width'], concentration=protocol['site_concentration'])
            transitions.extend(rows)
        artifact = dict(states=target.states, query_trace=target.query_trace, transitions=transitions,
            history_state_ids=ids, scale_choice_history=scale_choices, generator_state=generator.get_state(),
            scale_generator_state=scale_generator.get_state(), stream='development')
        torch.save(artifact, args.out/'trace.pt')
        limit = protocol['maximum_warm_queries_per_condition'] if args.stage == 'warm' else protocol['maximum_eval_queries_per_arm']
        if oracle.evaluated > limit:
            raise RuntimeError('Frozen physical-query ceiling exceeded')
        report.update(complete=True, trace_sha256=sha(args.out/'trace.pt'), new_raw_queries=oracle.evaluated,
            requested_raw_queries=oracle.requested_evaluations, seconds=time.perf_counter()-start,
            limitation='Physical feasibility pilot at fixed microstep and maximum-query caps; actual costs can differ, and no equilibrium or AI advantage follows.')
        write(output, report)
    except Exception as exc:
        if target is not None:
            torch.save(dict(states=target.states, query_trace=target.query_trace, transitions=transitions,
                history_state_ids=ids, generator_state=generator.get_state(), scale_generator_state=scale_generator.get_state()), args.out/'failed_trace.pt')
        report.update(failure=f'{type(exc).__name__}: {exc}', new_raw_queries=oracle.evaluated if oracle else 0,
                      requested_raw_queries=oracle.requested_evaluations if oracle else 0)
        write(output, report)
        raise
    finally:
        if oracle is not None:
            oracle.close()


if __name__ == '__main__':
    main()
