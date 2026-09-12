#!/usr/bin/env python3
"""Continue frozen physical controls to the learned arms' actual total costs."""
import argparse
import copy
import json
from pathlib import Path
import time
import torch
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.energy_oracle import EnergyOracle
from cfm_mol.terminal_rotation import uniform_internal_transition
from cfm_mol.joint_chemical_geometry import joint_chemical_transition
from scripts.research.evaluate_joint_chemical import sha, write


class OffsetOracle:
    """Count immutable inherited calls plus new calls in the continued trace."""
    def __init__(self, oracle, offset):
        self.oracle, self.offset = oracle, offset

    @property
    def evaluated(self):
        return self.offset+self.oracle.evaluated

    def evaluate_chunked(self, x, max_request):
        return self.oracle.evaluate_chunked(x, max_request=max_request)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ['previous', 'out', 'oracle-python', 'oracle-checkpoint']:
        parser.add_argument('--'+arg, type=Path, required=True)
    parser.add_argument('--method', choices=['deterministic', 'uniform', 'site'], required=True)
    parser.add_argument('--replica', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    pp = root/'research/evidence/joint_chemical_protocol_v1.json'
    cp = root/'research/evidence/joint_chemical_full_cost_protocol_v1.json'
    protocol, continuation = json.loads(pp.read_text()), json.loads(cp.read_text())
    assert sha(pp) == continuation['original_protocol_sha256']
    name = f'{args.method}_s{args.replica}'
    directory = args.previous/name
    parent = json.loads((directory/'results.json').read_text())
    assert parent['complete'] and parent['evaluation_protocol_sha256'] == sha(pp)
    assert sha(directory/'results.json') == continuation['parent_results_sha256'][name]
    assert sha(directory/'trace.pt') == parent['trace_sha256']
    assert parent['inherited_training_raw_queries'] == 0
    physical = json.loads((root/'research/evidence/parity_training_protocol_v1.json').read_text())
    assert sha(root/'research/evidence/parity_training_protocol_v1.json') == protocol['physical_protocol_sha256']
    assert sha(args.oracle_checkpoint) == physical['raw_oracle_sha256']
    data = torch.load(directory/'trace.pt', map_location='cpu', weights_only=False)
    generator, scale_generator = torch.Generator(), torch.Generator()
    generator.set_state(data['generator_state'])
    scale_generator.set_state(data['scale_generator_state'])
    report = copy.deepcopy(parent)
    for key in ['trace_sha256', 'seconds', 'new_raw_queries', 'requested_raw_queries', 'proposals', 'invalid_proposals', 'accepted_proposals']:
        report.pop(key, None)
    budget = continuation['target_total_raw_queries'][args.replica]
    inherited = parent['inherited_source_raw_queries']+parent['inherited_development_warm_raw_queries']
    report.update(complete=False, continuation_protocol_sha256=sha(cp), parent_results_sha256=sha(directory/'results.json'),
        inherited_sampling_raw_queries=parent['new_raw_queries'], target_total_raw_queries=budget,
        parent_sampling_seconds=parent['seconds'])
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out/'results.json'
    if output.exists():
        raise FileExistsError(output)
    write(output, report)
    oracle = target = None
    start = time.perf_counter()
    try:
        oracle = EnergyOracle(args.oracle_python, root/'scripts/research/oracle_worker.py', args.oracle_checkpoint,
            numbers=parent['condition']['numbers'], charge=parent['condition']['charge'],
            spin_multiplicity=parent['condition']['spin_multiplicity'], device='cuda', batch_size=32)
        if oracle.handshake.get('tf32') or oracle.handshake.get('base_precision_dtype') != 'torch.float32':
            raise ValueError('Unqualified physical precision')
        report['continuation_oracle_runtime'] = oracle.handshake
        counter = OffsetOracle(oracle, parent['new_raw_queries'])
        target = ChemicalTarget(counter, parent['condition'], physical['kT_eV'], physical['restraint_eV_A2'])
        # Saved conservative values have already been independently replayed.
        # Reuse them; a process restart does not require new physical queries.
        target.states, target.query_trace = data['states'], data['query_trace']
        states = [target.states[i] for i in data['history_state_ids'][-1]]
        first = len(data['history_state_ids'])-1
        assert first == protocol['steps']
        for step in range(first, continuation['maximum_total_steps']):
            # One whole four-chain microstep costs at most eight raw queries.
            # Keep a complete kernel step and report the possible <=6-call gap.
            if budget-(counter.evaluated+inherited) < 2*len(states):
                break
            choices = torch.randint(len(protocol['local_scales']), (len(states),), generator=scale_generator)
            data['scale_choice_history'].append(choices.tolist())
            scales = torch.tensor(protocol['local_scales'], dtype=torch.float64)[choices]*target.kT**.5
            kind = protocol['schedule'][step % len(protocol['schedule'])]
            phase = f'evaluation_{step}'
            if kind == 'local':
                states, rows = target.transition(states, policy=None, generator=generator, proposal_std=scales, phase=phase, local_only=True)
                for row in rows:
                    row['kind'] = kind
            elif kind == 'force_rotation':
                states, rows = uniform_internal_transition(target, states, kind=kind, generator=generator, phase=phase)
            else:
                states, rows = joint_chemical_transition(target, states, kind=args.method, generator=generator, phase=phase,
                    radial_width=protocol['radial_width'], site_concentration=protocol['site_concentration'])
            data['transitions'].extend(rows)
            data['history_state_ids'].append([s['state_id'] for s in states])
            history = dict(step=step+1, energy_eV=[float(s['energy_eV']) for s in states],
                potential_eV=[float(s['potential_eV']) for s in states], smiles=[s['graph']['connectivity_smiles'] for s in states],
                new_raw_queries=counter.evaluated, total_raw_queries=counter.evaluated+inherited)
            report['history'].append(history)
            if (step+1) % 128 == 0:
                print(json.dumps(history), flush=True)
                write(output, report)
        data['generator_state'], data['scale_generator_state'] = generator.get_state(), scale_generator.get_state()
        torch.save(data, args.out/'trace.pt')
        elapsed = time.perf_counter()-start
        report.update(complete=True, new_raw_queries=counter.evaluated,
            requested_raw_queries=parent['requested_raw_queries']+oracle.requested_evaluations,
            additional_raw_queries=oracle.evaluated, additional_requested_raw_queries=oracle.requested_evaluations,
            additional_seconds=elapsed, seconds=parent['seconds']+elapsed,
            trace_sha256=sha(args.out/'trace.pt'), proposals=len(data['transitions']),
            invalid_proposals=sum(not row['valid'] for row in data['transitions']),
            accepted_proposals=sum(row['accepted'] for row in data['transitions']),
            actual_total_steps=len(data['history_state_ids'])-1, total_budget_gap=budget-counter.evaluated-inherited)
        assert report['total_budget_gap'] >= 0
        write(output, report)
    except Exception as exc:
        if target is not None:
            data['generator_state'], data['scale_generator_state'] = generator.get_state(), scale_generator.get_state()
            torch.save(data, args.out/'failed_trace.pt')
        report.update(failure=f'{type(exc).__name__}: {exc}',
            additional_raw_queries=oracle.evaluated if oracle else 0,
            additional_requested_raw_queries=oracle.requested_evaluations if oracle else 0)
        write(output, report)
        raise
    finally:
        if oracle is not None:
            oracle.close()


if __name__ == '__main__':
    main()
