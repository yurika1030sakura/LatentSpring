#!/usr/bin/env python3
"""Real-flow/RNG replay and independent auxiliary-density checks of EACF MH."""
import argparse
import json
import math
from pathlib import Path
import torch
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.eacf_transport import EACFTransport
from cfm_mol.directional_flow_mh import directional_flow_transition
from cfm_mol.terminal_rotation import uniform_internal_transition
from scripts.research.audit_masked_angular import ReplayOracle, equal, sha


class CheckedTransport:
    def __init__(self, transport, saved):
        self.transport, self.saved = transport, saved
        self.index = 0
        self.aux_scale = transport.aux_scale
        self.maximum_error = 0.
        self.inverse_error = 0.
        self.inverse_volume_error = 0.

    def transform(self, x, a, d):
        expected = self.saved[self.index]
        equal(x, expected['positions'])
        equal(a, expected['auxiliary'])
        equal(d, expected['directions'])
        actual = self.transport.transform(x, a, d, check_inverse=True)
        for key in ['positions', 'auxiliary', 'log_volume']:
            self.maximum_error = max(self.maximum_error, float((actual[key]-expected['response'][key]).abs().max()))
            torch.testing.assert_close(actual[key], expected['response'][key], atol=1e-7, rtol=1e-8)
        self.inverse_error = max(self.inverse_error, actual['inverse_error'])
        self.inverse_volume_error = max(self.inverse_volume_error, actual['inverse_volume_error'])
        assert actual['inverse_error'] < 1e-7 and actual['inverse_volume_error'] < 1e-7
        self.index += 1
        return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'flows', 'table', 'transport-python', 'upstream', 'out']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--profile', choices=['published', 'compact'], required=True)
    parser.add_argument('--replica', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    protocol_path = root/'research/evidence/eacf_directional_protocol_v1.json'
    protocol = json.loads(protocol_path.read_text())
    shared = json.loads((root/'research/evidence/normalized_site_evaluation_protocol_v1.json').read_text())
    physical = json.loads((root/'research/evidence/parity_training_protocol_v1.json').read_text())
    report = json.loads((args.run/'results.json').read_text())
    assert report['complete'] and report['protocol_sha256'] == sha(protocol_path)
    assert report['trace_sha256'] == sha(args.run/'trace.pt')
    assert report['flow_trace_sha256'] == sha(args.run/'flow_trace.npz')
    assert report['development_sha256'] == sha(args.table/'development.pt')
    warm = torch.load(args.table/'development.pt', map_location='cpu', weights_only=False)
    assert warm['condition'] == report['condition'] and warm['source_parent_ids'] == report['development_parent_ids']
    saved = torch.load(args.run/'trace.pt', map_location='cpu', weights_only=False)
    stem = 'eacf_joint' if args.profile == 'published' else 'eacf_compact'
    directory = args.flows/f'{stem}_condition_00_s{args.replica}_v1'
    assert sha(directory/'adapter.pkl') == report['checkpoint_sha256'] == protocol['checkpoint_sha256'][args.profile][args.replica]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    with EACFTransport(args.transport_python, root/'scripts/research/eacf_transport_worker.py', directory,
                       args.upstream, log_path=args.out.with_suffix('.worker.log')) as worker:
        transport = CheckedTransport(worker, saved['transport_traces'])
        oracle = ReplayOracle(saved['query_trace'])
        target = ChemicalTarget(oracle, report['condition'], physical['kT_eV'], physical['restraint_eV_A2'])
        states = target.evaluate([target.coordinate_state(warm['states'][i]['positions']) for i in warm['warm_state_ids']], phase='initial')
        rng = torch.Generator().manual_seed(shared['evaluation_seeds'][args.replica])
        srng = torch.Generator().manual_seed(shared['scale_seed']+args.replica)
        totals = {kind: dict(attempts=0, valid=0, accepted=0) for kind in ['local', 'force_rotation', 'directional_flow']}
        independent_ratios = 0
        for step in range(shared['steps']):
            assert [s['state_id'] for s in states] == saved['history_state_ids'][step]
            choices = torch.randint(len(shared['local_scales']), (len(states),), generator=srng)
            assert choices.tolist() == saved['scale_choice_history'][step]
            scales = torch.tensor(shared['local_scales'], dtype=torch.float64)[choices]*target.kT**.5
            kind, phase = shared['schedule'][step % 4], f'evaluation_{step}'
            old = list(states)
            if kind == 'local':
                states, rows = target.transition(states, policy=None, generator=rng, proposal_std=scales, phase=phase, local_only=True)
                for row in rows:
                    row['kind'] = kind
            elif kind == 'force_rotation':
                states, rows = uniform_internal_transition(target, states, kind=kind, generator=rng, phase=phase)
            else:
                kind = 'directional_flow'
                states, rows = directional_flow_transition(target, states, transport, generator=rng, phase=phase)
                for index, row in enumerate(rows):
                    before = old[index]
                    x, a, y, b = before['positions'], row['source_auxiliary'], row['proposal_positions'], row['proposal_auxiliary']
                    scale = worker.aux_scale
                    dimension = 3*len(x)
                    forward = -.5*float(((a-x)/scale).square().sum())-dimension*math.log(scale*math.sqrt(2*math.pi))
                    reverse = -.5*float(((b-y)/scale).square().sum())-dimension*math.log(scale*math.sqrt(2*math.pi))
                    assert abs(forward-float(row['forward_auxiliary_log_prob'])) < 1e-7
                    assert abs(reverse-float(row['reverse_auxiliary_log_prob'])) < 1e-7
                    assert row['reverse_direction'] == 1-row['direction']
                    if row['valid']:
                        after = target.states[row['new_state_id']]
                        potential_change = float(after['energy_eV']-before['energy_eV'])+physical['restraint_eV_A2']/2*float(y.square().sum()-x.square().sum())
                        ratio = -potential_change/physical['kT_eV']+reverse-forward+float(row['log_volume'])
                        assert abs(ratio-row['log_acceptance_ratio']) < 1e-7
                        assert row['accepted'] == (row['log_uniform'] < min(0., ratio))
                        independent_ratios += 1
            equal(rows, saved['transitions'][4*step:4*(step+1)])
            assert [s['state_id'] for s in states] == saved['history_state_ids'][step+1]
            assert report['history'][step+1]['new_raw_queries'] == oracle.evaluated
            for row in rows:
                totals[kind]['attempts'] += 1
                totals[kind]['valid'] += row['valid']
                totals[kind]['accepted'] += row['accepted']
        equal(rng.get_state(), saved['generator_state'])
        equal(srng.get_state(), saved['scale_generator_state'])
        equal(target.states, saved['states'])
        equal(target.query_trace, saved['query_trace'])
        assert oracle.evaluated == report['new_raw_queries'] == report['requested_raw_queries']
        assert transport.index == len(saved['transport_traces'])
        reference = 'CS(F)(F)(F)(F)F'
        result = dict(complete=True, method=report['method'], replica=args.replica, results_sha256=sha(args.run/'results.json'),
            trace_sha256=report['trace_sha256'], flow_trace_sha256=report['flow_trace_sha256'],
            raw_queries=oracle.evaluated, total_raw_queries=report['history'][-1]['total_raw_queries'],
            full_producer_and_real_flow_replay=True, all_random_streams_replayed=True, independent_ratios=independent_ratios,
            real_flow_maximum_error=transport.maximum_error, maximum_inverse_error=transport.inverse_error,
            maximum_inverse_volume_error=transport.inverse_volume_error, maximum_oracle_position_error_A=oracle.maximum_position_error,
            moves=totals, reference_connectivity_first_hit_step=[next((h['step'] for h in report['history'] if h['smiles'][i] == reference), None) for i in range(4)],
            final_energy_eV=report['history'][-1]['energy_eV'], original_seconds=report['seconds'],
            new_physical_queries=0, validation_transformations=worker.validation_transformations,
            scientific_submission_ready=False)
        args.out.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
