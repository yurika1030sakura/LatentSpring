#!/usr/bin/env python3
"""Full saved-oracle replay and independent normalized joint-density audit."""
import argparse
import json
import math
from pathlib import Path
import torch
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.chemical_path_guide import exchanged_bond_graph
from cfm_mol.joint_chemical_geometry import joint_chemical_transition, distinct_anchor_actions
from cfm_mol.terminal_rotation import uniform_internal_transition
from scripts.research.audit_masked_angular import ReplayOracle, equal, sha
from scripts.research.evaluate_joint_chemical import load_model


def independent_log_q(x, y, bonds, numbers, electronic, radii, action, order, kind, model, protocol, trace):
    i, j, k, l = action
    desired = exchanged_bond_graph(bonds, action)
    root_pairs = [(i, l, j), (j, k, i)]
    rr = [(y[a]-y[b]).norm() for a, b, _ in root_pairs]
    sigma = protocol['radial_width']
    log_q = 0.
    template = x.clone()
    for radius, (a, b, partner) in zip(rr, root_pairs):
        mean = float((radii[a]+radii[b]).log())
        ell = float(radius.log())
        log_q += -.5*((ell-mean)/sigma)**2-math.log(sigma)-.5*math.log(2*math.pi)-3*ell
        old = x[partner]-x[b]
        template[a] = x[b]+radius*old/old.norm()
    template -= template.mean(0)
    torch.testing.assert_close(template, trace['template'], atol=1e-9, rtol=1e-9)
    for stage, index in enumerate([0, 1] if order == 0 else [1, 0]):
        a, b, _ = root_pairs[index]
        row = trace['steps'][stage]
        assert tuple(row['root']) == (a, b)
        torch.testing.assert_close(template, row['context'], atol=1e-9, rtol=1e-9)
        if model is not None:
            with torch.no_grad():
                e, matrix, _ = model(template[None], desired[None], numbers, electronic,
                                     torch.tensor([[a, b]], dtype=torch.long))
            e, matrix = e[0], matrix[0]
        else:
            e = x.new_zeros(3)
            matrix = x.new_zeros(3, 3)
            if kind == 'site':
                for neighbor in range(len(x)):
                    if neighbor != a and desired[b, neighbor] > 0:
                        v = template[neighbor]-template[b]
                        e -= v/v.norm().clamp_min(1e-12)
                e *= protocol['site_concentration']/e.norm().clamp_min(1e-12)
        torch.testing.assert_close(e, row['eta'], atol=1e-8, rtol=1e-9)
        torch.testing.assert_close(matrix, row['matrix'], atol=1e-8, rtol=1e-9)
        # Independently evaluate the explicitly normalized mixture of two vMFs.
        values, axes = torch.linalg.eigh(matrix)
        delta = values[-1]-values[-2]
        params = torch.stack([e+delta*axes[:, -1], e-delta*axes[:, -1]])
        log_c = []
        for eta in params:
            kappa = float(eta.norm())
            if kappa < 1e-5:
                value = -math.log(4*math.pi)-kappa*kappa/6+kappa**4/180
            else:
                value = math.log(kappa)-math.log(4*math.pi)-math.log(math.sinh(kappa))
            log_c.append(value)
        log_c = torch.tensor(log_c, dtype=x.dtype)
        weights = (-log_c).log_softmax(0)
        u = (y[a]-y[b])/rr[index]
        lp = float(torch.logsumexp(weights+log_c+params@u, 0))
        assert abs(lp-float(row['angular_log_density'])) < 1e-7
        log_q += lp
        template[a] = template[b]+rr[index]*u
        template -= template.mean(0)
    torch.testing.assert_close(template, y, atol=1e-9, rtol=1e-9)
    assert abs(log_q-float(trace['log_coordinate_density'])) < 1e-7
    return log_q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ['run', 'models', 'table', 'out']:
        parser.add_argument('--'+arg, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    pp = root/'research/evidence/joint_chemical_protocol_v1.json'
    protocol = json.loads(pp.read_text())
    physical = json.loads((root/'research/evidence/parity_training_protocol_v1.json').read_text())
    header = json.loads((args.table/'results.json').read_text())
    assert sha(args.table/'results.json') == protocol['table_results_sha256']
    assert sha(args.table/'development.pt') == header['artifacts']['development']
    starts = torch.load(args.table/'development.pt', map_location='cpu', weights_only=False)
    summaries = []
    for method in protocol['methods']:
        for replica in protocol['replicas']:
            directory = args.run/f'{method}_s{replica}'
            report = json.loads((directory/'results.json').read_text())
            assert report['complete'] and report['evaluation_protocol_sha256'] == sha(pp)
            assert report['trace_sha256'] == sha(directory/'trace.pt')
            assert report['condition'] == starts['condition']
            assert report['development_parent_ids'] == starts['source_parent_ids']
            data = torch.load(directory/'trace.pt', map_location='cpu', weights_only=False)
            oracle = ReplayOracle(data['query_trace'])
            target = ChemicalTarget(oracle, report['condition'], physical['kT_eV'], physical['restraint_eV_A2'])
            model, metadata = load_model(args.models, method, replica, protocol, header['artifacts']['training'])
            assert report['policy_sha256'] == (metadata['checkpoint_sha256'] if metadata else None)
            assert report['inherited_training_raw_queries'] == (metadata['inherited_training_raw_queries'] if metadata else 0)
            assert report['inherited_source_raw_queries'] == (header['inherited_source_raw_queries'] if metadata else 512)
            assert report['inherited_development_warm_raw_queries'] == header['streams']['development']['raw_queries']
            states = target.evaluate([target.coordinate_state(starts['states'][i]['positions']) for i in starts['warm_state_ids']], phase='initial')
            rng = torch.Generator().manual_seed(protocol['evaluation_seeds'][replica])
            srng = torch.Generator().manual_seed(protocol['scale_seed']+replica)
            numbers = torch.tensor(target.numbers, dtype=torch.long)
            electronic = torch.tensor([target.condition['charge'], target.condition['spin_multiplicity'], target.kT], dtype=torch.float64)
            chains = len(states)
            assert chains == 4
            totals = {kind: dict(attempts=0, valid=0, accepted=0, by_parent_valid=[0]*chains, by_parent_accepted=[0]*chains) for kind in protocol['schedule']}
            checked = 0
            for step in range(protocol['steps']):
                assert [s['state_id'] for s in states] == data['history_state_ids'][step]
                choices = torch.randint(len(protocol['local_scales']), (chains,), generator=srng)
                assert choices.tolist() == data['scale_choice_history'][step]
                scales = torch.tensor(protocol['local_scales'], dtype=torch.float64)[choices]*target.kT**.5
                kind = protocol['schedule'][step % len(protocol['schedule'])]
                old = list(states)
                phase = f'evaluation_{step}'
                if kind == 'local':
                    states, rows = target.transition(states, policy=None, generator=rng, proposal_std=scales, phase=phase, local_only=True)
                    for row in rows:
                        row['kind'] = kind
                elif kind == 'force_rotation':
                    states, rows = uniform_internal_transition(target, states, kind=kind, generator=rng, phase=phase)
                else:
                    states, rows = joint_chemical_transition(target, states, kind=method, generator=rng,
                        phase=phase, model=model, radial_width=protocol['radial_width'], site_concentration=protocol['site_concentration'])
                    for index, row in enumerate(rows):
                        if not row['valid']:
                            continue
                        new = target.states[row['new_state_id']]
                        x, y = old[index]['positions'], new['positions']
                        if method == 'deterministic':
                            i, j, k, l = row['action']
                            r = target.radii
                            correction = 3*float(((r[i]+r[l])/(r[j]+r[l])).log()+((r[j]+r[k])/(r[i]+r[k])).log())
                        else:
                            qf = independent_log_q(x, y, old[index]['graph']['bond_orders'], numbers, electronic,
                                target.radii, row['action'], row['order'], method, model, protocol, row['forward'])
                            qr = independent_log_q(y, x, new['graph']['bond_orders'], numbers, electronic,
                                target.radii, row['inverse_action'], row['order'], method, model, protocol, row['reverse'])
                            correction = qr-qf
                        nf = len(distinct_anchor_actions(numbers, old[index]['graph']['bond_orders']))
                        nr = len(distinct_anchor_actions(numbers, new['graph']['bond_orders']))
                        ratio = -float(new['potential_eV']-old[index]['potential_eV'])/target.kT+correction+math.log(nf/nr)
                        assert abs(ratio-row['log_acceptance_ratio']) < 1e-7
                        assert row['accepted'] == (row['log_uniform'] < min(0., ratio))
                        checked += 1
                equal(rows, data['transitions'][step*chains:(step+1)*chains])
                assert [s['state_id'] for s in states] == data['history_state_ids'][step+1]
                history = report['history'][step+1]
                assert history['smiles'] == [s['graph']['connectivity_smiles'] for s in states]
                assert history['energy_eV'] == [float(s['energy_eV']) for s in states]
                assert history['new_raw_queries'] == oracle.evaluated
                assert history['total_raw_queries'] == oracle.evaluated+sum(report[k] for k in ['inherited_training_raw_queries', 'inherited_source_raw_queries', 'inherited_development_warm_raw_queries'])
                for index, row in enumerate(rows):
                    totals[kind]['attempts'] += 1
                    totals[kind]['valid'] += row['valid']
                    totals[kind]['accepted'] += row['accepted']
                    totals[kind]['by_parent_valid'][index] += row['valid']
                    totals[kind]['by_parent_accepted'][index] += row['accepted']
            equal(target.states, data['states'])
            equal(target.query_trace, data['query_trace'])
            equal(rng.get_state(), data['generator_state'])
            equal(srng.get_state(), data['scale_generator_state'])
            assert oracle.index == len(oracle.queries)
            assert oracle.evaluated == report['new_raw_queries'] == report['requested_raw_queries']
            assert len(data['transitions']) == protocol['steps']*chains
            assert len(data['history_state_ids']) == len(report['history']) == protocol['steps']+1
            reference = 'CS(F)(F)(F)(F)F'
            summaries.append(dict(method=method, replica=replica, raw_queries=oracle.evaluated,
                total_raw_queries=report['history'][-1]['total_raw_queries'], full_producer_replay=True,
                independent_joint_MH_ratio_checks=checked, all_random_streams_replayed=True,
                maximum_replay_position_error_A=oracle.maximum_position_error, oracle_requeried=False, moves=totals,
                reference_connectivity_first_hit_step=[next((h['step'] for h in report['history'] if h['smiles'][i] == reference), None) for i in range(chains)],
                final_energy_eV=report['history'][-1]['energy_eV'], seconds=report['seconds'], trace_sha256=report['trace_sha256']))
            print(json.dumps(summaries[-1]), flush=True)
    result = dict(complete=True, rows=summaries, evaluation_protocol_sha256=sha(pp),
                  total_new_raw_queries=sum(s['raw_queries'] for s in summaries), scientific_submission_ready=False)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
