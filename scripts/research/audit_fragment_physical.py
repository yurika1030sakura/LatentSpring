#!/usr/bin/env python3
"""Replay the fragment pilot and independently reconstruct its map and MH ratio."""
import argparse
import json
import math
from pathlib import Path
import torch
from cfm_mol.entropy_source import load_entropy_source
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.fragment_sampler import fragment_transition, fragment_method_actions
from cfm_mol.terminal_rotation import uniform_internal_transition
from scripts.research.audit_masked_angular import ReplayOracle, equal, sha


def independent_fragment_map(x, action, vectors, angles):
    i, j, k, l = action['roots']
    old = torch.stack([x[i]-x[k], x[j]-x[l]])
    y = x.clone()
    for index, (root, anchor, atoms) in enumerate([(i, l, action['fragments'][0]), (j, k, action['fragments'][1])]):
        a = old[index]/old[index].norm()
        b = vectors[index]/vectors[index].norm()
        cross = torch.linalg.cross(a, b)
        cosine = torch.dot(a, b)
        offset = x[list(atoms)]-x[root]
        first = torch.linalg.cross(cross.expand_as(offset), offset)
        aligned = offset+first+torch.linalg.cross(cross.expand_as(offset), first)/(1+cosine)
        angle = angles[index]
        rotated = angle.cos()*aligned+angle.sin()*torch.linalg.cross(b.expand_as(aligned), aligned)
        rotated += (1-angle.cos())*(aligned*b).sum(1, keepdim=True)*b
        y[list(atoms)] = x[anchor]+vectors[index]+rotated
    return y-y.mean(0), old


def independent_auxiliary_logp(x, target_bonds, radii, roots, vectors, sigma, concentration):
    result = -2*math.log(2*math.pi)
    for (leaf, anchor), vector in zip(roots, vectors):
        eta = torch.zeros(3, dtype=x.dtype)
        for node in range(len(x)):
            if node != leaf and target_bonds[anchor, node] > 0:
                delta = x[node]-x[anchor]
                eta -= delta/delta.norm().clamp_min(1e-12)
        eta *= concentration/eta.norm().clamp_min(1e-12)
        kappa = float(eta.norm())
        log_c = (-math.log(4*math.pi)-kappa*kappa/6+kappa**4/180 if kappa < 1e-5
                 else math.log(kappa)-math.log(4*math.pi)-math.log(math.sinh(kappa)))
        r = float(vector.norm())
        ell = math.log(r)
        mean = math.log(float(radii[leaf]+radii[anchor]))
        result += -.5*((ell-mean)/sigma)**2-math.log(sigma*math.sqrt(2*math.pi))-3*ell
        result += log_c+float(torch.dot(eta, vector/r))
    return result


def audit_arm(directory, positions, condition, parents, protocol, protocol_hash, physical, *, stage, method=None, replica=0, warm_queries=0, initial_signs=None):
    report = json.loads((directory/'results.json').read_text())
    assert report['complete'] and report['protocol_sha256'] == protocol_hash
    assert report['condition'] == condition and report['parent_ids'] == parents
    assert report['stream'] == 'development' and report['only_development_positions_used']
    assert sha(directory/'trace.pt') == report['trace_sha256']
    assert report['inherited_warm_raw_queries'] == warm_queries
    data = torch.load(directory/'trace.pt', map_location='cpu', weights_only=False)
    assert data['stream'] == 'development'
    index = report['condition_index']
    rng = torch.Generator().manual_seed(protocol['warm_seed']+index if stage == 'warm' else protocol['evaluation_seeds'][replica]+100*index)
    if stage == 'warm':
        signs = 2*torch.randint(2, (len(parents),), generator=rng)-1
        assert signs.tolist() == report['inversion_signs']
        positions = positions*signs[:, None, None]
    srng = torch.Generator().manual_seed(protocol['scale_seed']+100*index+(0 if stage == 'warm' else 10+replica))
    oracle = ReplayOracle(data['query_trace'])
    target = ChemicalTarget(oracle, condition, physical['kT_eV'], physical['restraint_eV_A2'])
    states = target.evaluate([target.coordinate_state(x) for x in positions], phase='initial')
    steps = protocol['warm_steps'] if stage == 'warm' else protocol['evaluation_steps']
    counts = {kind: dict(attempts=0, valid=0, accepted=0, accepted_multiatom=0, accepted_backbone=0,
              accepted_constitution=0) for kind in ['local', 'force_rotation', 'fragment_exchange']}
    independent_checks = 0
    for step in range(steps):
        assert [s['state_id'] for s in states] == data['history_state_ids'][step]
        choices = torch.randint(len(protocol['local_scales']), (len(states),), generator=srng)
        assert choices.tolist() == data['scale_choice_history'][step]
        scales = torch.tensor(protocol['local_scales'], dtype=torch.float64)[choices]*target.kT**.5
        kind = 'local' if stage == 'warm' else protocol['schedule'][step % 4]
        if method == 'local_only' and kind == 'fragment_exchange':
            kind = 'local'
        phase = f'{stage}_{step}'
        old = list(states)
        if kind == 'local':
            states, rows = target.transition(states, policy=None, generator=rng, proposal_std=scales, phase=phase, local_only=True)
            for row in rows:
                row['kind'] = kind
        elif kind == 'force_rotation':
            states, rows = uniform_internal_transition(target, states, kind=kind, generator=rng, phase=phase)
        else:
            states, rows = fragment_transition(target, states, method=method, generator=rng, phase=phase,
                max_fragment_atoms=protocol['max_fragment_atoms'], radial_width=protocol['radial_width'], concentration=protocol['site_concentration'])
            for chain, row in enumerate(rows):
                if 'proposal' not in row or not row['proposal']['map_valid']:
                    continue
                p = row['proposal']
                before = old[chain]
                y, auxiliary_old = independent_fragment_map(before['positions'], row['action'], p['new_vectors'], p['torsions'])
                torch.testing.assert_close(y, row['proposal_positions'], atol=1e-8, rtol=1e-9)
                torch.testing.assert_close(auxiliary_old, p['old_vectors'])
                recovered, _ = independent_fragment_map(y, row['inverse_action'], auxiliary_old, -p['torsions'])
                torch.testing.assert_close(recovered, before['positions'], atol=1e-8, rtol=1e-9)
                i, j, k, l = row['action']['roots']
                graph = before['graph']['bond_orders'].clone()
                graph[i, k] = graph[k, i] = graph[j, l] = graph[l, j] = 0
                graph[i, l] = graph[l, i] = graph[j, k] = graph[k, j] = 1
                torch.testing.assert_close(graph, p['desired_bonds'])
                forward = independent_auxiliary_logp(before['positions'], graph, target.radii, [(i, l), (j, k)],
                    p['new_vectors'], protocol['radial_width'], protocol['site_concentration'])
                reverse = independent_auxiliary_logp(y, before['graph']['bond_orders'], target.radii, [(i, k), (j, l)],
                    auxiliary_old, protocol['radial_width'], protocol['site_concentration'])
                assert abs(forward-float(p['log_forward'])) < 1e-7
                assert abs(reverse-float(p['log_reverse'])) < 1e-7
                assert p['augmented_log_jacobian'] == 0.
                if row['valid']:
                    after = target.states[row['new_state_id']]
                    nf = len(fragment_method_actions(before, condition['numbers'], method, protocol['max_fragment_atoms']))
                    nr = len(fragment_method_actions(after, condition['numbers'], method, protocol['max_fragment_atoms']))
                    ratio = -float(after['potential_eV']-before['potential_eV'])/target.kT+reverse-forward+math.log(nf/nr)
                    assert abs(ratio-row['log_acceptance_ratio']) < 1e-7
                    assert row['accepted'] == (row['log_uniform'] < min(0., ratio))
                independent_checks += 1
        equal(rows, data['transitions'][step*len(states):(step+1)*len(states)])
        assert [s['state_id'] for s in states] == data['history_state_ids'][step+1]
        history = report['history'][step+1]
        assert history['smiles'] == [s['graph']['connectivity_smiles'] for s in states]
        assert history['energy_eV'] == [float(s['energy_eV']) for s in states]
        assert history['new_raw_queries'] == oracle.evaluated
        assert history['total_raw_queries'] == oracle.evaluated+512+warm_queries
        for row in rows:
            counts[kind]['attempts'] += 1
            counts[kind]['valid'] += row['valid']
            counts[kind]['accepted'] += row['accepted']
            for field, key in [('multiatom', 'accepted_multiatom'), ('backbone_changed', 'accepted_backbone'), ('constitution_changed', 'accepted_constitution')]:
                counts[kind][key] += bool(row['accepted'] and row.get(field, False))
    equal(target.states, data['states'])
    equal(target.query_trace, data['query_trace'])
    equal(rng.get_state(), data['generator_state'])
    equal(srng.get_state(), data['scale_generator_state'])
    assert oracle.index == len(data['query_trace'])
    assert oracle.evaluated == report['new_raw_queries'] == report['requested_raw_queries']
    assert len(data['transitions']) == steps*len(states)
    summary = dict(condition_index=index, stage=stage, method=method, replica=replica, raw_queries=oracle.evaluated,
        total_raw_queries=report['history'][-1]['total_raw_queries'], moves=counts, independent_fragment_checks=independent_checks,
        full_producer_replay=True, all_random_streams_replayed=True, maximum_replay_position_error_A=oracle.maximum_position_error,
        trace_sha256=report['trace_sha256'], final_energy_eV=report['history'][-1]['energy_eV'],
        unique_connectivity_per_parent=[len(set(h['smiles'][i] for h in report['history'])) for i in range(len(states))],
        seconds=report['seconds'], oracle_requeried=False)
    print(json.dumps(summary), flush=True)
    return summary, torch.stack([s['positions'] for s in states])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'source', 'out']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    pp = root/'research/evidence/fragment_physical_protocol_v1.json'
    protocol = json.loads(pp.read_text())
    physical = json.loads((root/'research/evidence/parity_training_protocol_v1.json').read_text())
    census_path = root/'research/evidence/chemical_source_panel_audit_v2.json'
    assert sha(census_path) == protocol['census_sha256']
    census = json.loads(census_path.read_text())
    result = dict(complete=False, warm=[], rows=[], scientific_submission_ready=False)
    for index in protocol['condition_indices']:
        source = load_entropy_source(args.source, index,
            protocol_path=root/'research/evidence/species_breadth_source_protocol_v2.json',
            manifest_path=root/'research/evidence/development_panel_v1.json')
        parents = [r['parent_id'] for r in census['conditions'][index]['streams']['development']['supported_parents']][:protocol['chains']]
        warm, endpoint = audit_arm(args.run/f'warm_{index}', source['development']['positions'][parents], source['condition'],
            parents, protocol, sha(pp), physical, stage='warm')
        result['warm'].append(warm)
        for method in protocol['methods']:
            for replica in protocol['replicas']:
                summary, _ = audit_arm(args.run/f'condition_{index}/{method}_s{replica}', endpoint, source['condition'], parents,
                    protocol, sha(pp), physical, stage='evaluate', method=method, replica=replica, warm_queries=warm['raw_queries'])
                result['rows'].append(summary)
    result.update(complete=True, new_physical_queries=sum(r['raw_queries'] for r in result['warm']+result['rows']))
    assert result['new_physical_queries'] <= protocol['maximum_new_raw_queries']
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
