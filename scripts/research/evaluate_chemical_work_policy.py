#!/usr/bin/env python3
"""Exact one-step expectations over real molecular edit catalogues (not chains).

All candidate energies are obtained for evaluation only. Forward probabilities
are frozen before querying candidates; deployed selection uses no such labels.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path
import torch
from cfm_mol.chemical_work import PairedChemicalWork, LinearBondWork, ResidualChemicalWork
from cfm_mol.chemical_work_policy import catalogue, policy
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.energy_oracle import EnergyOracle
from scripts.research.evaluate_edit_bridge import inputs
from scripts.research.audit_masked_angular import ReplayOracle, equal, sha
from scripts.research.evaluate_chemical_policy import write


def load_models(project, protocol):
    models = dict(uniform='uniform', force='force')
    for name, spec in protocol['work_models'].items():
        path = project/spec['path']; assert sha(path) == spec['sha256']
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert saved['phase'] == 'full_fit' and set(saved['training_source_ids']) == set(range(36))
        cls = ResidualChemicalWork if saved.get('architecture') == 'residual' else (LinearBondWork if saved['variant'] == 'linear' else PairedChemicalWork)
        model = cls(**saved['configuration']).double()
        model.load_state_dict(saved['state_dict']); model.eval(); model.requires_grad_(False)
        models[name] = model
    return models


def run(target, models, sources, protocol):
    initial = [target.coordinate_state(old['positions']) for _, old in sources]
    target.evaluate(initial, phase='sources')
    rows, action_rows = [], []
    for old, (spec, previous) in zip(initial, sources):
        equal(old['graph'], previous['graph'])
        assert abs(float(old['potential_eV']-previous['potential_eV'])) < protocol['source_energy_tolerance_eV']
        forward = catalogue(target, old)
        # Enforce that every forward decision is independent of destination E/F.
        frozen = {name: policy(target, old, forward, model, protocol['uniform_fraction']) for name, model in models.items()}
        pending = [copy.deepcopy(item['candidate']) for item in forward['valid']]
        target.evaluate(pending, phase=f'parent_{spec["parent"]}_all_candidates_evaluation_only')
        totals = {name: dict(acceptance=0., utility_eV=0., constitutional_flow=0., proposed_work_eV=0.) for name in models}
        for action_index, (item, new) in enumerate(zip(forward['valid'], pending)):
            reverse = catalogue(target, new)
            matches = [i for i, row in enumerate(reverse['valid']) if tuple(row['action']) == tuple(item['record']['inverse_action'])]
            assert len(matches) == 1
            inverse_index = matches[0]
            torch.testing.assert_close(reverse['valid'][inverse_index]['candidate']['positions'], old['positions'], atol=1e-9, rtol=0)
            delta = float(new['potential_eV']-old['potential_eV'])
            changed = old['graph']['connectivity_smiles'] != new['graph']['connectivity_smiles']
            for name, model in models.items():
                backward = policy(target, new, reverse, model, protocol['uniform_fraction'])
                fwd = float(frozen[name]['log_probability'][action_index]); rev = float(backward['log_probability'][inverse_index])
                ratio = -delta/target.kT+item['record']['log_volume']+rev-fwd
                assert math.isfinite(ratio)
                alpha, probability = math.exp(min(0., ratio)), math.exp(fwd)
                total = totals[name]
                total['acceptance'] += probability*alpha
                total['utility_eV'] += -probability*alpha*delta
                total['constitutional_flow'] += probability*alpha*changed
                total['proposed_work_eV'] += probability*delta
                action_rows.append(dict(parent=spec['parent'], method=name, source_state_id=old['state_id'], candidate_state_id=new['state_id'],
                    action=item['action'], inverse_action=item['record']['inverse_action'], forward_valid_count=len(pending),
                    reverse_valid_count=len(reverse['valid']), forward_log_probability=fwd, reverse_log_probability=rev,
                    log_volume=item['record']['log_volume'], potential_change_eV=delta, log_acceptance_ratio=ratio,
                    forward_predicted_work_eV=float(frozen[name]['work'][action_index]),
                    reverse_predicted_work_eV=float(backward['work'][inverse_index]),
                    acceptance_probability=alpha, probability=probability))
        rows.append(dict(parent=spec['parent'], eligible=len(forward['valid'])+len(forward['failed']), valid=len(pending),
                         failed=forward['failed'], methods=totals))
    return dict(rows=rows, action_rows=action_rows, states=target.states, query_trace=target.query_trace)


def independent_checks(data, target):
    for row in data['action_rows']:
        old = data['states'][row['source_state_id']]; new = data['states'][row['candidate_state_id']]
        values = []
        for state in (old, new):
            q = data['query_trace'][state['query_batch']]; j = state['query_row']
            values.append((float(q['raw_energy_eV'][j])+float(q['inverted_energy_eV'][j]))/2
                          +target.restraint/2*float(state['positions'].square().sum()))
        expected = -(values[1]-values[0])/target.kT+row['log_volume']+row['reverse_log_probability']-row['forward_log_probability']
        assert abs(expected-row['log_acceptance_ratio']) < 1e-7
    for row in data['rows']:
        for method in row['methods']:
            selected = [r for r in data['action_rows'] if r['parent'] == row['parent'] and r['method'] == method]
            if selected:
                assert abs(sum(r['probability'] for r in selected)-1.) < 1e-10
                utility = sum(-r['probability']*math.exp(min(0., r['log_acceptance_ratio']))*r['potential_change_eV'] for r in selected)
                assert abs(utility-row['methods'][method]['utility_eV']) < 1e-10
    return len(data['action_rows'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'out', 'protocol'): parser.add_argument('--'+name, type=Path, required=True)
    for name in ('run', 'oracle-python', 'oracle-checkpoint'): parser.add_argument('--'+name, type=Path)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--phase', choices=['evaluate', 'reevaluate', 'audit'], required=True)
    args = parser.parse_args(); root = Path(__file__).resolve().parents[2]
    pp, protocol, physical, _, condition, sources = inputs(root, args.project, args.index, root/args.protocol)
    models = load_models(args.project, protocol)
    args.out.mkdir(parents=True, exist_ok=True); output = args.out/'results.json'
    if output.exists(): raise FileExistsError(output)
    report = dict(complete=False, index=args.index, phase=args.phase, protocol_sha256=sha(pp), new_raw_queries=0,
                  scientific_submission_ready=False, scope=protocol['interpretation'])
    write(output, report); oracle = target = None; start = time.monotonic()
    try:
        if args.phase in ('audit', 'reevaluate'):
            producer = json.loads((args.run/'results.json').read_text())
            assert producer['complete'] and sha(args.run/'trace.pt') == producer['trace_sha256']
            if args.phase == 'audit':
                assert producer['protocol_sha256'] == sha(pp)
            else:
                prior_path = root/protocol['reused_evaluation_protocol']
                assert sha(prior_path) == protocol['reused_evaluation_protocol_sha256'] == producer['protocol_sha256']
                prior = json.loads(prior_path.read_text())
                for key in ('physical_protocol', 'physical_protocol_sha256', 'sources', 'counts', 'uniform_fraction'):
                    assert prior[key] == protocol[key]
                for name, spec in prior['work_models'].items(): assert protocol['work_models'][name] == spec
            expected = torch.load(args.run/'trace.pt', map_location='cpu', weights_only=False)
            oracle = ReplayOracle(expected['query_trace'])
        else:
            assert sha(args.oracle_checkpoint) == physical['raw_oracle_sha256']
            oracle = EnergyOracle(args.oracle_python, root/'scripts/research/oracle_worker.py', args.oracle_checkpoint,
                numbers=condition['numbers'], charge=condition['charge'], spin_multiplicity=condition['spin_multiplicity'], device='cuda', batch_size=32)
            assert oracle.handshake['base_precision_dtype'] == 'torch.float32' and not oracle.handshake['tf32']
        target = ChemicalTarget(oracle, condition, physical['kT_eV'], physical['restraint_eV_A2'])
        actual = run(target, models, sources, protocol)
        assert oracle.evaluated == 2*(len(sources)+sum(r['valid'] for r in actual['rows']))
        assert oracle.evaluated == protocol['counts'][str(args.index)]['raw_queries']
        checks = independent_checks(actual, target)
        if args.phase == 'audit':
            equal(actual, expected); assert oracle.index == len(oracle.queries)
            report.update(complete=True, full_replay=True, source_results_sha256=sha(args.run/'results.json'),
                          trace_sha256=producer['trace_sha256'], producer_raw_queries=oracle.evaluated, independent_ratio_checks=checks)
        else:
            if args.phase == 'reevaluate':
                assert oracle.index == len(oracle.queries)
                equal(actual['states'], expected['states']); equal(actual['query_trace'], expected['query_trace'])
                old_methods = set(prior['work_models']) | {'uniform', 'force'}
                equal([row for row in actual['action_rows'] if row['method'] in old_methods], expected['action_rows'])
                report.update(reused_raw_queries=oracle.evaluated, reused_trace_sha256=producer['trace_sha256'],
                              reused_results_sha256=sha(args.run/'results.json'), unchanged_control_replay=True)
            else:
                assert oracle.evaluated == oracle.requested_evaluations
            torch.save(actual, args.out/'trace.pt')
            report.update(complete=True, trace_sha256=sha(args.out/'trace.pt'), physical_calls_in_trace=oracle.evaluated,
                new_raw_queries=0 if args.phase == 'reevaluate' else oracle.evaluated,
                rows=[{k:v for k,v in row.items() if k!='failed'} for row in actual['rows']], independent_ratio_checks=checks)
        report['elapsed_seconds'] = time.monotonic()-start; write(output, report)
        print(json.dumps({k:v for k,v in report.items() if k!='rows'}), flush=True)
    except Exception as exc:
        if target is not None: torch.save(dict(states=target.states, query_trace=target.query_trace), args.out/'failed_trace.pt')
        report['failure'] = f'{type(exc).__name__}: {exc}'
        if oracle is not None and args.phase == 'evaluate': report['new_raw_queries'] = oracle.evaluated
        write(output, report); raise
    finally:
        if oracle is not None and args.phase == 'evaluate': oracle.close()


if __name__ == '__main__': main()
