#!/usr/bin/env python3
"""Replay the frozen17-atom input panel with normalized defensive proposals."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.chemical_path_guide import exchanged_bond_graph
from cfm_mol.joint_chemical_geometry import joint_geometry_proposal, defensive_joint_proposal, distinct_anchor_actions
from scripts.research.evaluate_normalized_site import load_model, sha
from scripts.research.audit_joint_chemical import independent_log_q, independent_defensive_q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['previous', 'models', 'table', 'out']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    pp = root/'research/evidence/normalized_site_evaluation_protocol_v1.json'
    protocol = json.loads(pp.read_text())
    physical = json.loads((root/'research/evidence/parity_training_protocol_v1.json').read_text())
    header = json.loads((args.table/'results.json').read_text())
    assert sha(args.table/'results.json') == protocol['table_results_sha256']
    previous = json.loads((args.previous/'results.json').read_text())
    assert previous['complete'] and sha(args.previous/'trace.pt') == previous['trace_sha256']
    old = torch.load(args.previous/'trace.pt', map_location='cpu', weights_only=False)
    assert old['parent_ids'] == previous['parents'] and len(old['states']) == 32
    condition = json.loads((root/'research/evidence/chemical_source_panel_audit_v2.json').read_text())['conditions'][4]['condition']
    target = ChemicalTarget(None, condition, physical['kT_eV'], physical['restraint_eV_A2'])
    numbers = torch.tensor(target.numbers, dtype=torch.long)
    electronic = torch.tensor([condition['charge'], condition['spin_multiplicity'], target.kT], dtype=torch.float64)
    for state in old['states']:
        check = target.coordinate_state(state['positions'])
        assert torch.equal(check['graph']['bond_orders'], state['graph']['bond_orders'])
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'results.json').exists():
        raise FileExistsError(args.out/'results.json')
    records, rows = [], []
    for method in protocol['methods']:
        for replica in [0, 1]:
            model, metadata = load_model(args.models, method, replica, protocol, header['artifacts']['training'])
            proposal = joint_geometry_proposal if method == 'site' else defensive_joint_proposal
            kind = 'site' if method == 'site' else 'defensive_site'
            rng = torch.Generator().manual_seed(10831+replica)
            summary = dict(method=method, replica=replica, attempts=0, valid=0, validator_errors=0,
                no_action=0, independent_density_checks=0, checkpoint_sha256=metadata['checkpoint_sha256'] if metadata else None)
            for parent, state in zip(old['parent_ids'], old['states']):
                actions = distinct_anchor_actions(numbers, state['graph']['bond_orders'])
                for trial in range(4):
                    summary['attempts'] += 1
                    record = dict(method=method, replica=replica, parent_id=parent, trial=trial, valid=False)
                    if not actions:
                        summary['no_action'] += 1
                        records.append(record)
                        continue
                    choice = int(torch.randint(len(actions), (1,), generator=rng))
                    action = actions[choice]
                    i, j, k, l = action
                    order = int(torch.randint(2, (1,), generator=rng))
                    kwargs = dict(kind=kind, order=order, model=model, radial_width=protocol['radial_width'], site_concentration=protocol['site_concentration'])
                    x = state['positions']
                    y, q, forward = proposal(x, state['graph']['bond_orders'], numbers, electronic, target.radii, action, generator=rng, **kwargs)
                    record.update(action=action, order=order, choice_index=choice, positions=y, forward=forward, forward_q=q)
                    desired = exchanged_bond_graph(state['graph']['bond_orders'], action)
                    try:
                        candidate = target.coordinate_state(y)
                        if not torch.equal(candidate['graph']['bond_orders'], desired):
                            raise ValueError('Different endpoint bond matrix')
                    except (ValueError, IndexError, RuntimeError) as exc:
                        record.update(reason=str(exc), error_type=type(exc).__name__)
                        summary['validator_errors'] += not isinstance(exc, ValueError)
                        records.append(record)
                        continue
                    inverse = (i, j, l, k)
                    assert inverse in distinct_anchor_actions(numbers, desired)
                    recovered, qr, reverse = proposal(y, desired, numbers, electronic, target.radii, inverse, observed=x, **kwargs)
                    torch.testing.assert_close(recovered, x, atol=1e-9, rtol=1e-9)
                    if method == 'site':
                        independent_log_q(x, y, state['graph']['bond_orders'], numbers, electronic, target.radii, action, order, 'site', None, protocol, forward)
                        independent_log_q(y, x, desired, numbers, electronic, target.radii, inverse, order, 'site', None, protocol, reverse)
                    else:
                        independent_defensive_q(x, y, state['graph']['bond_orders'], numbers, electronic, target.radii, action, order, model, protocol, forward)
                        independent_defensive_q(y, x, desired, numbers, electronic, target.radii, inverse, order, model, protocol, reverse)
                    record.update(valid=True, reverse=reverse, reverse_q=qr)
                    records.append(record)
                    summary['valid'] += 1
                    summary['independent_density_checks'] += 2
            summary['generator_state_sha256'] = hashlib.sha256(rng.get_state().numpy().tobytes()).hexdigest()
            if method == 'site':
                match = next(row for row in previous['rows'] if row['method'] == 'site' and row['replica'] == replica)
                assert summary['valid'] == match['valid']
            rows.append(summary)
            print(json.dumps(summary), flush=True)
    torch.save(dict(records=records, input_parent_ids=old['parent_ids']), args.out/'trace.pt')
    result = dict(complete=True, condition_index=4, n_atoms=17, original_input_trace_sha256=previous['trace_sha256'],
        trace_sha256=sha(args.out/'trace.pt'), protocol_sha256=sha(pp), parents=old['parent_ids'], rows=rows,
        physical_queries=0, trained=False, only_cached_training_positions_loaded=True, development_positions_loaded=False,
        reference_coordinates_loaded=False, scientific_submission_ready=False,
        limitation='The physical component can itself explain improved support. This is a fixed cold-geometry screen, not an energy or molecular sampling generalization test.')
    (args.out/'results.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
