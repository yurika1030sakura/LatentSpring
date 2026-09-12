#!/usr/bin/env python3
"""Geometry-only transfer screen on the next eligible development composition.

Use the first32 chemically supported TRAINING parents of the17-atom condition4.
No optimization, development-parent selection, reference coordinates or new
physical queries. Frozen condition0 models have unseen element embeddings here.
This is a numerical/support screen, not a generalization performance experiment.
"""
import argparse
import json
from pathlib import Path
import torch
from cfm_mol.entropy_source import load_entropy_source
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.chemical_moves import exchange_terminal_sites
from cfm_mol.chemical_path_guide import exchanged_bond_graph
from cfm_mol.joint_chemical_geometry import joint_geometry_proposal, distinct_anchor_actions
from scripts.research.evaluate_joint_chemical import load_model, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ['source', 'models', 'table', 'out']:
        parser.add_argument('--'+arg, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    protocol = json.loads((root/'research/evidence/joint_chemical_protocol_v1.json').read_text())
    physical = json.loads((root/'research/evidence/parity_training_protocol_v1.json').read_text())
    header = json.loads((args.table/'results.json').read_text())
    assert sha(args.table/'results.json') == protocol['table_results_sha256']
    census_path = root/'research/evidence/chemical_source_panel_audit_v2.json'
    census = json.loads(census_path.read_text())
    eligible = [row['index'] for row in census['conditions'] if row['index'] > 0 and row['streams']['training']['with_exchange']]
    assert eligible == [4]
    source = load_entropy_source(args.source, 4,
        protocol_path=root/'research/evidence/species_breadth_source_protocol_v2.json',
        manifest_path=root/'research/evidence/development_panel_v1.json')
    ids = [p['parent_id'] for p in census['conditions'][4]['streams']['training']['supported_parents']][:32]
    target = ChemicalTarget(None, source['condition'], physical['kT_eV'], physical['restraint_eV_A2'])
    states = [target.coordinate_state(source['training']['positions'][i]) for i in ids]
    numbers = torch.tensor(target.numbers, dtype=torch.long)
    electronic = torch.tensor([target.condition['charge'], target.condition['spin_multiplicity'], target.kT], dtype=torch.float64)
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out/'results.json'
    if output.exists():
        raise FileExistsError(output)
    rows, records = [], []
    for method in protocol['methods']:
        for replica in [0, 1]:
            model, metadata = load_model(args.models, method, replica, protocol, header['artifacts']['training'])
            rng = torch.Generator().manual_seed(10831+replica)
            totals = dict(method=method, replica=replica, attempts=0, valid=0, no_action=0, validator_errors=0,
                          model_sha256=metadata['checkpoint_sha256'] if metadata else None)
            for parent, old in zip(ids, states):
                actions = distinct_anchor_actions(numbers, old['graph']['bond_orders'])
                for trial in range(4):
                    row = dict(method=method, replica=replica, parent_id=parent, trial=trial, valid=False)
                    totals['attempts'] += 1
                    if not actions:
                        totals['no_action'] += 1
                        records.append(row)
                        continue
                    index = int(torch.randint(len(actions), (1,), generator=rng))
                    action = actions[index]
                    i, j, k, l = action
                    desired = exchanged_bond_graph(old['graph']['bond_orders'], action)
                    row.update(action=action, choice_index=index)
                    if method == 'deterministic':
                        y, volume, _ = exchange_terminal_sites(old['positions'], target.radii, action)
                        row['log_volume'] = volume
                    else:
                        order = int(torch.randint(2, (1,), generator=rng))
                        y, q, forward = joint_geometry_proposal(old['positions'], old['graph']['bond_orders'],
                            numbers, electronic, target.radii, action, kind=method, order=order, generator=rng, model=model,
                            radial_width=protocol['radial_width'], site_concentration=protocol['site_concentration'])
                        row.update(order=order, forward=forward, forward_q=q)
                    row['positions'] = y
                    try:
                        new = target.coordinate_state(y)
                        if not torch.equal(new['graph']['bond_orders'], desired):
                            raise ValueError('Different endpoint bond matrix')
                        inverse = (i, j, l, k)
                        if inverse not in distinct_anchor_actions(numbers, desired):
                            raise ValueError('Inverse ineligible')
                        if method != 'deterministic':
                            recovered, qr, reverse = joint_geometry_proposal(y, desired, numbers, electronic, target.radii,
                                inverse, kind=method, order=order, observed=old['positions'], model=model,
                                radial_width=protocol['radial_width'], site_concentration=protocol['site_concentration'])
                            torch.testing.assert_close(recovered, old['positions'], atol=1e-9, rtol=1e-9)
                            assert torch.isfinite(qr)
                            row.update(reverse=reverse, reverse_q=qr)
                        row['valid'] = True
                        totals['valid'] += 1
                    except (ValueError, IndexError, RuntimeError) as exc:
                        row.update(reason=str(exc), error_type=type(exc).__name__)
                        totals['validator_errors'] += not isinstance(exc, ValueError)
                    records.append(row)
            totals['generator_state'] = rng.get_state().tolist()
            rows.append(totals)
    torch.save(dict(states=states, parent_ids=ids, records=records), args.out/'trace.pt')
    result = dict(complete=True, condition_index=4, n_atoms=len(numbers), numbers=target.numbers,
        parents=ids, source_results_sha256=source['source_results_sha256'], training_source_sha256=source['training_sha256'],
        census_sha256=sha(census_path), original_training_elements=[1, 6, 9, 16], unseen_elements=[8, 15, 17],
        source_provenance_loader_reads_both_streams=True, only_training_positions_used_for_proposals=True,
        energy_outcomes_used=False, physical_queries=0, reference_coordinates_loaded=False, trained=False,
        trace_sha256=sha(args.out/'trace.pt'), rows=rows, scientific_submission_ready=False,
        interpretation='Numerical density and graph-support screen on cold generated training coordinates. Conditional acceptance, energy distribution and learned molecular transfer performance are unmeasured.')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({**result, 'rows': [{k: v for k, v in row.items() if k != 'generator_state'} for row in rows]}, indent=2))


if __name__ == '__main__':
    main()
