#!/usr/bin/env python3
"""Geometry-only pendant-fragment coverage on the prescribed eight compositions."""
import argparse
import json
from pathlib import Path
import torch
from cfm_mol.entropy_source import load_entropy_source
from cfm_mol.chemical_sampler import ChemicalTarget
from cfm_mol.joint_chemical_geometry import distinct_anchor_actions
from cfm_mol.fragment_exchange import (fragment_exchange_actions, inverse_fragment_action,
    fragment_bond_graph, physical_fragment_proposal)
from scripts.research.evaluate_chemical_policy import sha


def actions_for(state, numbers, method, cap):
    if method == 'original_terminal':
        return [dict(roots=a, fragments=((a[0],), (a[1],))) for a in distinct_anchor_actions(numbers, state['graph']['bond_orders'])]
    return fragment_exchange_actions(state['graph']['bond_orders'], max_fragment_atoms=1 if method == 'all_singletons' else cap)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    pp = root/'research/evidence/fragment_coverage_protocol_v1.json'
    protocol = json.loads(pp.read_text())
    census_path = root/'research/evidence/chemical_source_panel_audit_v2.json'
    assert sha(census_path) == protocol['census_sha256']
    census = json.loads(census_path.read_text())
    physical = json.loads((root/'research/evidence/parity_training_protocol_v1.json').read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'results.json').exists():
        raise FileExistsError(args.out/'results.json')
    records, conditions = [], []
    for index in range(8):
        source = load_entropy_source(args.source, index,
            protocol_path=root/'research/evidence/species_breadth_source_protocol_v2.json',
            manifest_path=root/'research/evidence/development_panel_v1.json')
        row = census['conditions'][index]
        assert row['source_results_sha256'] == source['source_results_sha256']
        parents = [p['parent_id'] for p in row['streams']['training']['supported_parents']][:protocol['parents_per_condition']]
        target = ChemicalTarget(None, source['condition'], physical['kT_eV'], physical['restraint_eV_A2'])
        states = [target.coordinate_state(source['training']['positions'][i]) for i in parents]
        # A fixed element mask distinguishes changes to C/N/O/P/S/etc connectivity
        # from mere relocation of terminal halogens. This is a labelled diagnostic;
        # canonical connectivity changes are reported separately.
        backbone = torch.tensor([z not in (1, 9, 17, 35, 53) for z in target.numbers])
        report = dict(index=index, condition=source['condition'], source_results_sha256=source['source_results_sha256'],
            training_source_sha256=source['training_sha256'], source_attempts=4096,
            chemically_supported_training=row['streams']['training']['chemically_supported'],
            source_validator_errors=row['streams']['training']['validator_algorithm_errors'],
            parents=parents, methods=[])
        for method in protocol['methods']:
            generator = torch.Generator().manual_seed(protocol['seed']+index)
            summary = dict(method=method, sampled_parents=len(parents), parents_with_actions=0,
                parents_with_multiatom_actions=0, total_eligible_actions=0, attempts=0, no_action=0,
                proposed_multiatom=0, proposed_backbone_changes=0, map_failures=0,
                graph_rejections=0, validator_errors=0, supported=0,
                supported_multiatom=0, supported_backbone_changes=0, supported_constitution_changes=0)
            for parent, state in zip(parents, states):
                actions = actions_for(state, target.numbers, method, protocol['max_fragment_atoms'])
                summary['parents_with_actions'] += bool(actions)
                summary['parents_with_multiatom_actions'] += any(max(map(len, a['fragments'])) > 1 for a in actions)
                summary['total_eligible_actions'] += len(actions)
                for trial in range(protocol['draws_per_parent']):
                    summary['attempts'] += 1
                    record = dict(condition_index=index, method=method, parent_id=parent, trial=trial, valid=False)
                    if not actions:
                        summary['no_action'] += 1
                        records.append(record)
                        continue
                    choice = int(torch.randint(len(actions), (1,), generator=generator))
                    action = actions[choice]
                    desired = fragment_bond_graph(state['graph']['bond_orders'], action)
                    multiatom = max(map(len, action['fragments'])) > 1
                    changed = not torch.equal(desired[backbone][:, backbone], state['graph']['bond_orders'][backbone][:, backbone])
                    summary['proposed_multiatom'] += multiatom
                    summary['proposed_backbone_changes'] += changed
                    y, proposal = physical_fragment_proposal(state['positions'], state['graph']['bond_orders'],
                        target.radii, action, generator=generator, radial_width=protocol['radial_width'], concentration=protocol['site_concentration'])
                    record.update(action=action, action_index=choice, forward_action_count=len(actions),
                                  multiatom=multiatom, backbone_changed=changed, proposal=proposal)
                    if y is None:
                        summary['map_failures'] += 1
                        records.append(record)
                        continue
                    record['positions'] = y
                    try:
                        new = target.coordinate_state(y)
                        if not torch.equal(new['graph']['bond_orders'], desired):
                            raise ValueError('Endpoint differs from desired fragment graph')
                    except (ValueError, IndexError, RuntimeError) as exc:
                        record.update(reason=str(exc), error_type=type(exc).__name__)
                        summary['graph_rejections'] += isinstance(exc, ValueError)
                        summary['validator_errors'] += not isinstance(exc, ValueError)
                        records.append(record)
                        continue
                    inverse = inverse_fragment_action(action)
                    reverse_actions = actions_for(new, target.numbers, method, protocol['max_fragment_atoms'])
                    assert inverse in reverse_actions
                    constitution = new['graph']['connectivity_smiles'] != state['graph']['connectivity_smiles']
                    record.update(valid=True, reverse_action_count=len(reverse_actions),
                        original_smiles=state['graph']['connectivity_smiles'], new_smiles=new['graph']['connectivity_smiles'])
                    records.append(record)
                    summary['supported'] += 1
                    summary['supported_multiatom'] += multiatom
                    summary['supported_backbone_changes'] += changed
                    summary['supported_constitution_changes'] += constitution
            report['methods'].append(summary)
            print(json.dumps(dict(index=index, **summary)), flush=True)
        conditions.append(report)
    torch.save(dict(records=records, protocol_sha256=sha(pp)), args.out/'trace.pt')
    result = dict(complete=True, protocol_sha256=sha(pp), trace_sha256=sha(args.out/'trace.pt'), conditions=conditions,
        physical_queries=0, energy_outcomes_used=False, source_loader_reads_both_streams=True,
        only_training_positions_used=True, reference_coordinates_loaded=False, scientific_submission_ready=False,
        interpretation='Physical-map and graph-support coverage only. No energy, acceptance, equilibrium or learned-method advantage follows.')
    (args.out/'results.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
