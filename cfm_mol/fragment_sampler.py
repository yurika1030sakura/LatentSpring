"""Exact physical fragment-MH kernel with failed proposals and costs retained."""
import math
import torch
from cfm_mol.fragment_exchange import (fragment_exchange_actions, inverse_fragment_action,
    physical_fragment_proposal)
from cfm_mol.joint_chemical_geometry import distinct_anchor_actions


def fragment_method_actions(state, numbers, method, cap):
    if method == 'original_terminal':
        return [dict(roots=a, fragments=((a[0],), (a[1],)))
                for a in distinct_anchor_actions(numbers, state['graph']['bond_orders'])]
    if method not in ('all_singletons', 'fragments4'):
        raise ValueError('Unknown physical fragment method')
    return fragment_exchange_actions(state['graph']['bond_orders'], max_fragment_atoms=1 if method == 'all_singletons' else cap)


@torch.no_grad()
def fragment_transition(target, states, *, method, generator, phase, max_fragment_atoms=4,
                        radial_width=.05, concentration=10.):
    rows, candidates = [], []
    backbone = torch.tensor([z not in (1, 9, 17, 35, 53) for z in target.numbers])
    for old in states:
        actions = fragment_method_actions(old, target.numbers, method, max_fragment_atoms)
        row = dict(kind='fragment_exchange', method=method, phase=phase, old_state_id=old['state_id'],
            new_state_id=-1, forward_count=len(actions), valid=False, accepted=False)
        candidate = None
        if not actions:
            row['rejection_reason'] = 'No eligible fragment pair'
        else:
            choice = int(torch.randint(len(actions), (1,), generator=generator))
            action = actions[choice]
            y, proposal = physical_fragment_proposal(old['positions'], old['graph']['bond_orders'],
                target.radii, action, generator=generator, radial_width=radial_width, concentration=concentration)
            row.update(choice_index=choice, action=action, inverse_action=inverse_fragment_action(action),
                       proposal=proposal, multiatom=max(map(len, action['fragments'])) > 1)
            if y is None:
                row['rejection_reason'] = proposal['reason']
            else:
                row['proposal_positions'] = y
                try:
                    candidate = target.coordinate_state(y)
                    if not torch.equal(candidate['graph']['bond_orders'], proposal['desired_bonds']):
                        raise ValueError('Endpoint differs from desired fragment graph')
                    reverse_actions = fragment_method_actions(candidate, target.numbers, method, max_fragment_atoms)
                    if row['inverse_action'] not in reverse_actions:
                        raise ValueError('Inverse fragment pair is ineligible')
                    row.update(valid=True, reverse_count=len(reverse_actions),
                        action_log_ratio=math.log(len(actions)/len(reverse_actions)),
                        backbone_changed=not torch.equal(old['graph']['bond_orders'][backbone][:, backbone],
                            candidate['graph']['bond_orders'][backbone][:, backbone]),
                        constitution_changed=old['graph']['connectivity_smiles'] != candidate['graph']['connectivity_smiles'])
                except ValueError as exc:
                    candidate = None
                    row['rejection_reason'] = str(exc)
        rows.append(row)
        candidates.append(candidate)
    target.evaluate([s for s in candidates if s is not None], phase=phase)
    log_u = torch.rand(len(states), dtype=torch.float64, generator=generator).log()
    updated = list(states)
    for index, (old, new, row) in enumerate(zip(states, candidates, rows)):
        row['log_uniform'] = float(log_u[index])
        if new is None:
            continue
        physical = -float(new['potential_eV']-old['potential_eV'])/target.kT
        proposal = float(row['proposal']['log_reverse']-row['proposal']['log_forward'])
        ratio = physical+proposal+row['action_log_ratio']
        take = float(log_u[index]) < min(0., ratio)
        row.update(new_state_id=new['state_id'], physical_log_ratio=physical,
                   auxiliary_log_ratio=proposal, log_acceptance_ratio=ratio, accepted=take)
        if take:
            updated[index] = new
    return updated, rows
