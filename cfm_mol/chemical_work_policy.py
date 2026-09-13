"""Exact finite-catalogue selection correction for deterministic chemical edits.

Only the selected endpoint needs a physical query. Its freshly computed force
may inform the reverse *force control*, never the forward decision at the source.
"""
import copy
import torch
from cfm_mol.chemical_work import harmonic_change, informed_log_prob
from cfm_mol.edit_bridge_sampler import ZeroBridgeField, propose_edit, finish_edit
from cfm_mol.joint_chemical_geometry import distinct_anchor_actions


@torch.no_grad()
def catalogue(target, source):
    state = dict(source)
    state.setdefault('state_id', -1)
    valid, failed = [], []
    for action in distinct_anchor_actions(target.numbers, state['graph']['bond_orders']):
        candidate, record = propose_edit(target, state, action, torch.zeros_like(state['positions']), 0, 0,
            method='bare_edit', field=ZeroBridgeField(),
            bridge_options=dict(steps_per_side=0, kick_step=.2, drift_step=.02), arc_options={})
        if candidate is None:
            failed.append(record)
        else:
            valid.append(dict(action=action, candidate=candidate, record=record))
    return dict(valid=valid, failed=failed)


@torch.no_grad()
def policy(target, source, candidates, model, uniform_fraction=.1):
    if not candidates['valid']:
        return dict(log_probability=torch.empty(0, dtype=torch.float64), work=torch.empty(0, dtype=torch.float64))
    rows = candidates['valid']
    x = source['positions'][None].expand(len(rows), -1, -1)
    y = torch.stack([row['candidate']['positions'] for row in rows])
    volume = x.new_tensor([row['record']['log_volume'] for row in rows])
    if isinstance(model, str):
        if model == 'uniform':
            work = x.new_zeros(len(rows)); uniform_fraction = 1.
        elif model == 'force':
            work = -(source['force_eV_A'][None]*(y-x)).sum((1, 2))+harmonic_change(x, y, target.restraint)
        elif model == 'restraint':
            work = harmonic_change(x, y, target.restraint)
        else:
            raise ValueError('Unknown physical policy control')
    else:
        bonds = source['graph']['bond_orders'][None].expand(len(rows), -1, -1)
        new_bonds = torch.stack([row['candidate']['graph']['bond_orders'] for row in rows])
        numbers = torch.tensor(target.numbers, dtype=torch.long, device=x.device)
        electronic = x.new_tensor([source['charge'], source['spin_multiplicity'], target.kT]).expand(len(rows), -1)
        active = torch.tensor([row['action'][:2] for row in rows], dtype=torch.long, device=x.device)
        work = model(x, y, bonds, new_bonds, numbers, electronic, active)
    return dict(log_probability=informed_log_prob(work, volume, target.kT, uniform_fraction), work=work)


def select(candidates, probability, uniform):
    """Inverse CDF permits common random numbers across all policies."""
    if not 0 <= uniform < 1:
        raise ValueError('Selection variate must lie in [0,1)')
    if not candidates['valid']:
        return None, None
    cdf = probability['log_probability'].exp().cumsum(0)
    index = min(int(torch.searchsorted(cdf, cdf.new_tensor(uniform), right=True)), len(cdf)-1)
    item = candidates['valid'][index]
    record = copy.deepcopy(item['record'])
    record.update(forward_valid_count=len(cdf), forward_failed_count=len(candidates['failed']),
                  forward_selected_index=index, forward_log_probability=float(probability['log_probability'][index]),
                  forward_predicted_work_eV=float(probability['work'][index]))
    return copy.deepcopy(item['candidate']), record


@torch.no_grad()
def finish(target, source, candidate, record, model, log_uniform, uniform_fraction=.1):
    """Call after the selected candidate was physically evaluated, even if rejected."""
    reverse = catalogue(target, candidate)
    inverse = tuple(record['inverse_action'])
    indices = [i for i, row in enumerate(reverse['valid']) if tuple(row['action']) == inverse]
    if len(indices) != 1:
        raise ValueError('Inverse action absent or duplicated in reverse valid catalogue')
    index = indices[0]
    torch.testing.assert_close(reverse['valid'][index]['candidate']['positions'], source['positions'], atol=1e-9, rtol=0)
    backward = policy(target, candidate, reverse, model, uniform_fraction)
    record.update(reverse_valid_count=len(reverse['valid']), reverse_failed_count=len(reverse['failed']),
                  reverse_selected_index=index, reverse_log_probability=float(backward['log_probability'][index]),
                  reverse_predicted_work_eV=float(backward['work'][index]))
    record['action_log_ratio'] = record['reverse_log_probability']-record['forward_log_probability']
    return finish_edit(target, source, candidate, record, log_uniform)
