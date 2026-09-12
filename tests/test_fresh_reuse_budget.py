import pytest
import torch
from scripts.research.fresh_reuse import run_budget, update_query_counts


class Target:
    kT = 1.

    def __init__(self):
        self.oracle = type('Oracle', (), {'evaluated': 0})()
        self.states = []

    def coordinate_state(self, x):
        return {'parent': int(x), 'positions': x}

    def evaluate(self, states, *, phase):
        self.oracle.evaluated += 2*len(states)
        for state in states:
            state['state_id'] = len(self.states)
            self.states.append(state)
        return states

    def transition(self, states, *, phase, **kwargs):
        step = int(phase.rsplit('_', 1)[1])
        rows = []
        for state in states:
            # Parent 1 never produces a supported proposal; parent 2 does so
            # only on even steps. Every supported proposal is MH rejected.
            valid = state['parent'] == 0 or (state['parent'] == 2 and step % 2 == 0)
            rows.append(dict(valid=valid, accepted=False))
            self.oracle.evaluated += 2*int(valid)
        return states, rows


def test_independent_parent_budgets_charge_rejected_candidates_and_keep_censored_parent():
    target = Target()
    progress = {}
    protocol = dict(query_caps={'site': 8}, evaluation_seeds=[17], scale_seeds=[19], maximum_microsteps=9)
    shared = dict(local_scales=[.1, .03], schedule=['local'])
    states = run_budget(target, [torch.tensor(i) for i in range(3)], [100,200,300], None,
        'site', 0, 0, protocol, shared, progress)
    assert progress['final_queries_per_parent'] == [8,2,8]
    assert progress['query_cap_reached'] == [True,False,True]
    assert target.oracle.evaluated == 18
    assert len(progress['rounds']) == 9
    assert progress['rounds'][-1]['active_indices'] == [1]
    assert [s['state_id'] for s in states] == [0,1,2]
    assert all(max(counts) <= 8 for counts in progress['query_count_history'])


def test_ledger_rejects_unaccounted_queries_and_invalid_acceptance():
    with pytest.raises(ValueError, match='oracle calls'):
        update_query_counts([2], [0], [dict(valid=True, accepted=False)], 0, 4)
    with pytest.raises(ValueError, match='Unsupported'):
        update_query_counts([2], [0], [dict(valid=False, accepted=True)], 0, 4)
    with pytest.raises(ValueError, match='cap exceeded'):
        update_query_counts([4], [0], [dict(valid=True, accepted=False)], 2, 4)


def test_initialization_consumes_budget_without_additional_proposals():
    target, progress = Target(), {}
    protocol = dict(query_caps={'site': 2}, evaluation_seeds=[17], scale_seeds=[19], maximum_microsteps=9)
    run_budget(target, [torch.tensor(0)], [100], None, 'site', 0, 0, protocol,
        dict(local_scales=[.1], schedule=['local']), progress)
    assert progress['rounds'] == [] and target.oracle.evaluated == 2
    assert progress['query_cap_reached'] == [True]
