import pytest
import torch

from cfm_mol.energy_oracle import EnergyOracle


def fake_oracle(fail_request=None):
    oracle = object.__new__(EnergyOracle)
    oracle.evaluated = 0
    oracle.requested_evaluations = 0
    oracle.sizes = []
    def evaluate(x):
        oracle.sizes.append(len(x))
        oracle.requested_evaluations += len(x)
        if len(oracle.sizes) == fail_request:
            raise TimeoutError('No acknowledgement')
        oracle.evaluated += len(x)
        return x.square().sum((1, 2)), -2*x
    oracle.evaluate = evaluate
    return oracle


def test_chunked_physical_values_order_and_exact_accounting():
    x = torch.arange(129*6, dtype=torch.float64).reshape(129, 2, 3)
    oracle = fake_oracle()
    energy, force = oracle.evaluate_chunked(x)
    torch.testing.assert_close(energy, x.square().sum((1, 2)))
    torch.testing.assert_close(force, -2*x)
    assert oracle.sizes == [32, 32, 32, 32, 1]
    assert oracle.evaluated == oracle.requested_evaluations == 129


def test_timeout_keeps_unacknowledged_queries_distinct_from_zero_cost():
    oracle = fake_oracle(fail_request=2)
    with pytest.raises(TimeoutError):
        oracle.evaluate_chunked(torch.zeros(129, 2, 3))
    assert oracle.evaluated == 32
    assert oracle.requested_evaluations == 64
    assert oracle.sizes == [32, 32]
