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


def test_tensor_oracle_buffered_worker_values_cost_and_elapsed_time(tmp_path):
    import sys
    worker=tmp_path/'worker.py'
    worker.write_text("""import json,sys
print('noise\\nBGFM_ORACLE_JSON '+json.dumps({'ready':True}),flush=True)
for line in sys.stdin:
    data=json.loads(line);x=data['positions'];n=len(x)
    result={'ok':True,'energies_eV':[sum(v*v for atom in r for v in atom)/2 for r in x],
            'forces_eV_A':[[[-v for v in atom] for atom in r] for r in x],'attempted_evaluations':n}
    print('noise\\nBGFM_ORACLE_JSON '+json.dumps(result),flush=True)
""")
    x=torch.arange(18,dtype=torch.float64).reshape(3,2,3)
    with EnergyOracle(sys.executable,worker,'unused',numbers=[6,6],charge=0,spin_multiplicity=1,timeout_seconds=2.) as oracle:
        assert oracle.evaluation_seconds==0
        energy,force=oracle.evaluate_chunked(x,max_request=2)
        torch.testing.assert_close(energy,.5*x.square().sum((1,2)))
        torch.testing.assert_close(force,-x)
        assert oracle.evaluated==oracle.requested_evaluations==3
        assert oracle.evaluation_seconds>0
