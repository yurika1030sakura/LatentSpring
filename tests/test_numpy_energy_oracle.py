import sys

import numpy as np
import pytest

from cfm_mol.numpy_energy_oracle import NumpyEnergyOracle


@pytest.fixture
def worker(tmp_path):
    path=tmp_path/'oracle.py'
    path.write_text('''import json,sys
print('startup noise\\nBGFM_ORACLE_JSON '+json.dumps({'ready':True}),flush=True)
for line in sys.stdin:
    p=json.loads(line); x=p['positions']; n=len(x)
    if x[0][0][0] < -1e6:
        result={'ok':False,'error':'physical worker failure','attempted_evaluations':1}
    else:
        result={'ok':True,'energies_eV':[sum(v*v for atom in r for v in atom)/2 for r in x],
                'forces_eV_A':[[[-v for v in atom] for atom in r] for r in x],
                'attempted_evaluations':n}
    print('worker noise\\nBGFM_ORACLE_JSON '+json.dumps(result),flush=True)
''')
    return path


def test_buffered_protocol_and_chunk_counts(worker):
    x=np.arange(90,dtype=np.float64).reshape(10,3,3)
    with NumpyEnergyOracle(sys.executable,worker,'unused',numbers=[6]*3,charge=0,spin_multiplicity=1,timeout_seconds=3.) as oracle:
        energy,force=oracle.evaluate_chunked(x,max_request=4)
        np.testing.assert_allclose(energy,.5*(x*x).sum((1,2)))
        np.testing.assert_allclose(force,-x)
        assert oracle.evaluated==oracle.requested_evaluations==10
        assert oracle.callback_requests==3


def test_failed_worker_keeps_requested_and_acknowledged_counts(worker):
    x=np.full((4,3,3),-2e6)
    with NumpyEnergyOracle(sys.executable,worker,'unused',numbers=[6]*3,charge=0,spin_multiplicity=1) as oracle:
        with pytest.raises(RuntimeError,match='physical worker failure'):oracle.evaluate(x)
        assert oracle.requested_evaluations==4 and oracle.evaluated==1
        with pytest.raises(ValueError):oracle.evaluate(np.ones((2,4,3)))
        assert oracle.requested_evaluations==4
