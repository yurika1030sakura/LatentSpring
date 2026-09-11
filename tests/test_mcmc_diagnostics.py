import numpy as np
from cfm_mol.mcmc_diagnostics import diagnose_chains


def test_independent_and_stuck_chains_are_distinguished():
    rng=np.random.default_rng(9871)
    independent=diagnose_chains(rng.normal(size=(16,2048)))
    assert independent['split_rank_rhat']<1.01
    assert independent['ess']>20000
    stuck=np.repeat(np.arange(16)[:,None],2048,axis=1)+rng.normal(0,.001,(16,2048))
    diagnostic=diagnose_chains(stuck)
    assert diagnostic['split_rank_rhat']>1.2 and diagnostic['ess']<32
    constant=diagnose_chains(np.zeros((4,32)))
    assert constant['ess'] is None and constant['split_rank_rhat'] is None
