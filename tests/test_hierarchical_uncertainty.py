import pytest
import torch
from cfm_mol.hierarchical_uncertainty import hierarchical_paired_mean


def test_repeated_parents_do_not_create_independent_compositions():
    # Perfectly correlated parents within each of two opposite compositions.
    x=torch.tensor([[-1.]*16,[1.]*16]).expand(2,-1,-1)
    result=hierarchical_paired_mean(x,generator=torch.Generator().manual_seed(2))
    assert result['mean']==0.
    assert result['hierarchical_95_percent_interval']==[-1.,1.]
    assert result['independent_compositions']==2


def test_algorithm_replicas_are_paired_and_missing_endpoints_fail_closed():
    x=torch.ones(2,3,4,dtype=torch.float64);x[0]*=-1
    result=hierarchical_paired_mean(x,generator=torch.Generator().manual_seed(3))
    assert result['hierarchical_95_percent_interval']==[0.,0.]
    with pytest.raises(ValueError):
        hierarchical_paired_mean(x[:, :1],generator=torch.Generator())
    x[0,0,0]=torch.nan
    with pytest.raises(ValueError):
        hierarchical_paired_mean(x,generator=torch.Generator())
