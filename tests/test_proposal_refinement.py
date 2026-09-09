import torch
from cfm_mol.tempered_smc import DensityValue
from cfm_mol.proposal_refinement import refine_centers


def test_refinement_decreases_energy_for_every_center():
    x=torch.tensor([[1.,-2.],[-3.,4.]],dtype=torch.float64)
    target=lambda z:DensityValue(-z.square().sum(-1)/2,-z)
    out,info=refine_centers(x,target,steps=30,max_step=.2)
    assert torch.all(out.square().sum(-1)<x.square().sum(-1))
    assert info['minimum_log_target_gain']>=0
    assert info['target_evaluations']==62


def test_wrong_direction_is_rejected_without_dropping_centers():
    x=torch.tensor([[1.,2.],[-3.,4.]],dtype=torch.float64)
    wrong=lambda z:DensityValue(-z.square().sum(-1)/2,z)
    out,info=refine_centers(x,wrong,steps=10)
    torch.testing.assert_close(out,x,rtol=0,atol=0)
    assert all(row['accepted_centers']==0 for row in info['history'])
