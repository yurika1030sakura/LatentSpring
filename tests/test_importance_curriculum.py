import pytest
import torch

from cfm_mol.importance_curriculum import weight_controls


def test_maximal_linear_step_and_power_control_match_requested_ess():
    logs=torch.tensor([0.,-2.,-4.,-8.,-12.,-16.,-20.,-25.],dtype=torch.float64)
    weights,info=weight_controls(logs,.5)
    assert 0<info['linear_target_fraction']<1
    assert info['ess']['linear']==pytest.approx(4.,abs=1e-12)
    assert info['ess']['power']==pytest.approx(4.,abs=1e-12)
    eta=info['linear_target_fraction']+1e-5;larger=(1-eta)*weights['uniform']+eta*weights['full']
    assert float(larger.square().sum().reciprocal())<4.
    shifted,other=weight_controls(logs+10000.,.5)
    for key in weights:torch.testing.assert_close(weights[key],shifted[key],rtol=1e-12,atol=1e-12)
    assert other['linear_target_fraction']==pytest.approx(info['linear_target_fraction'])


def test_good_weights_use_full_target_and_uniform_weights_are_well_defined():
    weights,info=weight_controls(torch.zeros(8),.5)
    assert info['linear_target_fraction']==info['power_exponent']==1.
    torch.testing.assert_close(weights['linear'],weights['uniform'])


def test_latent_power_weights_can_move_an_already_correct_endpoint_distribution():
    # Rows are endpoints; columns are latent states. Q and R have exactly the
    # same endpoint marginal, but different conditionals in the second row.
    q=torch.tensor([[.25,.25],[.495,.005]],dtype=torch.float64)
    r=torch.tensor([[.25,.25],[.005,.495]],dtype=torch.float64)
    importance=r/q
    for eta in [.01,.3,1.]:
        mixed=q*((1-eta)+eta*importance)
        torch.testing.assert_close(mixed.sum(1),torch.tensor([.5,.5],dtype=torch.float64),atol=1e-15,rtol=0)
    powered=q*importance.sqrt();powered/=powered.sum()
    assert float(powered.sum(1)[1])<.17
    # This is an exact finite-state population calculation, not a noisy sample
    # experiment or a claim that a finite empirical mixture is calibrated.
