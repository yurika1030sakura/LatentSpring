from copy import deepcopy

import pytest
import torch

from cfm_mol.work_resume import validate_resume_recipe,restore_work_optimizer


def recipe():
    return dict(mode='joint',batch=16,path_steps=16,lr=1e-5,noise=.2,kT=.025852,
        restraint=.1,seed=9051,reference_kernel='gaussian',mean_parameterization='native',
        neutralize_temperature_input=True,noise_annealing_power=.5,prior_std=1.,
        max_drift_per_sqrt_dimension=20.,steps=500)


def test_continuation_accepts_old_mean_work_recipe_but_rejects_changed_physics_or_objective():
    old=recipe();source={'complete':True,'configuration':old};state={'proposal_protocol':deepcopy(old),'global_step':500}
    requested={**old,'steps':1500,'objective':'mean_work'}
    assert validate_resume_recipe(source,state,requested)==500
    for key,value in [('kT',1.),('mode','forward_energy_only'),('objective','log_variance'),('seed',9052),('batch',32),('precision_kind','learned'),('precision_strength',8.)]:
        with pytest.raises(ValueError,match=key):validate_resume_recipe(source,state,{**requested,key:value})
    with pytest.raises(ValueError,match='total steps'):validate_resume_recipe(source,state,old)
    with pytest.raises(ValueError,match='diagnostic'):
        validate_resume_recipe({**source,'gradient_diagnostics':{'weights_updated':False}},state,requested)


def test_optimizer_resume_preserves_adam_moments_and_global_count():
    parameter=torch.nn.Parameter(torch.tensor([.1,.5]))
    optimizer=torch.optim.AdamW([parameter],lr=1e-5)
    for i in range(3):
        optimizer.zero_grad();parameter.square().sum().backward();optimizer.step()
    state=deepcopy(optimizer.state_dict());new_parameter=torch.nn.Parameter(parameter.detach().clone())
    restored=torch.optim.AdamW([new_parameter],lr=1e-5);restore_work_optimizer(restored,state,3)
    optimizer.zero_grad();restored.zero_grad()
    parameter.square().sum().backward();new_parameter.square().sum().backward()
    optimizer.step();restored.step()
    torch.testing.assert_close(new_parameter,parameter,rtol=0,atol=0)
    with pytest.raises(ValueError,match='update counts'):restore_work_optimizer(restored,state,500)
