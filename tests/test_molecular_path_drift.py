from types import SimpleNamespace

import torch
from flowmol.models.ctmc_vector_field import CTMCVectorField
from flowmol.models.interpolant_scheduler import InterpolantScheduler

from test_clamped_density import graph_batch
from cfm_mol.electronic_conditioning import attach_electronic_state,patch_electronic_conditioning
from cfm_mol.molecular_path_drift import MolecularPathDrift
from cfm_mol.path_balance import fixed_path_log_factors


def test_real_flowmol_time_batch_matches_separate_observations_and_parameter_gradients():
    torch.manual_seed(9066)
    field=CTMCVectorField(n_atom_types=3,canonical_feat_order=['x','a','c','e'],
        interpolant_scheduler=InterpolantScheduler(['x','a','c','e'],schedule_type='linear'),
        n_vec_channels=4,n_hidden_scalars=8,n_hidden_edge_feats=8,n_molecule_updates=1,
        convs_per_update=2,n_message_gvps=1,n_update_gvps=1,n_expansion_gvps=1,
        rbf_dim=4,self_conditioning=True)
    model=SimpleNamespace(vector_field=field.train());patch_electronic_conditioning(model)
    base,_,_=graph_batch((3,),dtype=torch.float32)
    attach_electronic_state(base,0,2,.025852,atomic_numbers=torch.ones(3,dtype=torch.long))
    with torch.no_grad():field.electronic_embedding[-1].weight.normal_(std=.1)
    saved={k:v.clone() for k,v in base.ndata.items()}
    drift=MolecularPathDrift(model,base)
    z=torch.randn(6,6,dtype=torch.float64);t=torch.tensor([0.,.2,.4,.6,.8,1.])
    actual=drift(z,t)
    expected=torch.cat([drift(row[None],clock) for row,clock in zip(z,t)])
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
    parameters=tuple(field.parameters())
    left=torch.autograd.grad(actual.square().sum(),parameters,allow_unused=True)
    right=torch.autograd.grad(expected.square().sum(),parameters,allow_unused=True)
    for a,b in zip(left,right):
        if a is None:assert b is None
        else:torch.testing.assert_close(a,b,rtol=5e-4,atol=1e-5)
    # Observed paths include distinct times within each molecule-major block.
    states=torch.randn(2,4,6,dtype=torch.float64)
    kwargs=dict(terminal_std=.51,mean_parameterization='native',noise_annealing_power=.5,max_drift_norm=20.)
    def sequential(z,t):return torch.cat([drift(row[None],clock) for row,clock in zip(z,t)])
    batched=fixed_path_log_factors(states,drift,drift,[0.,.2,.6,1.],.2,**kwargs)
    separate=fixed_path_log_factors(states,sequential,sequential,[0.,.2,.6,1.],.2,**kwargs)
    for a,b in zip(batched,separate):torch.testing.assert_close(a,b,rtol=2e-6,atol=1e-3)
    a=torch.autograd.grad((batched[1]-batched[2]).mean(),parameters,allow_unused=True)
    b=torch.autograd.grad((separate[1]-separate[2]).mean(),parameters,allow_unused=True)
    for l,r in zip(a,b):
        if l is None:assert r is None
        else:torch.testing.assert_close(l,r,rtol=5e-4,atol=2e-3)
    assert field.training and field.self_conditioning
    assert len(field.scalar_embedding._forward_hooks)==0
    for key,value in saved.items():torch.testing.assert_close(base.ndata[key],value,rtol=0,atol=0)
