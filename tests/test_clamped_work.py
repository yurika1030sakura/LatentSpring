import math
from types import SimpleNamespace

import pytest
import torch

from test_clamped_density import graph_batch,LinearHead
from cfm_mol.clamped_work import sample_clamped_work_proposal


def test_zero_field_brownian_factor_intrinsic_prior_and_graph_restoration():
    graph,nbi,uem=graph_batch((3,3,3,3))
    original={key:value.clone() for key,value in graph.ndata.items()}
    field=LinearHead(a=0.);field.train()
    model=SimpleNamespace(vector_field=field)
    g=torch.Generator().manual_seed(515)
    initial=torch.randn((4,6),dtype=torch.float64,generator=torch.Generator().manual_seed(515))
    result=sample_clamped_work_proposal(model,graph,nbi,uem,steps=7,terminal_time=.8,
        noise_scale=.3,generator=g,parameterization='endpoint')
    expected=.5*initial.square().sum(-1)+3*math.log(2*math.pi)
    torch.testing.assert_close(result['proposal_log_factor'],expected,rtol=1e-13,atol=1e-13)
    torch.testing.assert_close(result['positions'].sum(1),torch.zeros(4,3,dtype=torch.float64),atol=1e-14,rtol=0)
    assert field.training
    for key,value in original.items():torch.testing.assert_close(graph.ndata[key],value,rtol=0,atol=0)
    # Coordinates from the dataset must not initialize this generation path.
    graph.ndata['x_1_true']+=100
    other=sample_clamped_work_proposal(model,graph,nbi,uem,steps=7,terminal_time=.8,
        noise_scale=.3,generator=torch.Generator().manual_seed(515),parameterization='endpoint')
    torch.testing.assert_close(other['positions'],result['positions'],rtol=0,atol=0)


def test_reject_mixed_conditions_before_path_weighting():
    graph,nbi,uem=graph_batch((3,4))
    with pytest.raises(ValueError,match='fixed atom count'):
        sample_clamped_work_proposal(SimpleNamespace(vector_field=LinearHead()),graph,nbi,uem,
            steps=2,terminal_time=.8,noise_scale=.2,generator=torch.Generator(),parameterization='endpoint')
