"""True upstream API test. It is SKIPPED, not simulated, without DGL/FlowMol."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
pytest.importorskip('dgl',reason='Real DGL is required for this native-backbone test')
pytest.importorskip('flowmol',reason='Real FlowMol is required for this native-backbone test')
from flowmol.models.ctmc_vector_field import CTMCVectorField
from flowmol.models.interpolant_scheduler import InterpolantScheduler
from test_clamped_density import graph_batch
from cfm_mol.geometry_self_conditioning import patch_geometry_self_conditioning
from cfm_mol.tree_prior_controls import TreePriorControl
from cfm_mol.weighted_endpoints import EndpointDraw
from cfm_mol.weighted_endpoint_fm import endpoint_fm_loss


def test_weighted_endpoint_adapter_real_two_pass_flowmol():
    torch.manual_seed(125)
    field=CTMCVectorField(n_atom_types=3,canonical_feat_order=['x','a','c','e'],
        interpolant_scheduler=InterpolantScheduler(['x','a','c','e'],schedule_type='linear'),
        n_vec_channels=4,n_hidden_scalars=8,n_hidden_edge_feats=8,n_molecule_updates=1,
        convs_per_update=2,n_message_gvps=1,n_update_gvps=1,n_expansion_gvps=1,rbf_dim=4,
        self_conditioning=True)
    model=SimpleNamespace(vector_field=field,_research_prior_kind='harmonic_tree')
    patch_geometry_self_conditioning(model)
    graph,nbi,uem=graph_batch((3,),dtype=torch.float32)
    c=dict(numbers=[6,6,6],atomic_numbers=[6,6,6],charge=0,spin_multiplicity=1,n_atoms=3)
    draws=[EndpointDraw('test','particle','basin',c,graph.ndata['x_1_true'].numpy())]
    prior=TreePriorControl('harmonic_tree').double()
    loss=endpoint_fm_loss(model,graph,nbi,uem,draws,prior,120)
    loss.backward()
    grads=[p.grad for p in field.parameters() if p.grad is not None]
    assert torch.isfinite(loss) and grads and all(torch.isfinite(v).all() for v in grads)
    assert any(v.abs().sum()>0 for v in grads)
