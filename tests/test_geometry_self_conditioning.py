import torch
import pytest
from test_latent_tree_context import small_model,predict
from cfm_mol.condition_systems import graph_from_condition
from flowmol.data_processing.utils import get_batch_idxs,get_upper_edge_mask

def prepare_graph():
    g=graph_from_condition(dict(atomic_numbers=[6,6,1,1],charge=0,spin_multiplicity=1),["H","C","O"])
    x=torch.randn((4,3),generator=torch.Generator().manual_seed(33803));x-=x.mean(0)
    g.ndata["x_t"]=x;g.ndata["x_1_true"]=x.clone()
    for key in ["a_t","a_1_true","c_t","c_1_true"]:g.ndata[key]=torch.nn.functional.pad(g.ndata[key],(0,1))
    for key in ["e_t","e_1_true"]:g.edata[key]=torch.nn.functional.pad(g.edata[key],(0,1))
    g.ndata["has_reference_geometry"]=torch.ones((4,1),dtype=torch.bool)
    return g,get_batch_idxs(g)[0],get_upper_edge_mask(g)
from cfm_mol.geometry_self_conditioning import patch_geometry_self_conditioning
from cfm_mol.dynamic_tree_attention import patch_dynamic_tree_attention
from cfm_mol.clamped_density import deterministic_field
from cfm_mol.clamped_fm import clamped_fm_loss
from cfm_mol.radial_reference import prepare_research_backbone


def test_endpoint_formula_identity_and_full_gradient():
    model=small_model();g,nbi,uem=prepare_graph();baseline=predict(model,g,nbi,uem)
    patch_geometry_self_conditioning(model)
    records=[]
    hook=model.vector_field.self_conditioning_residual_layer.register_forward_pre_hook(lambda module,args:records.append(args[5]['x']))
    with deterministic_field(model.vector_field):out=model.vector_field(g,torch.tensor([.4]),node_batch_idx=nbi,upper_edge_mask=uem)
    hook.remove();assert len(records)==1
    torch.testing.assert_close(out['x'],baseline,atol=0,rtol=0)
    x=g.ndata['x_t'];expected=x+.6*(out['_geometry_sc_first_x']-x)
    torch.testing.assert_close(records[0],expected)
    assert records[0].requires_grad
    loss=clamped_fm_loss(model,g,nbi,uem,terminal_time=1.,parameterization='displacement',generator=torch.Generator().manual_seed(33801))
    loss.backward()
    layers=model.vector_field.self_conditioning_residual_layer
    for stack in [layers.node_residual_mlp,layers.edge_residual_mlp]:
        linear=next(m for m in reversed(stack) if isinstance(m,torch.nn.Linear))
        assert torch.isfinite(linear.weight.grad).all() and linear.weight.grad.norm()>0
    assert '_geometry_sc_endpoint' not in g.ndata


def test_endpoint_structure_symmetry_restore_and_no_reference_leak():
    model=small_model();g,nbi,uem=prepare_graph()
    config=dict(atomic_numbers_by_type=[1,6,8],mode='tree_learned',hidden=8,geometry='endpoint')
    patch_dynamic_tree_attention(model,**config);patch_geometry_self_conditioning(model)
    with torch.no_grad():
        model.vector_field.dynamic_tree_attention.messages[-1].weight.normal_(std=.03)
        for stack in [model.vector_field.self_conditioning_residual_layer.node_residual_mlp,model.vector_field.self_conditioning_residual_layer.edge_residual_mlp]:
            next(m for m in reversed(stack) if isinstance(m,torch.nn.Linear)).weight.normal_(std=.02)
    original=predict(model,g,nbi,uem)
    with g.local_scope():
        g.ndata['x_1_true']=torch.randn_like(g.ndata['x_1_true'])*100
        torch.testing.assert_close(predict(model,g,nbi,uem),original,atol=0,rtol=0)
    rotation,_=torch.linalg.qr(torch.randn(3,3,generator=torch.Generator().manual_seed(33802)))
    permutation=torch.tensor([2,0,3,1])
    with g.local_scope():
        g.ndata['x_t']=g.ndata['x_t'][permutation]@rotation
        for key in ['a_t','a_1_true','c_t','c_1_true']:g.ndata[key]=g.ndata[key][permutation]
        transformed=predict(model,g,nbi,uem)
    torch.testing.assert_close(transformed,original[permutation]@rotation,atol=4e-5,rtol=4e-5)
    restored=small_model()
    prepare_research_backbone(restored,dict(position_parameterization='displacement',geometry_self_conditioning=dict(zero_init=True,deep_supervision=True),dynamic_tree_attention=config))
    restored.vector_field.load_state_dict(model.vector_field.state_dict(),strict=True)
    torch.testing.assert_close(predict(restored,g,nbi,uem),original,atol=0,rtol=0)
    with pytest.raises(ValueError,match='external history'):
        model.vector_field(g,torch.tensor([.4]),node_batch_idx=nbi,upper_edge_mask=uem,prev_dst_dict={})


@pytest.mark.parametrize('mode',['latent','pooled'])
def test_coordinate_loss_trains_unlabelled_relation_head(mode):
    model=small_model();g,nbi,uem=prepare_graph()
    patch_geometry_self_conditioning(model,edge_feedback=mode)
    layers=model.vector_field.self_conditioning_residual_layer
    with torch.no_grad():
        next(m for m in reversed(layers.edge_residual_mlp) if isinstance(m,torch.nn.Linear)).weight.normal_(std=.03)
    loss=clamped_fm_loss(model,g,nbi,uem,terminal_time=1.,parameterization='displacement',generator=torch.Generator().manual_seed(33804))
    loss.backward()
    gradient=model.vector_field.to_edge_logits[-1].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.norm()>0
    assert g.edata['e_1_true'].argmax(-1).eq(0).all()
