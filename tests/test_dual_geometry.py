import torch
from test_geometry_self_conditioning import small_model,prepare_graph
from cfm_mol.dual_geometry import patch_dual_geometry
from cfm_mol.clamped_density import deterministic_field
from cfm_mol.clamped_fm import clamped_fm_loss
from cfm_mol.radial_reference import prepare_research_backbone


def prediction(model,g,nbi,uem,t):
    with deterministic_field(model.vector_field):return model.vector_field(g,torch.tensor([t]),node_batch_idx=nbi,upper_edge_mask=uem)


def test_time_geometry_identity_limits_and_parameter_count():
    model=small_model();g,nbi,uem=prepare_graph();count=sum(p.numel() for p in model.vector_field.parameters())
    reference=prediction(model,g,nbi,uem,0.)['x']
    patch_dual_geometry(model,'time')
    torch.testing.assert_close(prediction(model,g,nbi,uem,0.)['x'],reference,atol=2e-6,rtol=2e-6)
    assert sum(p.numel() for p in model.vector_field.parameters())==count
    for t in [0.,.25,.7,1.]:
        out=prediction(model,g,nbi,uem,t);x=g.ndata['x_t']
        torch.testing.assert_close(out['_dual_geometry_endpoint'],x+(1-t)*(out['x']-x),atol=2e-6,rtol=2e-6)
    fixed=small_model();fixed.vector_field.load_state_dict(model.vector_field.state_dict());patch_dual_geometry(fixed,'fixed')
    torch.testing.assert_close(prediction(model,g,nbi,uem,1.)['x'],prediction(fixed,g,nbi,uem,1.)['x'],atol=0,rtol=0)
    assert '_dual_geometry_scale' not in g.ndata


def test_dual_geometry_gradients_symmetry_and_restoration():
    model=small_model();g,nbi,uem=prepare_graph();patch_dual_geometry(model,'time')
    loss=clamped_fm_loss(model,g,nbi,uem,terminal_time=1.,parameterization='displacement',generator=torch.Generator().manual_seed(34801))
    loss.backward();assert any(p.grad is not None and p.grad.norm()>0 for p in model.vector_field.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.vector_field.parameters() if p.grad is not None)
    original=prediction(model,g,nbi,uem,.4)['x'];perm=torch.tensor([2,0,3,1])
    rotation,_=torch.linalg.qr(torch.randn(3,3,generator=torch.Generator().manual_seed(34802)))
    with g.local_scope():
        g.ndata['x_t']=g.ndata['x_t'][perm]@rotation
        for key in ['a_t','a_1_true','c_t','c_1_true']:g.ndata[key]=g.ndata[key][perm]
        changed=prediction(model,g,nbi,uem,.4)['x']
    torch.testing.assert_close(changed,original[perm]@rotation,atol=4e-5,rtol=4e-5)
    restored=small_model();prepare_research_backbone(restored,dict(position_parameterization='displacement',dual_geometry=dict(mode='time')))
    restored.vector_field.load_state_dict(model.vector_field.state_dict(),strict=True)
    torch.testing.assert_close(prediction(restored,g,nbi,uem,.4)['x'],original,atol=0,rtol=0)
