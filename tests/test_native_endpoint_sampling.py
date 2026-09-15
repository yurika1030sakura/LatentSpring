import torch
import pytest
from test_latent_tree_context import small_model
from test_geometry_self_conditioning import prepare_graph
from cfm_mol.native_endpoint_sampling import sample_native_endpoint


@pytest.mark.parametrize('history',['native','clamped'])
def test_original_endpoint_bootstrap_budget_and_manual_euler(history):
    model=small_model();g,nbi,uem=prepare_graph();x0=g.ndata['x_t'].clone()
    result,info=sample_native_endpoint(model,g,nbi,uem,x0,primitive_calls=5,history=history)
    assert info['primitive_denoiser_calls']==5 and model.vector_field.training
    field=model.vector_field;field.eval();field.self_conditioning=True
    def clamped(d):
        return dict(x=d['x'],a=g.ndata['a_t'][:,:field.n_atom_types],c=g.ndata['c_t'][:,:field.n_charges],e=g.edata['e_t'][uem,:field.n_bond_types])
    with torch.no_grad(),g.local_scope():
        x=x0.clone();previous=None
        for i in range(4):
            t=torch.tensor([i/4]);g.ndata['x_t']=x
            if i==0 and history=='clamped':
                field.self_conditioning=False;previous=clamped(field(g,t,nbi,uem,apply_softmax=True,remove_com=True));field.self_conditioning=True
            d=field(g,t,nbi,uem,apply_softmax=True,remove_com=True,prev_dst_dict=previous)
            x=x+.25/(1-i/4)*(d['x']-x);x-=x.mean(0)
            previous=d if history=='native' else clamped(d)
    torch.testing.assert_close(result,x,atol=2e-6,rtol=2e-6)
    with g.local_scope():
        g.ndata['x_1_true']=torch.randn_like(x0)*100
        repeated,_=sample_native_endpoint(model,g,nbi,uem,x0,primitive_calls=5,history=history)
    torch.testing.assert_close(repeated,result,atol=0,rtol=0)
