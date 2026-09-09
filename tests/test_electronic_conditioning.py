from types import SimpleNamespace

import pytest
import torch
from torch import nn

from test_clamped_density import graph_batch
from cfm_mol.electronic_conditioning import attach_electronic_state,patch_electronic_conditioning


class SmallField(nn.Module):
    def __init__(self):
        super().__init__();self.scalar_embedding=nn.Sequential(nn.Linear(12,6),nn.SiLU(),nn.Linear(6,6)).double()
        self.fail=False
    def forward(self,g,t,node_batch_idx,**kwargs):
        features=torch.cat([g.ndata['a_t'],g.ndata['c_t'],t[node_batch_idx,None]],-1)
        scalar=self.scalar_embedding(features)
        if self.fail:raise RuntimeError('deliberate failure')
        return {'x':g.ndata['x_t']*(1+scalar.mean(-1,keepdim=True))}


def prepared():
    torch.manual_seed(740)
    graph,nbi,uem=graph_batch((8,8))
    for key in ['a','c']:graph.ndata[key+'_t']=graph.ndata[key+'_1_true'].clone()
    model=SimpleNamespace(vector_field=SmallField())
    attach_electronic_state(graph,[4,0],[1,1],1.,atomic_numbers=torch.ones(16,dtype=torch.long))
    patch_electronic_conditioning(model)
    return graph,nbi,uem,model


def test_unclipped_charge_is_broadcast_and_old_marker_location_does_not_matter():
    g,nbi,uem,model=prepared();t=torch.full((2,),.4,dtype=torch.float64)
    assert torch.equal(g.ndata['electronic_state'][:8,0],torch.full((8,),4.,dtype=torch.float64))
    first=model.vector_field(g,t,nbi)['x']
    g.ndata['c_t'][0]=torch.nn.functional.one_hot(torch.tensor(5),7).double()
    original=g.ndata['c_t'].clone()
    second=model.vector_field(g,t,nbi)['x']
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    torch.testing.assert_close(g.ndata['c_t'],original,rtol=0,atol=0)


def test_electronic_embedding_gets_gradient_and_can_distinguish_spin_states():
    g,nbi,uem,model=prepared();t=torch.full((2,),.4,dtype=torch.float64)
    output=model.vector_field(g,t,nbi)['x'];output.square().sum().backward()
    assert model.vector_field.electronic_embedding[-1].weight.grad.abs().sum()>0
    with torch.no_grad():model.vector_field.electronic_embedding[-1].weight.fill_(.1)
    attach_electronic_state(g,[0,0],[1,1],1.,atomic_numbers=torch.ones(16,dtype=torch.long))
    singlet=model.vector_field(g,t,nbi)['x']
    attach_electronic_state(g,[0,0],[3,3],1.,atomic_numbers=torch.ones(16,dtype=torch.long))
    triplet=model.vector_field(g,t,nbi)['x']
    assert not torch.allclose(singlet,triplet)


def test_missing_state_is_rejected_and_failed_calls_remove_hooks():
    g,nbi,uem,model=prepared();saved=g.ndata['electronic_state'].clone();del g.ndata['electronic_state']
    with pytest.raises(ValueError,match='requires explicit'):model.vector_field(g,torch.ones(2,dtype=torch.float64),nbi)
    g.ndata['electronic_state']=saved;model.vector_field.fail=True;original=g.ndata['c_t'].clone()
    with pytest.raises(RuntimeError,match='deliberate'):model.vector_field(g,torch.ones(2,dtype=torch.float64),nbi)
    assert len(model.vector_field.scalar_embedding._forward_hooks)==0
    torch.testing.assert_close(g.ndata['c_t'],original,rtol=0,atol=0)


def test_bad_spin_parity_and_nonconstant_node_metadata_are_rejected():
    g,nbi,uem,model=prepared()
    with pytest.raises(ValueError,match='parity'):
        attach_electronic_state(g,[0,0],[2,2],1.,atomic_numbers=torch.ones(16,dtype=torch.long))
    g.ndata['electronic_state'][1,0]+=1
    with pytest.raises(ValueError,match='constant within'):
        model.vector_field(g,torch.ones(2,dtype=torch.float64),nbi)
