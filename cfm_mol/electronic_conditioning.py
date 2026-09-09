"""Explicit whole-molecule electronic-state conditioning for the geometry branch.

This runtime patch is separate from the frozen composition prior. It broadcasts
unclipped total charge, spin multiplicity and a requested energy scale to node
embeddings, and neutralizes the legacy atom-zero charge marker. New checkpoints
must record this protocol and reapply the patch before state_dict loading.
No original electronic state is inferred when metadata is missing.
"""
import types
import math

import torch
from torch import nn


@torch.no_grad()
def neutralize_constant_temperature_input(model,reference_kT):
    """Preserve the field at its trained constant input for any initial new kT.

    Fold the old log-kT contribution into the first embedding bias and zero
    that input column. All parameters remain trainable. Use only when the
    checkpoint protocol establishes constant temperature-feature training;
    this is not a claim of pretrained thermodynamic temperature dependence.
    """
    if not math.isfinite(reference_kT) or reference_kT<=0:raise ValueError('Positive trained reference kT required')
    embedding=getattr(model.vector_field,'electronic_embedding',None)
    if not isinstance(embedding,nn.Sequential) or not isinstance(embedding[0],nn.Linear):
        raise ValueError('Explicit electronic embedding is required')
    first=embedding[0]
    if first.in_features!=5 or first.bias is None:raise ValueError('Unexpected electronic embedding layout')
    first.bias.add_(first.weight[:,2]*math.log(reference_kT))
    first.weight[:,2].zero_()


def attach_electronic_state(graph,charges,spins,kT_eV,*,atomic_numbers=None):
    batch=graph.batch_size;device=graph.device
    def values(value):
        value=torch.as_tensor(value,device=device,dtype=torch.float64).reshape(-1)
        return value.expand(batch) if value.numel()==1 else value
    q,s,kt=map(values,[charges,spins,kT_eV])
    if any(value.shape!=(batch,) or not torch.isfinite(value).all() for value in [q,s,kt]):
        raise ValueError('One finite electronic state is required per graph')
    if not torch.equal(q,q.round()) or not torch.equal(s,s.round()) or (s<1).any() or (kt<=0).any():
        raise ValueError('Charge/spin must be integers, multiplicity and requested kT positive')
    counts=graph.batch_num_nodes().to(device);node_batch=torch.repeat_interleave(torch.arange(batch,device=device),counts)
    if atomic_numbers is not None:
        numbers=torch.as_tensor(atomic_numbers,device=device,dtype=torch.float64)
        if numbers.shape!=(graph.num_nodes(),) or not torch.equal(numbers,numbers.round()) or (numbers<1).any():
            raise ValueError('Invalid atomic numbers')
        electrons=q.new_zeros(batch).index_add(0,node_batch,numbers)-q
        if (electrons<1).any() or (electrons<s-1).any() or ((electrons-s+1).remainder(2)!=0).any():
            raise ValueError('Electronic state violates electron-count parity')
    graph.ndata['electronic_state']=torch.stack([q,s,kt],-1)[node_batch]
    return graph


def patch_electronic_conditioning(model):
    field=model.vector_field
    if hasattr(field,'electronic_embedding'):return model
    if not isinstance(getattr(field,'scalar_embedding',None),nn.Sequential):
        raise ValueError('Electronic conditioning requires the FlowMol scalar embedding interface')
    linears=[module for module in field.scalar_embedding if isinstance(module,nn.Linear)]
    if not linears:raise ValueError('Cannot determine scalar embedding width')
    parameter=next(field.parameters());width=linears[-1].out_features
    field.electronic_embedding=nn.Sequential(nn.Linear(5,32),nn.SiLU(),nn.Linear(32,width)).to(parameter)
    nn.init.zeros_(field.electronic_embedding[-1].weight);nn.init.zeros_(field.electronic_embedding[-1].bias)
    field.electronic_embedding.train(field.training)
    original=field.forward

    def forward(self,graph,t,*args,**kwargs):
        if 'electronic_state' not in graph.ndata:
            raise ValueError('This checkpoint requires explicit total charge, multiplicity and requested kT')
        state=graph.ndata['electronic_state']
        if state.shape!=(graph.num_nodes(),3) or not torch.isfinite(state).all():raise ValueError('Invalid electronic state')
        counts=graph.batch_num_nodes().to(state.device)
        nbi=torch.repeat_interleave(torch.arange(graph.batch_size,device=state.device),counts)
        first=torch.cat([counts.new_zeros(1),counts.cumsum(0)[:-1]])
        if not torch.equal(state,state[first][nbi]):raise ValueError('Electronic state must be constant within each molecule')
        q,s,kt=state.unbind(-1)
        if not torch.equal(q,q.round()) or not torch.equal(s,s.round()) or (s<1).any() or (kt<=0).any():
            raise ValueError('Invalid total charge, multiplicity or requested kT')
        count=counts[nbi].to(state)
        features=torch.stack([q/4,torch.log(s),torch.log(kt),torch.log(count),(s-1)/count],-1)
        with graph.local_scope():
            # The old feature stores a molecule's *total* charge on atom zero,
            # rather than physical atom-resolved charges. Its position marker
            # must not determine the new geometry conditional distribution.
            legacy=graph.ndata['c_t'];neutral=torch.zeros_like(legacy)
            if neutral.shape[1]<=2:raise ValueError('Legacy charge encoding lacks neutral class 2')
            neutral[:,2]=1;graph.ndata['c_t']=neutral
            def inject(module,inputs,output):
                return output+self.electronic_embedding(features.to(output))
            hook=self.scalar_embedding.register_forward_hook(inject)
            try:return original(graph,t,*args,**kwargs)
            finally:hook.remove()
    field.forward=types.MethodType(forward,field)
    return model
