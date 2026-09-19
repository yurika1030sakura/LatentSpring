"""Small, bounded geometry residual on a frozen molecular flow backbone.

Uses symmetric pair features and antisymmetric vector messages, established
equivariant constructions. Soft contacts come from the parent's own endpoint
estimate, never supplied bonds. This is a candidate architecture, not an
established performance or thermodynamic claim.
"""
import math
from types import MethodType
import torch
from torch import nn
from .chemical_moves import covalent_radii
from .clamped_density import center_by_graph


class PhysicalConnection(nn.Module):
    def __init__(self,atomic_numbers,embedding_dim=16,hidden_dim=64,velocity_scale=1.,gate_power=2):
        super().__init__()
        if velocity_scale<=0 or gate_power<1:raise ValueError('Positive velocity bound and time gate required')
        self.configuration=dict(atomic_numbers=list(atomic_numbers),embedding_dim=embedding_dim,hidden_dim=hidden_dim,
            velocity_scale=velocity_scale,gate_power=gate_power)
        self.register_buffer('radii',covalent_radii(atomic_numbers).float())
        self.atom_embedding=nn.Embedding(len(atomic_numbers),embedding_dim)
        self.pair_network=nn.Sequential(nn.Linear(2*embedding_dim+7,hidden_dim),nn.SiLU(),
            nn.Linear(hidden_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,2))
        nn.init.zeros_(self.pair_network[-1].weight);nn.init.zeros_(self.pair_network[-1].bias)
        self.velocity_scale=float(velocity_scale);self.gate_power=gate_power;self.forward_calls=0

    def forward(self,x,endpoint,types,t,node_batch_idx,source,destination):
        self.forward_calls+=1
        n_graphs=len(t);counts=torch.bincount(node_batch_idx,minlength=n_graphs).to(x)
        keep=source<destination;i,j=source[keep],destination[keep];batch=node_batch_idx[i]
        radii=self.radii[types];length=radii[i]+radii[j]
        r=x[i]-x[j];h=endpoint[i]-endpoint[j]
        r2=r.square().sum(-1);h2=h.square().sum(-1)
        rn=r/(r2+length.square()).sqrt()[:,None];hn=h/(h2+length.square()).sqrt()[:,None]
        embeddings=self.atom_embedding(types)
        scalars=torch.stack([torch.log1p(r2/length.square()),torch.log1p(h2/length.square()),
            (rn*hn).sum(-1),t[batch],t[batch].square(),torch.log1p(counts[batch])/math.log(201.),length/2],-1)
        features=torch.cat([embeddings[i]+embeddings[j],embeddings[i]*embeddings[j],scalars],-1)
        coefficients=torch.tanh(self.pair_network(features))
        contact=torch.sigmoid((1.25-(h2+1e-12).sqrt()/length)/.2)
        degree=x.new_zeros(len(x)).index_add(0,i,contact).index_add(0,j,contact)
        maximum=x.new_zeros(n_graphs).scatter_reduce(0,node_batch_idx,degree,reduce='amax',include_self=True).clamp_min(1.)
        messages=contact[:,None]*(coefficients[:,:1]*rn+coefficients[:,1:]*hn)/(2*maximum[batch,None])
        residual=x.new_zeros(x.shape).index_add(0,i,messages).index_add(0,j,-messages)
        return self.velocity_scale*t[node_batch_idx,None].pow(self.gate_power)*residual


class ContextualPhysicalConnection(PhysicalConnection):
    """Pair corrections conditioned on surrounding atoms and local geometry.

    Scalar message passing and invariant angular summaries are established
    constructions. This candidate tests missing molecular context; it is not
    a claim of a new equivariant architecture or established generation benefit.
    """
    def __init__(self,atomic_numbers,embedding_dim=16,hidden_dim=64,velocity_scale=1.,
                 gate_power=2,context_layers=2,context_width=32):
        if context_layers<1 or context_width<1:raise ValueError('Positive context dimensions required')
        super().__init__(atomic_numbers,embedding_dim,hidden_dim,velocity_scale,gate_power)
        self.configuration.update(context_layers=context_layers,context_width=context_width)
        self.node_input=nn.Linear(embedding_dim,context_width)
        self.messages=nn.ModuleList([nn.Sequential(nn.Linear(2*context_width+12,hidden_dim),nn.SiLU(),
            nn.Linear(hidden_dim,context_width)) for _ in range(context_layers)])
        self.updates=nn.ModuleList([nn.Sequential(nn.Linear(2*context_width,hidden_dim),nn.SiLU(),
            nn.Linear(hidden_dim,context_width)) for _ in range(context_layers)])
        self.pair_network=nn.Sequential(nn.Linear(2*context_width+12,hidden_dim),nn.SiLU(),
            nn.Linear(hidden_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,2))
        nn.init.zeros_(self.pair_network[-1].weight);nn.init.zeros_(self.pair_network[-1].bias)

    def forward(self,x,endpoint,types,t,node_batch_idx,source,destination):
        self.forward_calls+=1
        n_graphs=len(t);counts=torch.bincount(node_batch_idx,minlength=n_graphs).to(x)
        keep=source<destination;i,j=source[keep],destination[keep];batch=node_batch_idx[i]
        radii=self.radii[types];length=radii[i]+radii[j]
        r=x[i]-x[j];h=endpoint[i]-endpoint[j]
        r2=r.square().sum(-1);h2=h.square().sum(-1)
        rn=r/(r2+length.square()).sqrt()[:,None];hn=h/(h2+length.square()).sqrt()[:,None]
        contact=torch.sigmoid((1.25-(h2+1e-12).sqrt()/length)/.2)
        degree=x.new_zeros(len(x)).index_add(0,i,contact).index_add(0,j,contact)
        denominator=degree.clamp_min(1.)
        neighborhood=x.new_zeros(x.shape).index_add(0,i,contact[:,None]*hn).index_add(0,j,-contact[:,None]*hn)/denominator[:,None]
        log_degree=torch.log1p(degree)
        scalars=torch.stack([torch.log1p(r2/length.square()),torch.log1p(h2/length.square()),
            (rn*hn).sum(-1),t[batch],t[batch].square(),torch.log1p(counts[batch])/math.log(201.),length/2,
            log_degree[i]+log_degree[j],(log_degree[i]-log_degree[j]).square(),
            ((neighborhood[i]-neighborhood[j])*hn).sum(-1),
            neighborhood[i].square().sum(-1)+neighborhood[j].square().sum(-1),
            (neighborhood[i]*neighborhood[j]).sum(-1)],-1)
        nodes=self.node_input(self.atom_embedding(types))
        def features():return torch.cat([nodes[i]+nodes[j],nodes[i]*nodes[j],scalars],-1)
        for message,update in zip(self.messages,self.updates):
            edge=contact[:,None]*message(features())
            aggregate=nodes.new_zeros(nodes.shape).index_add(0,i,edge).index_add(0,j,edge)/denominator[:,None]
            nodes=nodes+update(torch.cat([nodes,aggregate],-1))
        coefficients=torch.tanh(self.pair_network(features()))
        maximum=x.new_zeros(n_graphs).scatter_reduce(0,node_batch_idx,degree,reduce='amax',include_self=True).clamp_min(1.)
        messages=contact[:,None]*(coefficients[:,:1]*rn+coefficients[:,1:]*hn)/(2*maximum[batch,None])
        residual=x.new_zeros(x.shape).index_add(0,i,messages).index_add(0,j,-messages)
        return self.velocity_scale*t[node_batch_idx,None].pow(self.gate_power)*residual


def make_physical_connection(**configuration):
    cls=ContextualPhysicalConnection if configuration.get('context_layers',0)>0 else PhysicalConnection
    return cls(**configuration)


def patch_physical_connection(model,**configuration):
    field=model.vector_field
    if hasattr(field,'physical_connection'):
        if field.physical_connection.configuration!=configuration:raise ValueError('Conflicting physical-connection configuration')
        return model
    parameters=model.parameters() if hasattr(model,'parameters') else field.parameters()
    for p in parameters:p.requires_grad_(False)
    reference=next(field.parameters())
    field.physical_connection=make_physical_connection(**configuration).to(reference)
    original=field.forward
    def forward(self,graph,t,node_batch_idx,upper_edge_mask=None,**kwargs):
        output=original(graph,t,node_batch_idx=node_batch_idx,upper_edge_mask=upper_edge_mask,**kwargs)
        x=center_by_graph(graph.ndata['x_t'],node_batch_idx,graph.batch_size)
        predicted=center_by_graph(output['x'],node_batch_idx,graph.batch_size)
        endpoint=x+(1-t[node_batch_idx,None])*(predicted-x)
        types=graph.ndata['a_t'].argmax(-1);src,dst=graph.edges()
        delta=self.physical_connection(x,endpoint,types,t,node_batch_idx,src,dst)
        result=dict(output);result['x']=output['x']+delta
        # The parent is frozen. Supervise the corrected final prediction;
        # the first-pass parent's error is a parameter-independent constant.
        result.pop('_geometry_sc_first_x',None)
        return result
    field.forward=MethodType(forward,field)
    return model
