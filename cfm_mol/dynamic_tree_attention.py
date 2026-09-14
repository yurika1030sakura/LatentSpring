"""State-dependent structured connectivity for bond-free molecular flow.

Matrix-tree attention is established prior art. This module tests its molecular
value, not a new matrix-tree theorem or a validity guarantee for output geometry.
"""
from types import MethodType
import torch
from torch import nn
from rdkit import Chem


def tree_edge_marginals(log_weights,relative_floor=1e-8):
    """Marginals with a shared1e-8 relative edge-weight floor for stable solves.

    The floor changes edge weights, not the tree family; no diagonal ridge or
    spanning-forest approximation is introduced. Local controls use the same floor.
    """
    n=log_weights.shape[-1]
    if log_weights.shape[-2]!=n or n<2: raise ValueError('At least two nodes required')
    mask=torch.eye(n,dtype=torch.bool,device=log_weights.device)
    off=log_weights.masked_fill(mask,-torch.inf)
    w=(off-off.amax((-1,-2),keepdim=True)).exp().clamp_min(relative_floor).masked_fill(torch.eye(n,dtype=torch.bool,device=log_weights.device),0.)
    lap=torch.diag_embed(w.sum(-1))-w
    q=torch.nn.functional.pad(torch.linalg.inv(lap[...,:-1,:-1]),(0,1,0,1))
    diagonal=q.diagonal(dim1=-2,dim2=-1)
    m=w*(diagonal[..., :,None]+diagonal[...,None,:]-q-q.transpose(-1,-2))
    if not torch.isfinite(m).all():raise FloatingPointError('Nonfinite dynamic tree marginals')
    return m


def local_edge_mass(log_weights,relative_floor=1e-8):
    """Independent edge weights with the same total undirected mass N-1."""
    n=log_weights.shape[-1]
    off=log_weights.masked_fill(torch.eye(n,dtype=torch.bool,device=log_weights.device),-torch.inf)
    w=(off-off.amax((-1,-2),keepdim=True)).exp().clamp_min(relative_floor).masked_fill(torch.eye(n,dtype=torch.bool,device=log_weights.device),0.)
    return w*(2*(n-1)/w.sum((-1,-2),keepdim=True))


class DynamicTreeAttention(nn.Module):
    def __init__(self,node_width,edge_width,atomic_numbers_by_type,mode='tree_learned',hidden=32):
        super().__init__()
        if mode not in ['tree_learned','tree_fixed','local_learned']:raise ValueError('Unknown dynamic connection rule')
        self.mode=mode
        table=Chem.GetPeriodicTable()
        radii=[table.GetRcovalent(z) for z in atomic_numbers_by_type]
        propensity=[.05 if z in [1,9,17,35,53] else float(max(1,table.GetDefaultValence(z)-1)) for z in atomic_numbers_by_type]
        self.register_buffer('type_radii',torch.tensor(radii))
        self.register_buffer('type_log_propensity',torch.tensor(propensity).log())
        self.node_summary=nn.Linear(node_width,8)
        self.affinity=nn.Sequential(nn.Linear(16+edge_width+3,hidden),nn.SiLU(),nn.Linear(hidden,1))
        self.messages=nn.Sequential(nn.Linear(3,hidden),nn.SiLU(),nn.Linear(hidden,edge_width))
        for head in [self.affinity,self.messages]:
            nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
        if mode=='tree_fixed':
            self.node_summary.requires_grad_(False);self.affinity.requires_grad_(False)

    def forward(self,graph,node_scalar,positions,edge_features,node_batch):
        counts=graph.batch_num_nodes().long();offsets=torch.cat([counts.new_zeros(1),counts.cumsum(0)[:-1]])
        src,dst=graph.edges();eb=node_batch[src]
        if not torch.equal(eb,node_batch[dst]):raise ValueError('Cross-molecule edges are forbidden')
        types=graph.ndata['a_1_true'].argmax(-1)
        if (types>=len(self.type_radii)).any():raise ValueError('Atomic type mapping mismatch')
        if (counts<2).any() or (counts>200).any():raise ValueError('Require2--200 atoms per graph')
        h=self.node_summary(node_scalar)
        features=edge_features.new_empty((graph.num_edges(),3))
        for n in counts.unique().tolist():
            chosen=torch.where(counts==n)[0]
            nodes=offsets[chosen,None]+torch.arange(n,device=positions.device)[None,:]
            radii=self.type_radii[types[nodes]].double()
            length=radii[:,:,None]+radii[:,None,:]
            x=positions[nodes].double()
            distance2=(x[:,:,None,:]-x[:,None,:,:]).square().sum(-1)
            relative=distance2/length.square()
            geometry=torch.stack([torch.log1p(relative),1/(1+relative),length],-1)
            propensity=self.type_log_propensity[types[nodes]].double()
            logw=propensity[:,:,None]+propensity[:,None,:]-2*torch.log1p(relative)
            graph_to_local=counts.new_full((len(counts),),-1);graph_to_local[chosen]=torch.arange(len(chosen),device=counts.device)
            selected=graph_to_local[eb]>=0;bi=graph_to_local[eb[selected]]
            i=src[selected]-offsets[eb[selected]];j=dst[selected]-offsets[eb[selected]]
            dense=edge_features.new_zeros((len(chosen),n,n,edge_features.shape[-1]))
            dense[bi,i,j]=edge_features[selected]
            dense=.5*(dense+dense.transpose(1,2))
            if self.mode!='tree_fixed':
                hn=h[nodes]
                inputs=torch.cat([hn[:,:,None,:]+hn[:,None,:,:],hn[:,:,None,:]*hn[:,None,:,:],dense,geometry.to(dense)],-1)
                logw=logw+2*torch.tanh(self.affinity(inputs).squeeze(-1)).double()
            mass=local_edge_mass(logw) if self.mode=='local_learned' else tree_edge_marginals(logw)
            degree=mass.sum(-1)
            normalized=mass/(degree[:,:,None]*degree[:,None,:]).clamp_min(1e-20).sqrt()
            values=torch.stack([torch.log1p(mass),normalized,torch.log1p(.5*(degree[:,:,None]+degree[:,None,:]))],-1)
            features[selected]=values[bi,i,j].to(features)
        return self.messages(features)


def patch_dynamic_tree_attention(model,atomic_numbers_by_type,mode='tree_learned',hidden=32):
    """Recompute soft connectivity at each denoiser call from its current geometry."""
    field=model.vector_field
    configuration=dict(atomic_numbers_by_type=list(atomic_numbers_by_type),mode=mode,hidden=hidden)
    if hasattr(field,'dynamic_tree_attention'):
        if field._dynamic_tree_configuration!=configuration:raise ValueError('Dynamic tree configuration changed')
        return model
    if hasattr(field,'latent_tree_adapter'):raise ValueError('Do not combine dynamic and static tree experiments')
    node_width=next(m.out_features for m in reversed(field.scalar_embedding) if isinstance(m,nn.Linear))
    edge_width=next(m.out_features for m in reversed(field.edge_embedding) if isinstance(m,nn.Linear))
    field.dynamic_tree_attention=DynamicTreeAttention(node_width,edge_width,atomic_numbers_by_type,mode,hidden).to(next(field.parameters()))
    field._dynamic_tree_configuration=configuration
    original=field.denoise_graph
    def denoise(self,g,node_scalar_features,node_vec_features,node_positions,edge_features,node_batch_idx,upper_edge_mask,apply_softmax=False,remove_com=False):
        residual=self.dynamic_tree_attention(g,node_scalar_features,node_positions,edge_features,node_batch_idx)
        return original(g,node_scalar_features,node_vec_features,node_positions,edge_features+residual,node_batch_idx,upper_edge_mask,apply_softmax,remove_com)
    field.denoise_graph=MethodType(denoise,field)
    return model
