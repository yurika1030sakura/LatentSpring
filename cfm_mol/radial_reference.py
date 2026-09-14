"""Smooth equivariant pair flow with analytic divergence, as a reference model.

Pair-kernel equivariant CNFs are prior work (Koehler, Klein and Noe, ICML 2020).
This intentionally restricted reference isolates noisy-trace objectives from
the cost of the original backbone. It is not a claimed new architecture.
The installed patch only defines the conditional position factor: a separate
frozen original checkpoint must provide the composition prior.
"""
import math
from types import MethodType

import torch
from torch import nn
from .clamped_density import center_by_graph


class RadialPairReference(nn.Module):
    """v_i = b x_i + sum_j phi_ij(||r_ij||^2) r_ij / max(N-1,1).

    Pair coefficients depend on time and symmetric atom/charge embeddings,
    never coordinates. Gaussian kernels of squared distance avoid norm
    singularities at collisions. Fixed positive widths and bounded network
    coefficients give a globally Lipschitz field for each finite composition.
    Pair interactions plus an isotropic global term are capacity restrictions.
    """
    def __init__(self, n_atom_types, n_charges, embedding_dim=16, hidden_dim=64,
                 n_kernels=24, radius_squared_max=36., kernel_width=2.,
                 coefficient_bound=4.):
        super().__init__()
        if min(n_atom_types,n_charges,embedding_dim,hidden_dim,n_kernels)<1:
            raise ValueError('Reference dimensions must be positive')
        if any(not math.isfinite(v) or v<=0 for v in
               [radius_squared_max,kernel_width,coefficient_bound]):
            raise ValueError('Kernel scales must be positive and finite')
        self.atom_embedding=nn.Linear(n_atom_types,embedding_dim,bias=False)
        self.charge_embedding=nn.Linear(n_charges,embedding_dim,bias=False)
        self.pair_network=nn.Sequential(nn.Linear(2*embedding_dim+5,hidden_dim),nn.SiLU(),
            nn.Linear(hidden_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,n_kernels))
        self.global_network=nn.Sequential(nn.Linear(embedding_dim+5,hidden_dim),nn.SiLU(),
            nn.Linear(hidden_dim,1))
        self.register_buffer('centers',torch.linspace(0,radius_squared_max,n_kernels))
        self.register_buffer('width',torch.tensor(float(kernel_width)))
        self.coefficient_bound=float(coefficient_bound)
        # Near identity, with small nonzero pair weights to exercise derivatives.
        for final in [self.pair_network[-1],self.global_network[-1]]:
            nn.init.normal_(final.weight,std=.005);nn.init.zeros_(final.bias)

    def velocity_and_divergence(self, graph, x, t, node_batch_idx):
        if x.shape[-1]!=3:
            raise ValueError('Reference assumes three spatial dimensions')
        n_graphs=graph.batch_size
        x=center_by_graph(x,node_batch_idx,n_graphs)
        counts=torch.bincount(node_batch_idx,minlength=n_graphs).to(x)
        features=self.atom_embedding(graph.ndata['a_t'])+self.charge_embedding(graph.ndata['c_t'])
        pooled=features.new_zeros((n_graphs,features.shape[-1])).index_add(0,node_batch_idx,features)/counts[:,None]
        clock=torch.stack([t,t.square(),torch.sin(math.pi*t),torch.cos(math.pi*t),
                           torch.log1p(counts)/math.log(201.)],dim=-1)
        dilation=self.coefficient_bound*torch.tanh(self.global_network(torch.cat([pooled,clock],-1))).squeeze(-1)
        velocity=dilation[node_batch_idx,None]*x
        divergence=3*(counts-1)*dilation
        source,destination=graph.edges()
        selected=source<destination
        source,destination=source[selected],destination[selected]
        if len(source)==0:
            return velocity,divergence
        graph_index=node_batch_idx[source]
        # All repository graphs are complete, with one copy of each directed
        # edge. Counts/indices are structural constants under differentiation.
        symmetric=torch.cat([features[source]+features[destination],
                             features[source]*features[destination],clock[graph_index]],dim=-1)
        coefficients=self.coefficient_bound*torch.tanh(self.pair_network(symmetric))
        delta=x[source]-x[destination]
        squared=delta.square().sum(-1)
        scaled=(squared[:,None]-self.centers)/self.width
        kernels=torch.exp(-.5*scaled.square())
        normalizer=(counts[graph_index]-1).clamp_min(1)
        phi=(coefficients*kernels).sum(-1)/normalizer
        derivative=(coefficients*kernels*(-scaled/self.width)).sum(-1)/normalizer
        pair_velocity=phi[:,None]*delta
        velocity=velocity.index_add(0,source,pair_velocity).index_add(0,destination,-pair_velocity)
        # An unordered pair contributes through BOTH atom diagonal blocks.
        pair_divergence=2*(3*phi+2*squared*derivative)
        divergence=divergence.index_add(0,graph_index,pair_divergence)
        return velocity,divergence


def patch_radial_reference(model):
    """Install a conditional position-only reference on the existing instance.

    The original modules and state keys are retained and frozen, so no shared
    FlowMol source is edited. Reapply this patch BEFORE loading a radial
    checkpoint; apply AFTER loading an original checkpoint to initialize it.
    """
    vf=model.vector_field
    if hasattr(vf,'bgfm_radial_reference'):
        return model
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    anchor=next(vf.parameters())
    vf.bgfm_radial_reference=RadialPairReference(vf.n_atom_types,vf.n_charges).to(anchor)

    def forward(self,graph,t,node_batch_idx,upper_edge_mask=None,**kwargs):
        x=graph.ndata['x_t']
        velocity,_=self.bgfm_radial_reference.velocity_and_divergence(graph,x,t,node_batch_idx)
        return {'x':x+velocity}  # explicitly a displacement head

    def divergence(self,graph,x,t,node_batch_idx,parameterization):
        if parameterization!='displacement':
            raise ValueError('Radial reference requires displacement velocity semantics')
        return self.bgfm_radial_reference.velocity_and_divergence(graph,x,t,node_batch_idx)[1]

    vf.forward=MethodType(forward,vf)
    vf.exact_clamped_divergence=MethodType(divergence,vf)
    return model


def prepare_research_backbone(model, protocol):
    """Restore a declared research backbone before strict checkpoint loading."""
    model._research_prior_kind=protocol.get('source_prior_kind','gaussian')
    backbone=protocol.get('position_backbone','flowmol')
    if backbone=='radial_reference':
        patch_radial_reference(model)
    elif backbone!='flowmol':
        raise ValueError(f'Unknown checkpoint position backbone: {backbone}')
    if protocol.get('electronic_conditioning',False):
        if backbone!='flowmol':raise ValueError('Electronic conditioning requires the FlowMol backbone')
        from cfm_mol.electronic_conditioning import patch_electronic_conditioning
        patch_electronic_conditioning(model)
    if protocol.get('dual_geometry'):
        if protocol.get('position_parameterization')!='displacement':raise ValueError('Dual geometry requires a displacement head')
        from cfm_mol.dual_geometry import patch_dual_geometry
        patch_dual_geometry(model,**protocol['dual_geometry'])
    if protocol.get('latent_tree_context'):
        if backbone!='flowmol':raise ValueError('Tree context requires the FlowMol backbone')
        from cfm_mol.latent_tree_context import patch_latent_tree_context
        patch_latent_tree_context(model,**protocol['latent_tree_context'])
    if protocol.get('dynamic_tree_attention'):
        if backbone!='flowmol':raise ValueError('Dynamic tree attention requires the FlowMol backbone')
        from cfm_mol.dynamic_tree_attention import patch_dynamic_tree_attention
        patch_dynamic_tree_attention(model,**protocol['dynamic_tree_attention'])
    if protocol.get('geometry_self_conditioning'):
        if protocol.get('position_parameterization')!='displacement':raise ValueError('Geometry SC requires its declared displacement head')
        from cfm_mol.geometry_self_conditioning import patch_geometry_self_conditioning
        patch_geometry_self_conditioning(model,**protocol['geometry_self_conditioning'])
    return model
