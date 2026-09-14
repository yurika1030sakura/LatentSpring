"""Memoryless geometry-only self-conditioning for a linear displacement FM head.

Self-conditioning is established prior art. For any linear FM coupling, the
conditional endpoint estimate is x+(1-t)*v. A native endpoint head must not be
fed the displacement-head output directly. Both passes remain differentiable.
"""
from types import MethodType
import torch
from torch import nn
from cfm_mol.clamped_density import center_by_graph


def patch_geometry_self_conditioning(model,zero_init=True,deep_supervision=True,edge_feedback="clamped"):
    field=model.vector_field
    configuration=dict(zero_init=bool(zero_init),deep_supervision=bool(deep_supervision),edge_feedback=edge_feedback)
    if edge_feedback not in ["clamped","latent","pooled"]:raise ValueError("Unknown edge feedback")
    if hasattr(field,'_geometry_sc_configuration'):
        if field._geometry_sc_configuration!=configuration:raise ValueError('Geometry SC configuration changed')
        return model
    if not hasattr(field,'self_conditioning_residual_layer'):raise ValueError('Construct the native self-conditioning modules first')
    if hasattr(field,'latent_tree_adapter'):raise ValueError('Do not mix this study with a fixed source-tree condition')
    parameter=next(field.parameters());grid=torch.tensor([0.,.5,1.],device=parameter.device,dtype=parameter.dtype)
    scheduler=field.interpolant_scheduler;index=list(scheduler.feats).index('x')
    if not torch.allclose(scheduler.alpha_t(grid.clone())[:,index],grid) or not torch.allclose(scheduler.alpha_t_prime(grid.clone())[:,index],torch.ones_like(grid)):
        raise ValueError('This endpoint formula requires the declared linear position schedule')
    if zero_init:
        for module in [field.self_conditioning_residual_layer.node_residual_mlp,field.self_conditioning_residual_layer.edge_residual_mlp]:
            last=next(layer for layer in reversed(module) if isinstance(layer,nn.Linear))
            nn.init.zeros_(last.weight);nn.init.zeros_(last.bias)
    field._geometry_sc_configuration=configuration
    original=field.forward

    def forward(self,g,t,node_batch_idx,upper_edge_mask,apply_softmax=False,remove_com=False,prev_dst_dict=None):
        if prev_dst_dict is not None:raise ValueError('Memoryless geometry SC does not consume an external history')
        old=self.self_conditioning;old_suppress=getattr(self,'_geometry_sc_suppress_tree',False)
        try:
            with g.local_scope():
                self.self_conditioning=False;self._geometry_sc_suppress_tree=True
                first=original(g,t,node_batch_idx=node_batch_idx,upper_edge_mask=upper_edge_mask,apply_softmax=False,remove_com=False)
                x=center_by_graph(g.ndata['x_t'],node_batch_idx,g.batch_size)
                first_x=center_by_graph(first['x'],node_batch_idx,g.batch_size)
                clean=x+(1-t[node_batch_idx,None])*(first_x-x)
                def fixed_state(values,width):
                    labels=values.argmax(-1)
                    if (labels>=width).any():raise ValueError('Clamped category is outside the vector-field vocabulary')
                    return torch.nn.functional.one_hot(labels,width).to(clean)
                atom=fixed_state(g.ndata['a_t'],self.n_atom_types)
                charge=fixed_state(g.ndata['c_t'],self.n_charges)
                if hasattr(self,'electronic_embedding'):
                    charge=torch.zeros_like(charge);charge[:,2]=1.
                bond=fixed_state(g.edata['e_t'][upper_edge_mask],self.n_bond_types)
                if edge_feedback!='clamped':
                    bond=first['e'].softmax(-1)
                    if edge_feedback=='pooled':
                        edge_batch=node_batch_idx[g.edges()[0][upper_edge_mask]]
                        sums=bond.new_zeros((g.batch_size,bond.shape[-1])).index_add(0,edge_batch,bond)
                        counts=torch.bincount(edge_batch,minlength=g.batch_size).to(bond)
                        bond=(sums/counts[:,None])[edge_batch]
                previous=dict(x=clean,a=atom,c=charge,e=bond)
                g.ndata['_geometry_sc_endpoint']=clean
                g.ndata['_geometry_sc_envelope']=t[node_batch_idx,None].square()
                self.self_conditioning=True;self._geometry_sc_suppress_tree=False
                second=original(g,t,node_batch_idx=node_batch_idx,upper_edge_mask=upper_edge_mask,
                                apply_softmax=apply_softmax,remove_com=remove_com,prev_dst_dict=previous)
                if deep_supervision:second['_geometry_sc_first_x']=first_x
                return second
        finally:
            self.self_conditioning=old;self._geometry_sc_suppress_tree=old_suppress

    field.forward=MethodType(forward,field)
    return model
