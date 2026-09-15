"""Task-clamped coordinate sampling using the original endpoint-head semantics.

This is a conditional adaptation of FlowMol3's Euler/history sampler, not its
published joint topology benchmark. The first native bootstrap costs two denoises.
"""
from types import MethodType
import torch
from cfm_mol.clamped_density import center_by_graph


@torch.no_grad()
def sample_native_endpoint(model,g,node_batch_idx,upper_edge_mask,x0,*,primitive_calls=128,history='native'):
    if history not in ['native','clamped']:raise ValueError('Unknown history mode')
    if primitive_calls<3:raise ValueError('Need at least three primitive calls')
    field=model.vector_field
    if hasattr(field,'_geometry_sc_configuration'):raise ValueError('Use original endpoint weights, not the displacement SC wrapper')
    old_sc=field.self_conditioning;training=[(m,m.training) for m in field.modules()]
    original=field.denoise_graph;counter=[0]
    def counted(self,*args,**kwargs):
        counter[0]+=1;return original(*args,**kwargs)
    field.denoise_graph=MethodType(counted,field);field.eval();field.self_conditioning=True
    def clamp_previous(dst):
        return dict(x=dst['x'],a=torch.nn.functional.one_hot(g.ndata['a_t'].argmax(-1),field.n_atom_types).to(x0),
                    c=torch.nn.functional.one_hot(g.ndata['c_t'].argmax(-1),field.n_charges).to(x0),
                    e=torch.nn.functional.one_hot(g.edata['e_t'][upper_edge_mask].argmax(-1),field.n_bond_types).to(x0))
    try:
        with g.local_scope():
            for key in ['a','c']:g.ndata[key+'_t']=g.ndata[key+'_1_true']
            g.edata['e_t']=g.edata['e_1_true']
            x=center_by_graph(x0,node_batch_idx,g.batch_size);previous=None;steps=primitive_calls-1
            grid=torch.linspace(0,1,steps+1,device=x.device,dtype=x.dtype)
            scheduler=field.interpolant_scheduler;feat=list(scheduler.feats).index('x')
            alpha=scheduler.alpha_t(grid.clone())[:,feat];derivative=scheduler.alpha_t_prime(grid.clone())[:,feat]
            for i in range(steps):
                g.ndata['x_t']=x;t=grid[i].expand(g.batch_size)
                if i==0 and history=='clamped':
                    field.self_conditioning=False
                    first=field(g,t,node_batch_idx,upper_edge_mask,apply_softmax=True,remove_com=True)
                    previous=clamp_previous(first);field.self_conditioning=True
                dst=field(g,t,node_batch_idx,upper_edge_mask,apply_softmax=True,remove_com=True,prev_dst_dict=previous)
                if not (1-alpha[i])>0:raise ValueError('Endpoint update evaluated at singular endpoint')
                x=center_by_graph(x+(grid[i+1]-grid[i])*derivative[i]/(1-alpha[i])*(dst['x']-x),node_batch_idx,g.batch_size)
                if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite original endpoint sampler')
                previous=dst if history=='native' else clamp_previous(dst)
            assert counter[0]==primitive_calls,(counter[0],primitive_calls)
            return x,dict(primitive_denoiser_calls=counter[0],euler_steps=steps,history=history)
    finally:
        field.denoise_graph=original;field.self_conditioning=old_sc
        for module,state in training:module.training=state
