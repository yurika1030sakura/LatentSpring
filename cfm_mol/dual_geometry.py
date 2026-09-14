"""Separate velocity accumulation from internal geometry in a coordinate GNN.

For linear FM, a velocity estimate V corresponds to endpoint estimate
Y=X_t+(1-t)*V. Native coordinate-updating layers otherwise use X_t+V as
geometry after adapting their final coordinate increment to a velocity head.
This runtime replacement reuses every original learned module and adds no
parameters. Its molecular usefulness is an experimental hypothesis.
"""
from types import MethodType
import torch
from cfm_mol.clamped_density import center_by_graph


def patch_dual_geometry(model,mode='time'):
    field=model.vector_field
    if mode not in ['time','fixed']:raise ValueError('Unknown internal geometry mode')
    if hasattr(field,'_dual_geometry_configuration'):
        if field._dual_geometry_configuration!={'mode':mode}:raise ValueError('Internal geometry configuration changed')
        return model
    parameter=next(field.parameters());grid=torch.tensor([0.,.5,1.],device=parameter.device,dtype=parameter.dtype)
    scheduler=field.interpolant_scheduler;feature=list(scheduler.feats).index('x')
    if not torch.allclose(scheduler.alpha_t(grid.clone())[:,feature],grid) or not torch.allclose(scheduler.alpha_t_prime(grid.clone())[:,feature],torch.ones_like(grid)):
        raise ValueError('Dual geometry currently requires linear position FM')
    original_forward=field.forward
    def forward(self,g,t,node_batch_idx,upper_edge_mask,*args,**kwargs):
        with g.local_scope():
            scale=(1-t[node_batch_idx,None]) if mode=='time' else torch.zeros_like(t[node_batch_idx,None])
            g.ndata['_dual_geometry_scale']=scale
            return original_forward(g,t,node_batch_idx,upper_edge_mask,*args,**kwargs)

    def denoise(self,g,node_scalar_features,node_vec_features,node_positions,edge_features,node_batch_idx,upper_edge_mask,apply_softmax=False,remove_com=False):
        if '_dual_geometry_scale' not in g.ndata:raise ValueError('Dual geometry requires its time-aware forward wrapper')
        initial=node_positions;velocity=torch.zeros_like(initial);scale=g.ndata['_dual_geometry_scale'].to(initial)
        geometry=initial;x_diff,d=self.precompute_distances(g,geometry)
        # Reuse the native convolution/update schedule and all original modules.
        for _ in range(self.n_recycles):
            for conv_index,conv in enumerate(self.conv_layers):
                node_scalar_features,node_vec_features=conv(g,scalar_feats=node_scalar_features,coord_feats=geometry,
                    vec_feats=node_vec_features,edge_feats=edge_features,x_diff=x_diff,d=d)
                if conv_index!=0 and (conv_index+1)%self.convs_per_update==0:
                    update_index=conv_index//self.convs_per_update if self.separate_mol_updaters else 0
                    raw=self.node_position_updaters[update_index](node_scalar_features,geometry,node_vec_features)
                    velocity=velocity+(raw-geometry)
                    geometry=initial+scale*velocity
                    x_diff,d=self.precompute_distances(g,geometry)
                    edge_features=self.edge_updaters[update_index](g,node_scalar_features,edge_features,d=d)
        node_logits=self.node_output_head(node_scalar_features)
        edge_logits=self.to_edge_logits(edge_features[upper_edge_mask]+edge_features[~upper_edge_mask])
        output=initial+velocity
        if remove_com:output=center_by_graph(output,node_batch_idx,g.batch_size)
        result=dict(x=output,a=node_logits[:,:self.n_atom_types],e=edge_logits,_dual_geometry_endpoint=geometry)
        if not self.exclude_charges:result['c']=node_logits[:,self.n_atom_types:]
        if apply_softmax:
            for key in ['a','c','e']:
                if key in result:result[key]=result[key].softmax(-1)
        return result
    field.forward=MethodType(forward,field);field.denoise_graph=MethodType(denoise,field)
    field._dual_geometry_configuration={'mode':mode}
    return model
