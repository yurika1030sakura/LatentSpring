"""Independent, time-batched FlowMol drift calls in a fixed molecular COM basis."""
import math

import dgl
import torch
from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask

from cfm_mol.clamped_density import deterministic_field, position_velocity
from cfm_mol.nonequilibrium import centered_orthonormal_basis


class MolecularPathDrift:
    """Replicate one conditioned graph; each row has its own position and time.

    The supplied graph must already carry the requested electronic state. Graph
    features and caller train/eval modes are restored after every field call.
    No trajectory history, reference geometry or sampler retraction is used.
    """
    def __init__(self, model, condition_graph, *, sign=1.):
        if condition_graph.batch_size != 1 or not math.isfinite(sign):
            raise ValueError('One conditioned molecule and a finite sign are required')
        self.model=model;self.base=condition_graph;self.sign=sign
        self.n=condition_graph.num_nodes()
        if not 2 <= self.n <= 200:raise ValueError('Require 2 to 200 atoms')
        self.basis=centered_orthonormal_basis(self.n,device=condition_graph.device)
        self.cache={}

    def __call__(self,z,t):
        if z.ndim!=2 or z.shape[1]!=3*(self.n-1) or len(z)<1 or not torch.isfinite(z).all():
            raise ValueError('Invalid molecular COM coordinates')
        if z.device!=self.base.device:raise ValueError('Graph and coordinates must share a device')
        batch=len(z);parameter=next(self.model.vector_field.parameters())
        times=torch.as_tensor(t,device=z.device,dtype=parameter.dtype)
        if times.ndim==0:times=times.expand(batch)
        if times.shape!=(batch,) or not torch.isfinite(times).all() or ((times<0)|(times>1)).any():
            raise ValueError('One time in [0,1] is required per path row')
        if batch not in self.cache:
            graph=dgl.batch([self.base]*batch)
            nbi,_=get_batch_idxs(graph)
            self.cache[batch]=(graph,nbi,get_upper_edge_mask(graph))
        graph,nbi,uem=self.cache[batch]
        x=torch.einsum('nk,bkd->bnd',self.basis,z.double().reshape(batch,self.n-1,3)).reshape(batch*self.n,3).to(parameter.dtype)
        with graph.local_scope(),deterministic_field(self.model.vector_field),torch.autocast(device_type=x.device.type,enabled=False):
            for key in ['a','c']:graph.ndata[key+'_t']=graph.ndata[key+'_1_true']
            graph.edata['e_t']=graph.edata['e_1_true']
            v=position_velocity(self.model,graph,x,times,nbi,uem,parameterization='displacement')
        return self.sign*torch.einsum('nk,bnd->bkd',self.basis,v.double().reshape(batch,self.n,3)).reshape_as(z)
