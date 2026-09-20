"""Bounded atom-normalized messages and a balanced local-force target.

Receiver normalization gives a weakly connected atom a usable message budget.
Centering restores the translational constraint; a factor of one half preserves
the original per-atom bound. These are standard equivariant operations, not a
new equivariance theorem or an established molecular-validity guarantee.
"""
import math
import torch
from .physical_connection import PhysicalConnection
from .clamped_density import center_by_graph


class AtomwisePhysicalConnection(PhysicalConnection):
    def __init__(self,atomic_numbers,embedding_dim=16,hidden_dim=64,velocity_scale=1.,gate_power=2,normalization='atomwise'):
        if normalization!='atomwise':raise ValueError('Expected atomwise normalization')
        super().__init__(atomic_numbers,embedding_dim,hidden_dim,velocity_scale,gate_power)
        self.configuration['normalization']=normalization

    def forward(self,x,endpoint,types,t,node_batch_idx,source,destination):
        self.forward_calls+=1;n_graphs=len(t);counts=torch.bincount(node_batch_idx,minlength=n_graphs).to(x)
        keep=source<destination;i,j=source[keep],destination[keep];batch=node_batch_idx[i]
        radii=self.radii[types];length=radii[i]+radii[j];r=x[i]-x[j];h=endpoint[i]-endpoint[j]
        r2=r.square().sum(-1);h2=h.square().sum(-1)
        rn=r/(r2+length.square()).sqrt()[:,None];hn=h/(h2+length.square()).sqrt()[:,None]
        embedding=self.atom_embedding(types)
        scalars=torch.stack([torch.log1p(r2/length.square()),torch.log1p(h2/length.square()),(rn*hn).sum(-1),
            t[batch],t[batch].square(),torch.log1p(counts[batch])/math.log(201.),length/2],-1)
        features=torch.cat([embedding[i]+embedding[j],embedding[i]*embedding[j],scalars],-1)
        coefficients=torch.tanh(self.pair_network(features));contact=torch.sigmoid((1.25-(h2+1e-12).sqrt()/length)/.2)
        degree=x.new_zeros(len(x)).index_add(0,i,contact).index_add(0,j,contact)
        messages=contact[:,None]*(coefficients[:,:1]*rn+coefficients[:,1:]*hn)/2
        accumulated=x.new_zeros(x.shape).index_add(0,i,messages).index_add(0,j,-messages)
        # Each row average has norm <=1. Subtracting the graph mean has norm
        # <=2; divide by2 before applying the same original2*t^2 bound.
        averaged=accumulated/degree.clamp_min(1e-12)[:,None]
        centered=center_by_graph(averaged,node_batch_idx,n_graphs)/2
        return self.velocity_scale*t[node_batch_idx,None].pow(self.gate_power)*centered


def balanced_force_shift(force,reference_shift,*,force_floor=1.):
    """Redistribute a fixed displacement norm using positive atomwise mobility.

For centered force F and P the coordinate-centering projector, let
M=diag(1/sqrt(||F_i||^2+floor^2)). The shift is a*P*M*P*F, with a>0
chosen to match the reference displacement norm. Thus F dot shift >=0:
the direction lowers a linearized energy. It is also the exact mean shift of
a local Gaussian energy tilt with covariance kT*a*P*M*P. No nonlinear-energy
decrease or final sampler distribution is guaranteed by this local identity.
"""
    if force.ndim!=3 or force.shape!=reference_shift.shape or force.shape[-1]!=3 or not math.isfinite(force_floor) or force_floor<=0:
        raise ValueError('Require matched batch/atom/vector shapes and positive force floor')
    if not torch.isfinite(force).all() or not torch.isfinite(reference_shift).all():raise ValueError('Nonfinite force target')
    force=force-force.mean(1,keepdim=True);reference_shift=reference_shift-reference_shift.mean(1,keepdim=True)
    mobility=(force.square().sum(-1)+force_floor**2).rsqrt()
    direction=mobility[...,None]*force;direction-=direction.mean(1,keepdim=True)
    length=direction.flatten(1).norm(dim=-1);desired=reference_shift.flatten(1).norm(dim=-1)
    if ((length==0)&(desired>0)).any():raise ValueError('A zero force cannot supply a nonzero reference displacement')
    scale=torch.where(length>0,desired/length.clamp_min(1e-30),torch.ones_like(length))
    return scale[:,None,None]*direction,mobility,scale
