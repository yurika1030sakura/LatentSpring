"""Masked equivariant angular energy guide with an invariant conditioning context.

Outputs are unnormalized angular scores, not normalized support likelihoods.
The moved leaf's direction is removed before ALL geometric/message features.
"""
import math
import torch
from torch import nn


def masked_angular_context(x,roots):
    if x.ndim!=3 or x.shape[-1]!=3 or roots.shape!=(len(x),2) or roots.dtype!=torch.long:
        raise ValueError('Molecular batches and integer leaf/anchor pairs required')
    if not torch.isfinite(x).all() or ((roots<0)|(roots>=x.shape[1])).any() or (roots[:,0]==roots[:,1]).any():
        raise ValueError('Finite coordinates and distinct valid roots required')
    batch=torch.arange(len(x),device=x.device);leaf,anchor=roots.unbind(1)
    relative=x-x[batch,anchor][:,None]
    radius=relative[batch,leaf].norm(dim=1)
    if (radius<=1e-10).any():raise ValueError('Positive retained bond radius required')
    role=torch.zeros(x.shape[:2],dtype=torch.long,device=x.device)
    role[batch,leaf]=1;role[batch,anchor]=2
    masked=relative.masked_fill((role==1)[...,None],0.)
    return masked,radius,role


class MaskedAngularGuide(nn.Module):
    def __init__(self,hidden=16,radial=16,bound=64.,tensor=True):
        super().__init__()
        if hidden<1 or radial<2 or bound<=0:raise ValueError('Positive model dimensions and bound required')
        self.configuration=dict(hidden=hidden,radial=radial,bound=bound,tensor=tensor)
        self.bound=bound;self.use_tensor=tensor
        self.elements=nn.Embedding(119,hidden);self.roles=nn.Embedding(3,hidden)
        self.state=nn.Sequential(nn.Linear(4,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        self.register_buffer('radial_centers',torch.linspace(0,8,radial))
        self.messages=nn.ModuleList([nn.Sequential(nn.Linear(2*hidden+radial+1,hidden),nn.SiLU(),
            nn.Linear(hidden,hidden),nn.SiLU()) for _ in range(2)])
        self.updates=nn.ModuleList([nn.Sequential(nn.Linear(2*hidden,hidden),nn.SiLU(),nn.Linear(hidden,hidden)) for _ in range(2)])
        self.vector_weight=nn.Linear(hidden,1);self.tensor_weight=nn.Linear(hidden,1)
        for head in [self.vector_weight,self.tensor_weight]:nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def forward(self,x,bonds,numbers,electronic,roots):
        if (not 2<=x.shape[1]<=200 or bonds.shape!=x.shape[:2]+(x.shape[1],)
                or numbers.shape!=(x.shape[1],) or numbers.dtype!=torch.long):
            raise ValueError('Invalid molecular guide batch')
        w,radius,roles=masked_angular_context(x,roots)
        if electronic.shape==(3,):electronic=electronic.expand(len(x),-1)
        if electronic.shape!=(len(x),3) or not torch.isfinite(electronic).all() or not torch.isfinite(bonds).all():
            raise ValueError('Finite graph/electronic inputs required')
        n=x.shape[1];distance=((w[:,:,None]-w[:,None,:]).square().sum(-1)+1e-8).sqrt()
        radial=torch.exp(-.5*((distance[...,None]-self.radial_centers)/.4)**2)
        nodes=self.elements(numbers)[None]+self.roles(roles)+self.state(torch.cat([electronic,radius[:,None]],1))[:,None]
        pair_mask=(~torch.eye(n,dtype=torch.bool,device=x.device))[None,:,:,None]
        for message,update in zip(self.messages,self.updates):
            a=nodes[:,:,None].expand(-1,-1,n,-1);b=nodes[:,None,:].expand(-1,n,-1,-1)
            pair=message(torch.cat([a+b,(a-b).square(),radial,bonds[...,None]/3],-1))
            nodes=nodes+update(torch.cat([nodes,(pair*pair_mask).sum(2)/n],-1))
        vectors=w/(w.square().sum(-1,keepdim=True)+.01).sqrt()
        eta=(self.vector_weight(nodes)*vectors).sum(1)/math.sqrt(n)
        eye=torch.eye(3,dtype=x.dtype,device=x.device)
        dyads=vectors[...,None]*vectors[:,:,None,:]-vectors.square().sum(-1)[:,:,None,None]*eye/3
        matrix=(self.tensor_weight(nodes)[...,None]*dyads).sum(1)/math.sqrt(n)
        if not self.use_tensor:matrix=matrix*0
        matrix=.5*(matrix+matrix.transpose(1,2));matrix=matrix-matrix.diagonal(dim1=1,dim2=2).sum(1)[:,None,None]*eye/3
        norm=eta.norm(dim=1)+matrix.flatten(1).norm(dim=1)
        scale=self.bound/(self.bound+norm);eta=eta*scale[:,None];matrix=matrix*scale[:,None,None]
        envelope=eta.norm(dim=1)+torch.linalg.eigvalsh(matrix)[:,-1]
        return eta,matrix,envelope


def angular_log_score(direction,eta,matrix):
    return (direction*eta).sum(-1)+torch.einsum('bi,bij,bj->b',direction,matrix,direction)


def angular_surface_score(direction,eta,matrix):
    raw=eta+2*torch.einsum('bij,bj->bi',matrix,direction)
    return raw-(raw*direction).sum(1,keepdim=True)*direction
