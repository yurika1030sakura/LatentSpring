"""Invariant policy over local MALA and reversible terminal-site exchanges.

Graphs are observed/validated states, not bond-supervision targets. Both move
family and action probabilities must enter the Metropolis reverse ratio.
"""
import torch
from torch import nn


class ChemicalMovePolicy(nn.Module):
    def __init__(self,hidden=16,radial=16,family_floor=.1,action_uniform_fraction=.1):
        super().__init__()
        if not 0<family_floor<.5 or not 0<action_uniform_fraction<=1:
            raise ValueError('Positive defensive probabilities are required')
        self.configuration=dict(hidden=hidden,radial=radial,family_floor=family_floor,
            action_uniform_fraction=action_uniform_fraction)
        self.family_floor=family_floor;self.action_uniform_fraction=action_uniform_fraction
        self.elements=nn.Embedding(119,hidden)
        self.state=nn.Sequential(nn.Linear(3,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        self.register_buffer('centers',torch.linspace(0,8,radial))
        self.messages=nn.ModuleList([nn.Sequential(nn.Linear(2*hidden+radial+1,hidden),nn.SiLU(),
            nn.Linear(hidden,hidden),nn.SiLU()) for _ in range(2)])
        self.updates=nn.ModuleList([nn.Sequential(nn.Linear(2*hidden,hidden),nn.SiLU(),nn.Linear(hidden,hidden)) for _ in range(2)])
        self.family_head=nn.Linear(hidden,1)
        self.action_head=nn.Sequential(nn.Linear(4*hidden+3,hidden),nn.SiLU(),nn.Linear(hidden,1))
        for head in [self.family_head,self.action_head[-1]]:
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def forward(self,x,bonds,numbers,electronic,actions,mask=None):
        if (x.ndim!=3 or x.shape[-1]!=3 or not 2<=x.shape[1]<=200
                or bonds.shape!=x.shape[:2]+(x.shape[1],) or numbers.shape!=(x.shape[1],)
                or actions.ndim!=3 or actions.shape[0]!=len(x) or actions.shape[-1]!=4):
            raise ValueError('Invalid molecular-policy batch shapes')
        if numbers.dtype!=torch.long or actions.dtype!=torch.long:raise ValueError('Integer identities/actions required')
        if electronic.shape==(3,):electronic=electronic.expand(len(x),-1)
        if electronic.shape!=(len(x),3) or not all(torch.isfinite(v).all() for v in [x,bonds,electronic]):
            raise ValueError('Finite state inputs required')
        n=x.shape[1];distance=((x[:,:,None]-x[:,None,:]).square().sum(-1)+1e-8).sqrt()
        radial=torch.exp(-.5*((distance[...,None]-self.centers)/.4)**2)
        nodes=self.elements(numbers)[None]+self.state(electronic)[:,None]
        pair_mask=(~torch.eye(n,dtype=torch.bool,device=x.device))[None,:,:,None]
        for message,update in zip(self.messages,self.updates):
            a=nodes[:,:,None].expand(-1,-1,n,-1);b=nodes[:,None,:].expand(-1,n,-1,-1)
            pairs=message(torch.cat([a+b,(a-b).square(),radial,bonds[...,None]/3],-1))
            nodes=nodes+update(torch.cat([nodes,(pairs*pair_mask).sum(2)/n],-1))
        if mask is None:mask=actions[...,0]>=0
        if mask.shape!=actions.shape[:2] or mask.dtype!=torch.bool:raise ValueError('Boolean action mask required')
        if (((actions<0)|(actions>=n))&mask[...,None]).any():
            raise ValueError('Action index outside the molecule')
        if actions.shape[1]==0:
            return nodes.sum((1,2))[:,None]*0
        safe=torch.where(mask[...,None],actions,torch.zeros_like(actions))
        batch=torch.arange(len(x),device=x.device)[:,None]
        i,j,k,l=safe.unbind(-1)
        hi,hj,hk,hl=[nodes[batch,index] for index in [i,j,k,l]]
        di=distance[batch,i,k];dj=distance[batch,j,l];dij=distance[batch,i,j]
        first=torch.cat([hi,hj,hk,hl,di[...,None],dj[...,None],dij[...,None]],-1)
        second=torch.cat([hj,hi,hl,hk,dj[...,None],di[...,None],dij[...,None]],-1)
        logits=.5*(self.action_head(first)[...,0]+self.action_head(second)[...,0])
        counts=mask.sum(1);available=counts>0
        local=self.family_floor+(1-2*self.family_floor)*self.family_head(nodes.mean(1))[:,0].sigmoid()
        local=torch.where(available,local,torch.ones_like(local))
        logits=logits.masked_fill(~mask,-torch.inf)
        logits=torch.where(available[:,None],logits,torch.zeros_like(logits))
        conditional=((1-self.action_uniform_fraction)*logits.softmax(1)*mask+
            self.action_uniform_fraction*mask/counts.clamp_min(1)[:,None])
        probabilities=torch.cat([local[:,None],(1-local)[:,None]*conditional],1)
        logp=probabilities.clamp_min(torch.finfo(x.dtype).tiny).log()
        legal=torch.cat([torch.ones(len(x),1,dtype=torch.bool,device=x.device),mask],1)
        return logp.masked_fill(~legal,-torch.inf)


def accepted_action_mass(log_forward,log_reverse,base_log_ratio):
    """p(a|x) alpha(x,a), for fixed proposals and an involutive reverse action.

    base_log_ratio includes target difference and map Jacobian (or the local
    Gaussian proposal ratio). Neither a state-dependent family probability nor
    its reverse may be omitted. Enumeration avoids a categorical score estimate.
    This is standard MH probability flow, not a new sampling identity.
    """
    return torch.minimum(log_forward,base_log_ratio+log_reverse).exp()
