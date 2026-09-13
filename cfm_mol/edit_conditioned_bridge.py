"""Edit-conditioned all-atom transport using reversible phase-space shears.

This is a candidate architecture, not an established molecular advantage.
Learned leapfrog/MH principles are prior art. Only the middle chemical map has
nonunit volume; intermediate learned fields require no physical oracle queries.
"""
import math
import torch
from torch import nn
from cfm_mol.chemical_moves import covalent_radii,exchange_terminal_sites


def center(x):return x-x.mean(0,keepdim=True)


class EditBridgeField(nn.Module):
    def __init__(self,hidden=24,radial=12,layers=2,force_scale=40.,edit_conditioned=True,roots_only=False):
        super().__init__()
        if min(hidden,radial,layers)<1 or not 0<force_scale<float('inf'):raise ValueError('Positive field dimensions and scale required')
        self.configuration=dict(hidden=hidden,radial=radial,layers=layers,force_scale=force_scale,edit_conditioned=edit_conditioned,roots_only=roots_only)
        self.force_scale=force_scale;self.edit_conditioned=edit_conditioned;self.roots_only=roots_only
        z=torch.arange(119,dtype=torch.float64)
        self.register_buffer('atomic_features',torch.stack([z/118,torch.log1p(z)/math.log(119),covalent_radii(range(119))/2],1))
        self.register_buffer('centers',torch.linspace(0,4,radial))
        self.atom=nn.Sequential(nn.Linear(3,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        self.state=nn.Sequential(nn.Linear(5,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        self.messages=nn.ModuleList([nn.Sequential(nn.Linear(2*hidden+radial+3,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU()) for _ in range(layers)])
        self.updates=nn.ModuleList([nn.Sequential(nn.Linear(2*hidden,hidden),nn.SiLU(),nn.Linear(hidden,hidden)) for _ in range(layers)])
        self.head=nn.Linear(hidden,1);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,x,bonds,new_bonds,numbers,electronic,action,t):
        n=len(x);delta=x[:,None]-x[None,:];distance=(delta.square().sum(-1)+1e-8).sqrt()
        radii=covalent_radii(numbers).to(x);scale=radii[:,None]+radii[None,:]
        radial=torch.exp(-.5*((distance[...,None]/scale[...,None]-self.centers)/.4)**2)
        # These features are identical under (G,G',t) -> (G',G,1-t).
        change=new_bonds-bonds
        if self.edit_conditioned:
            edge=torch.stack([(1-t)*bonds+t*new_bonds,change.abs(),(2*t-1)*change],-1)/3
            role=change.abs().sum(1).clamp_max(4)/4
        else:edge=x.new_zeros(n,n,3);role=x.new_zeros(n)
        e=electronic.to(x);features=torch.stack([e[0].expand(n)/4,e[1].expand(n)/4,e[2].log().expand(n)/4,role,x.new_full((n,),t*(1-t))],1)
        nodes=self.atom(self.atomic_features[numbers].to(x))+self.state(features)
        gate=(~torch.eye(n,dtype=torch.bool,device=x.device)).to(x)*torch.exp(-(distance/6)**4)
        for message,update in zip(self.messages,self.updates):
            a=nodes[:,None].expand(-1,n,-1);b=nodes[None,:].expand(n,-1,-1)
            pair=message(torch.cat([a+b,(a-b).square(),radial,edge],-1))
            aggregate=(pair*gate[...,None]).sum(1)/gate.sum(1).clamp_min(1)[:,None]
            nodes=nodes+update(torch.cat([nodes,aggregate],-1))
        coefficients=self.head(pair+nodes[:,None]+nodes[None,:]).squeeze(-1).tanh()*gate
        force=self.force_scale*(coefficients[...,None]*delta/(distance[...,None]+.01)).sum(1)/gate.sum(1).clamp_min(1)[:,None]
        if self.roots_only:
            mask=x.new_zeros(n);mask[list(action[:2])]=1;force=force*mask[:,None]
        return center(force)


def edit_bridge(x,momentum,bonds,new_bonds,numbers,electronic,action,field,*,steps_per_side=2,kick_step=.2,drift_step=.05):
    """Return (y,p_new,log|J|), including final momentum reversal.

    The reverse call swaps the graphs and uses the inverse chemical action.
    Context may depend on the evolving x and declared graphs, never on an
    unrecorded original geometry. Endpoint graph support/action ratios belong
    in the sampling caller; this primitive does not call a physical oracle.
    """
    if (x.ndim!=2 or x.shape[-1]!=3 or momentum.shape!=x.shape or not 2<=len(x)<=200
            or not isinstance(steps_per_side,int) or steps_per_side<0 or not 0<kick_step<float('inf') or not 0<drift_step<float('inf')):
        raise ValueError('Finite positive steps, matching COM states and2--200 atoms required')
    if not torch.isfinite(x).all() or not torch.isfinite(momentum).all() or float(torch.stack([x.mean(0).abs().max(),momentum.mean(0).abs().max()]).max())>1e-8:
        raise ValueError('Finite centered positions and momentum required')
    y=x;p=momentum;radii=covalent_radii(numbers).to(x);log_volume=x.new_zeros(())
    mobility=getattr(field,'mobility',None)
    mobile_basis=None
    if getattr(field,'roots_only',False):
        from cfm_mol.mobility_relaxation import mobility_basis
        mobile_basis=mobility_basis(len(x),list(action[:2]),'roots').to(x)
    for stage in range(2*steps_per_side+1):
        if stage==steps_per_side:y,log_volume,_=exchange_terminal_sites(y,radii,action)
        if stage==2*steps_per_side:break
        left=stage/(2*steps_per_side);right=(stage+1)/(2*steps_per_side)
        control=field(y,bonds,new_bonds,numbers,electronic,action,left)
        if mobility is not None:control=mobility(bonds,new_bonds,numbers,electronic,action,left)@control
        p=center(p+.5*kick_step*control)
        velocity=p if mobile_basis is None else mobile_basis@(mobile_basis.T@p)
        if mobility is not None:velocity=mobility(bonds,new_bonds,numbers,electronic,action,(left+right)/2)@p
        y=center(y+drift_step*velocity)
        control=field(y,bonds,new_bonds,numbers,electronic,action,right)
        if mobility is not None:control=mobility(bonds,new_bonds,numbers,electronic,action,right)@control
        p=center(p+.5*kick_step*control)
    return y,-p,log_volume


def augmented_log_ratio(old_potential,new_potential,momentum,new_momentum,kT,log_volume,action_log_ratio):
    return -(new_potential-old_potential)/kT+.5*(momentum.square().sum()-new_momentum.square().sum())+log_volume+action_log_ratio
