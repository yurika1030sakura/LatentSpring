"""Learn graph-conditioned correlated motion without coordinate-Jacobian estimates."""
import math
import torch
from torch import nn
from cfm_mol.edit_conditioned_bridge import EditBridgeField
from cfm_mol.mobility_relaxation import mobility_basis
from cfm_mol.chemical_moves import covalent_radii


class GraphEditMobility(nn.Module):
    """PSD operator on atom space; independent of active coordinates/momentum.

    Only graph/electronic context is used, with exact graph/time reversal tying.
    It acts in phase-space shears, so no determinant of this operator enters MH.
    """
    def __init__(self,variant,hidden=12,rank=3,passive_scale=.01,coefficient_bound=.25):
        super().__init__()
        if variant not in ['fixed','scalar','collective'] or not 0<passive_scale<=1:raise ValueError('Valid mobility variant and passive scale required')
        self.variant=variant;self.rank=rank;self.passive_scale=passive_scale;self.coefficient_bound=coefficient_bound
        z=torch.arange(119,dtype=torch.float64)
        self.register_buffer('atomic_features',torch.stack([z/118,torch.log1p(z)/math.log(119),covalent_radii(range(119))/2],1))
        self.atom=nn.Sequential(nn.Linear(3,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        self.context=nn.Sequential(nn.Linear(5,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        self.messages=nn.ModuleList([nn.Sequential(nn.Linear(2*hidden+3,hidden),nn.SiLU(),nn.Linear(hidden,hidden)) for _ in range(2)])
        self.updates=nn.ModuleList([nn.Sequential(nn.Linear(2*hidden,hidden),nn.SiLU(),nn.Linear(hidden,hidden)) for _ in range(2)])
        self.vectors=nn.Linear(hidden,rank);self.coefficients=nn.Linear(hidden,rank);self.scale=nn.Linear(hidden,1)
        for head in [self.coefficients,self.scale]:nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def forward(self,bonds,new_bonds,numbers,electronic,action,t):
        n=len(numbers);e=electronic;identity=torch.eye(n,dtype=e.dtype,device=e.device);P=identity-torch.ones_like(identity)/n
        root=mobility_basis(n,list(action[:2]),'roots').to(e);R=root@root.T
        A=R+self.passive_scale**.5*(P-R)
        if self.variant=='fixed':return A@A.T
        change=new_bonds-bonds;edges=torch.stack([(1-t)*bonds+t*new_bonds,change.abs(),(2*t-1)*change],-1)/3
        role=change.abs().sum(1).clamp_max(4)/4
        features=torch.stack([e[0].expand(n)/4,e[1].expand(n)/4,e[2].log().expand(n)/4,role,e.new_full((n,),t*(1-t))],1)
        nodes=self.atom(self.atomic_features[numbers].to(e))+self.context(features)
        gate=((bonds>0)|(new_bonds>0)).to(e)
        for message,update in zip(self.messages,self.updates):
            a=nodes[:,None].expand(-1,n,-1);b=nodes[None,:].expand(n,-1,-1)
            pairs=message(torch.cat([a+b,(a-b).square(),edges],-1))
            nodes=nodes+update(torch.cat([nodes,(pairs*gate[...,None]).sum(1)/gate.sum(1).clamp_min(1)[:,None]],-1))
        pooled=nodes.mean(0);scale=torch.exp(math.log(2)*self.scale(pooled).squeeze().tanh())
        if self.variant=='collective':
            V=P@self.vectors(nodes);V=V/(V.square().sum(0,keepdim=True)+1e-4).sqrt()
            c=self.coefficient_bound/self.rank*self.coefficients(pooled).tanh()
            A=A+(V*c[None])@V.T
        return scale*(A@A.T)


class MobilizedEditField(nn.Module):
    roots_only=False
    def __init__(self,force_configuration,mobility_configuration):
        super().__init__();self.configuration=dict(force_configuration=force_configuration,mobility_configuration=mobility_configuration)
        self.force_field=EditBridgeField(**dict(force_configuration,roots_only=False))
        self.mobility=GraphEditMobility(**mobility_configuration)
    def forward(self,*args):return self.force_field(*args)


def differentiable_oracle_potential(x,energy,force,restraint):
    """Attach the actual even-potential derivative; energy/force must be detached."""
    physical=energy.detach()+((x-x.detach())*(-force.detach())).sum((-1,-2))
    return physical+restraint/2*x.square().sum((-1,-2))


def graph_penalty(x,desired,numbers):
    """Smooth training penalty only; never added to the evaluation target."""
    radii=covalent_radii(numbers).to(x);scale=radii[:,None]+radii[None,:]
    distance=((x[:,None]-x[None,:]).square().sum(-1)+1e-8).sqrt()/scale
    mask=torch.triu(torch.ones_like(desired,dtype=torch.bool),1);bonded=desired>0
    penalty=torch.where(bonded,(.65-distance).clamp_min(0).square()+(distance-1.2).clamp_min(0).square(),(1.3-distance).clamp_min(0).square())
    return penalty[mask].sum()
