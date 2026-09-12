"""General positive pre-query gates using only the current state's cached force."""
import math
import torch
from torch import nn
from cfm_mol.delayed_acceptance import paired_screen_features
from cfm_mol.chemical_moves import covalent_radii
from cfm_mol.local_site_guide import smooth_cutoff


def screened_log_acceptance(log_ratio,log_forward_gate,log_reverse_gate):
    if not (log_ratio.shape==log_forward_gate.shape==log_reverse_gate.shape):
        raise ValueError('Matched ratio and gate shapes required')
    if any(not torch.isfinite(g).all() or (g>0).any() for g in [log_forward_gate,log_reverse_gate]):
        raise ValueError('Finite log gate probabilities <=0 required')
    if torch.isnan(log_ratio).any() or torch.isposinf(log_ratio).any():
        raise ValueError('Finite or minus-infinity MH ratio required')
    second=torch.minimum(torch.zeros_like(log_ratio),log_ratio+log_reverse_gate-log_forward_gate)
    return second,log_forward_gate+second


def source_features(x,y,bonds,new_bonds,numbers,electronic,proposal_ratio,force,restraint):
    if force.shape!=x.shape or not torch.isfinite(force).all():
        raise ValueError('Finite cached SOURCE force required')
    features=paired_screen_features(x,y,bonds,new_bonds,numbers,electronic,proposal_ratio,restraint)
    delta=(y-y.mean(0))-(x-x.mean(0))
    work=(force*delta).sum()/electronic[2]
    return torch.cat([features,work[None]])


class ForcePairEncoder(nn.Module):
    """Invariant paired-geometry encoder with atomwise source-force work inputs."""
    def __init__(self,hidden=16,radial=16):
        super().__init__()
        z=torch.arange(119,dtype=torch.float64)
        self.register_buffer('atomic_features',torch.stack([z/118,torch.log1p(z)/math.log(119),covalent_radii(range(119))/2],1))
        self.register_buffer('centers',torch.linspace(0,6,radial))
        self.atom=nn.Sequential(nn.Linear(3,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        self.state=nn.Sequential(nn.Linear(3,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        self.force=nn.Sequential(nn.Linear(3,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        self.roles=nn.Embedding(3,hidden)
        self.messages=nn.ModuleList([nn.Sequential(nn.Linear(2*hidden+2*radial+2,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU()) for _ in range(2)])
        self.updates=nn.ModuleList([nn.Sequential(nn.Linear(2*hidden,hidden),nn.SiLU(),nn.Linear(hidden,hidden)) for _ in range(2)])
        self.head=nn.Linear(hidden,1);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,x,y,bonds,new_bonds,numbers,electronic,action,force,restraint):
        n=len(x);x=x-x.mean(0);y=y-y.mean(0);delta=y-x
        radii=covalent_radii(numbers).to(x)
        f=force-restraint*x;f=f-f.mean(0)
        scaled_force=f*radii[:,None]/electronic[2];scaled_delta=delta/radii[:,None]
        scalars=torch.stack([torch.asinh((scaled_force*scaled_delta).sum(1)),
            torch.log1p(scaled_force.norm(dim=1)),torch.log1p(scaled_delta.norm(dim=1))],1)
        roles=torch.zeros(n,dtype=torch.long,device=x.device);i,j,k,l=action
        roles[[i,j]]=1;roles[[k,l]]=2
        nodes=self.atom(self.atomic_features[numbers].to(x))+self.state(electronic)[None]+self.force(scalars)+self.roles(roles)
        dx=((x[:,None]-x[None,:]).square().sum(-1)+1e-8).sqrt()
        dy=((y[:,None]-y[None,:]).square().sum(-1)+1e-8).sqrt()
        rsum=radii[:,None]+radii[None,:]
        radial=torch.cat([torch.exp(-.5*((d[...,None]/rsum[...,None]-self.centers)/.4)**2) for d in [dx,dy]],-1)
        gate=torch.maximum(smooth_cutoff(dx,6.),smooth_cutoff(dy,6.))*(~torch.eye(n,dtype=torch.bool,device=x.device))
        for message,update in zip(self.messages,self.updates):
            a=nodes[:,None].expand(-1,n,-1);b=nodes[None,:].expand(n,-1,-1)
            pair=message(torch.cat([a+b,(a-b).square(),radial,bonds[...,None]/3,new_bonds[...,None]/3],-1))
            aggregate=(pair*gate[...,None]).sum(1)/gate.sum(1).clamp_min(1)[:,None]
            nodes=nodes+update(torch.cat([nodes,aggregate],-1))
        return self.head(nodes).sum()/math.sqrt(n)


class SourceForceScreen(nn.Module):
    def __init__(self,variant,log_factor_bound=math.log(16),restraint=.1,hidden=16,radial=16,thinning_probability=1.):
        super().__init__()
        if variant not in ('zero','physical','work','linear','neural','thinning'):
            raise ValueError('Unknown force-screen variant')
        if (not 0<log_factor_bound<float('inf') or not 0<=restraint<float('inf')
                or not math.exp(-log_factor_bound)<=thinning_probability<=1 or hidden<1 or radial<2):
            raise ValueError('Finite bound/restraint and positive thinning probability required')
        self.variant,self.log_factor_bound,self.restraint=variant,log_factor_bound,restraint
        self.thinning_probability=thinning_probability
        initial=([1.,1.,1.,1.,0.] if variant=='physical' else [0.,0.,1.,1.,1.] if variant=='work' else [0.]*5)
        self.coefficients=nn.Parameter(torch.tensor(initial),requires_grad=variant in ('linear','neural'))
        self.encoder=ForcePairEncoder(hidden,radial) if variant=='neural' else None
        self.configuration=dict(variant=variant,log_factor_bound=log_factor_bound,restraint=restraint,hidden=hidden,radial=radial,thinning_probability=thinning_probability)

    def log_pass_probability(self,x,y,bonds,new_bonds,numbers,electronic,action,proposal_log_ratio,source_force):
        f=source_features(x,y,bonds,new_bonds,numbers,electronic,proposal_log_ratio,source_force,self.restraint)
        if self.variant=='thinning':return x.new_tensor(math.log(self.thinning_probability))
        raw=(self.coefficients*f).sum()
        if self.encoder is not None:raw=raw+self.encoder(x,y,bonds,new_bonds,numbers,electronic,action,source_force,self.restraint)
        score=self.log_factor_bound*torch.tanh(raw/self.log_factor_bound)
        if not torch.isfinite(score):raise ValueError('Nonfinite source-force gate')
        return torch.minimum(torch.zeros_like(score),score)


def force_edge_values(model,row):
    zero=sum(p.sum()*0 for p in model.parameters())
    if not row['valid']:return zero,zero,zero,zero,zero,zero
    proposal=row['log_behavior_reverse']-row['log_behavior_forward']+row['action_log_ratio']
    common=(row['numbers'],row['electronic'])
    forward=model.log_pass_probability(row['x'],row['y'],row['bonds'],row['new_bonds'],*common,row['action'],proposal,row['source_force_eV_A'])
    reverse=model.log_pass_probability(row['y'],row['x'],row['new_bonds'],row['bonds'],*common,row['inverse_action'],-proposal,row['candidate_force_eV_A'])
    _,total=screened_log_acceptance(row['target_log_ratio']+proposal,forward,reverse)
    acceptance=total.exp()
    return row['reward_eV']*acceptance,2*forward.exp(),acceptance*int(row['connectivity_changed']),forward,reverse,acceptance
