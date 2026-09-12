"""Local normalized angular correction with an analytic COM-restraint term.

Locality is an architectural property of the learned residual, not a claim that
all electronic forces are short ranged. The known global restraint is separate.
"""
import math
import torch
from cfm_mol.masked_angular_guide import masked_angular_context
from cfm_mol.normalized_site_guide import NormalizedSiteGuide,physical_site_parameter


def confinement_parameter(masked,radius,kT,restraint):
    """Exact angular log-density coefficient of gamma/2 times sum |x_COM|^2."""
    if restraint<0 or not math.isfinite(restraint) or not torch.isfinite(kT).all() or (kT<=0).any():
        raise ValueError('Nonnegative restraint and positive finite temperature required')
    return restraint*radius[:,None]*masked.sum(1)/(masked.shape[1]*kT[:,None])


def smooth_cutoff(distance,cutoff):
    return torch.where(distance<cutoff,.5*(1+torch.cos(math.pi*distance/cutoff)),torch.zeros_like(distance))


class LocalSiteGuide(NormalizedSiteGuide):
    def __init__(self,hidden=16,radial=16,bound=64.,site_concentration=10.,cutoff=6.,restraint=.1):
        super().__init__(hidden=hidden,radial=radial,bound=bound,mixture=False,site_concentration=site_concentration)
        if not math.isfinite(cutoff) or cutoff<=0 or not math.isfinite(restraint) or restraint<0:
            raise ValueError('Positive finite cutoff and nonnegative restraint required')
        self.cutoff,self.restraint=cutoff,restraint
        del self.component_weight,self.offset_weight
        self.radial_centers.copy_(torch.linspace(0,cutoff,radial))
        self.configuration=dict(hidden=hidden,radial=radial,bound=bound,site_concentration=site_concentration,
            cutoff=cutoff,restraint=restraint)

    def components(self,x,bonds,numbers,electronic,roots):
        if (not 2<=x.shape[1]<=200 or bonds.shape!=x.shape[:2]+(x.shape[1],)
                or numbers.shape!=(x.shape[1],) or numbers.dtype!=torch.long
                or ((numbers<1)|(numbers>118)).any()):
            raise ValueError('Invalid molecular guide batch')
        masked,radius,roles=masked_angular_context(x,roots)
        if electronic.shape==(3,):electronic=electronic.expand(len(x),-1)
        if electronic.shape!=(len(x),3) or not torch.isfinite(electronic).all() or not torch.isfinite(bonds).all():
            raise ValueError('Finite graph and electronic context required')
        distance=((masked[:,:,None]-masked[:,None,:]).square().sum(-1)+1e-8).sqrt()
        radial=torch.exp(-.5*((distance[...,None]-self.radial_centers)/.4)**2)
        gate=smooth_cutoff(distance,self.cutoff)*(~torch.eye(x.shape[1],dtype=torch.bool,device=x.device))[None]
        denominator=gate.sum(2).clamp_min(1)[...,None]
        nodes=self.elements(self.atomic_features[numbers].to(x.dtype))[None]+self.roles(roles)
        nodes=nodes+self.state(torch.cat([electronic,radius[:,None]],1))[:,None]
        n=x.shape[1]
        for message,update in zip(self.messages,self.updates):
            a=nodes[:,:,None].expand(-1,-1,n,-1);b=nodes[:,None,:].expand(-1,n,-1,-1)
            pair=message(torch.cat([a+b,(a-b).square(),radial,bonds[...,None]/3],-1))
            aggregate=(pair*gate[...,None]).sum(2)/denominator
            nodes=nodes+update(torch.cat([nodes,aggregate],-1))
        readout_gate=smooth_cutoff(masked.norm(dim=2),self.cutoff)
        vectors=masked/(masked.square().sum(-1,keepdim=True)+.01).sqrt()
        residual=(self.center_weight(nodes)*readout_gate[...,None]*vectors).sum(1)
        residual=residual/readout_gate.square().sum(1).clamp_min(1).sqrt()[:,None]
        residual=residual*self.bound/(self.bound+residual.norm(dim=1,keepdim=True))
        site=physical_site_parameter(masked,roles,bonds,roots,self.site_concentration)
        harmonic=confinement_parameter(masked,radius,electronic[:,2],self.restraint)
        return site,harmonic,residual

    def forward(self,x,bonds,numbers,electronic,roots):
        site,harmonic,residual=self.components(x,bonds,numbers,electronic,roots)
        return (site+harmonic+residual)[:,None],x.new_zeros(len(x),1)
