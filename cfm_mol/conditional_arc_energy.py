"""Masked context predicts radial interaction curves for a queried root direction.

The output is a conditional energy surrogate in eV, up to a context constant.
It is neither an oracle nor a normalized probability density. Great-circle
normalization is performed separately using the actual interpolated score law.
"""
import math
import torch
from torch import nn

from cfm_mol.local_site_guide import LocalSiteGuide, smooth_cutoff
from cfm_mol.masked_angular_guide import masked_angular_context
from cfm_mol.chemical_moves import covalent_radii


class ConditionalArcEnergy(LocalSiteGuide):
    """O(3)-invariant scalar energy with context-only coefficients, cached per root."""
    def __init__(self, hidden=16, radial=16, query_radial=24, coefficient_bound_eV=2.,
                 site_concentration=64., cutoff=6., restraint=.1):
        super().__init__(hidden=hidden, radial=radial, bound=64.,
            site_concentration=site_concentration, cutoff=cutoff, restraint=restraint)
        if query_radial<2 or not 0<coefficient_bound_eV<float('inf'):
            raise ValueError('Finite positive coefficient bound and radial basis required')
        self.coefficient_bound_eV=coefficient_bound_eV
        self.query_head=nn.Sequential(nn.Linear(3*hidden,hidden),nn.SiLU(),nn.Linear(hidden,query_radial))
        nn.init.zeros_(self.query_head[-1].weight); nn.init.zeros_(self.query_head[-1].bias)
        # Distances are normalized by the root/passive covalent-radius sum.
        self.register_buffer('query_centers',torch.linspace(0.,6.,query_radial))
        self.query_width=6./(query_radial-1)
        self.configuration=dict(hidden=hidden,radial=radial,query_radial=query_radial,
            coefficient_bound_eV=coefficient_bound_eV,site_concentration=site_concentration,
            cutoff=cutoff,restraint=restraint)

    def encode(self,x,bonds,numbers,electronic,roots):
        site,harmonic,_,nodes=self.components(x,bonds,numbers,electronic,roots,return_nodes=True)
        masked,radius,roles=masked_angular_context(x,roots)
        if electronic.shape==(3,):electronic=electronic.expand(len(x),-1)
        if (electronic[:,2]<=0).any():raise ValueError('Positive temperature required')
        batch=torch.arange(len(x),device=x.device)
        root=nodes[batch,roots[:,0]][:,None].expand_as(nodes)
        anchor=nodes[batch,roots[:,1]][:,None].expand_as(nodes)
        raw=self.query_head(torch.cat([nodes,root,anchor],-1))
        coeff=self.coefficient_bound_eV*torch.tanh(raw/self.coefficient_bound_eV)
        radii=covalent_radii(numbers).to(dtype=x.dtype,device=x.device)
        pair_radii=radii[None]+radii[roots[:,0]][:,None]
        return dict(masked=masked,radius=radius,roles=roles,coefficients=coeff,pair_radii=pair_radii,
            linear_energy_parameter=-electronic[:,2:3]*(site+harmonic))

    def energy(self,directions,context):
        """directions[B,...,3] -> energy[B,...]; context has no root direction."""
        if directions.shape[0]!=len(context['radius']) or directions.shape[-1]!=3:
            raise ValueError('One direction array per encoded context required')
        original=directions.shape[:-1]
        u=directions.reshape(len(directions),-1,3)
        location=context['radius'][:,None,None]*u
        vectors=location[:,:,None,:]-context['masked'][:,None,:,:]
        distance=(vectors.square().sum(-1)+1e-12).sqrt()
        scaled=distance/context['pair_radii'][:,None]
        radial=torch.exp(-.5*((scaled[...,None]-self.query_centers)/self.query_width)**2)
        gate=smooth_cutoff(distance,self.cutoff)*(context['roles']!=1)[:,None]
        residual=(radial*context['coefficients'][:,None]).sum(-1)
        residual=(residual*gate).sum(-1)/((context['roles']!=1).sum(1).to(u.dtype).sqrt()[:,None])
        energy=(u*context['linear_energy_parameter'][:,None]).sum(-1)+residual
        return energy.reshape(original)

    def forward(self,x,bonds,numbers,electronic,roots,directions):
        return self.energy(directions,self.encode(x,bonds,numbers,electronic,roots))

    def circle_log_score(self,context,kT):
        """Callable for ONE frozen context; every edge gets the actual same score."""
        if len(context['radius'])!=1 or not 0<float(kT)<float('inf'):
            raise ValueError('One context and positive temperature required')
        def score(directions):
            return -self.energy(directions[None],context)[0]/kT
        return score
