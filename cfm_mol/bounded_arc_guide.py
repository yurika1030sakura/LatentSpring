"""Bounded log-score correction to the physical site-arc proposal.

This is a proposal score, not a physical-energy surrogate. The radial readout
architecture is shared with ConditionalArcEnergy; its coefficients here carry
no physical-energy interpretation.
"""
import torch
from cfm_mol.conditional_arc_energy import ConditionalArcEnergy
from cfm_mol.normalized_site_guide import physical_site_parameter


class BoundedArcGuide(ConditionalArcEnergy):
    score_semantics='bounded_log_score'

    def __init__(self,hidden=16,radial=16,query_radial=24,site_concentration=64.,cutoff=6.,restraint=.1,log_score_bound=1.):
        if not 0<log_score_bound<float('inf'):raise ValueError('Positive finite log-score bound required')
        super().__init__(hidden=hidden,radial=radial,query_radial=query_radial,coefficient_bound_eV=1.,
            site_concentration=site_concentration,cutoff=cutoff,restraint=restraint)
        self.log_score_bound=log_score_bound
        self.configuration=dict(hidden=hidden,radial=radial,query_radial=query_radial,site_concentration=site_concentration,
            cutoff=cutoff,restraint=restraint,log_score_bound=log_score_bound)

    def encode(self,x,bonds,numbers,electronic,roots):
        context=super().encode(x,bonds,numbers,electronic,roots)
        context['site_parameter']=physical_site_parameter(context['masked'],context['roles'],bonds,roots,self.site_concentration)
        return context

    def log_score(self,directions,context):
        shape=directions.shape[:-1]
        u=directions.reshape(len(directions),-1,3)
        site=(u*context['site_parameter'][:,None]).sum(-1).reshape(shape)
        raw=self.radial_readout(directions,context)
        return site+self.log_score_bound*torch.tanh(raw/self.log_score_bound)

    def circle_log_score(self,context,kT):
        if len(context['radius'])!=1:raise ValueError('One context required')
        return lambda directions:self.log_score(directions[None],context)[0]

    def energy(self,*args,**kwargs):
        raise TypeError('BoundedArcGuide is a proposal score, not a physical-energy predictor')
