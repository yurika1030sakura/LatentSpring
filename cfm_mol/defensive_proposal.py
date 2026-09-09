"""Defensive proposals using the Gaussian supplied by harmonic confinement.

This is classical defensive importance sampling. If U=E+kappa*||z||^2/2,
kT>0 and E is bounded below, q_G=N(0,(kT/kappa)I) dominates exp(-U/kT)
up to a finite constant. Mixing epsilon*q_G into any normalized local proposal
therefore bounds unnormalized importance weights and every geometric-bridge
weight product. The lower energy bound is an assumption, not an observed min.
"""
import math

import torch

from cfm_mol.tempered_smc import DensityValue,IsotropicGaussianMixture


class DefensiveProposal:
    def __init__(self,local,dimension,confinement_std,wide_fraction=.2):
        if dimension<1 or not 0<wide_fraction<1 or not math.isfinite(wide_fraction):
            raise ValueError('Require positive dimension and defensive fraction in (0,1)')
        self.local=local;self.wide_fraction=wide_fraction;self.confinement_std=confinement_std
        device=local.centers.device
        self.wide=IsotropicGaussianMixture(torch.zeros(1,dimension,dtype=torch.float64,device=device),confinement_std)

    def sample(self,n,generator):
        if n<1:raise ValueError('Require positive sample count')
        wide=torch.rand(n,device=self.wide.centers.device,dtype=torch.float64,generator=generator)<self.wide_fraction
        narrow,labels=self.local.sample(n,generator)
        broad,_=self.wide.sample(n,generator)
        return torch.where(wide[:,None],broad,narrow),labels.masked_fill(wide,-1)

    def __call__(self,x):
        narrow=self.local(x);broad=self.wide(x)
        terms=torch.stack([narrow.log_value+math.log1p(-self.wide_fraction),
                           broad.log_value+math.log(self.wide_fraction)],-1)
        responsibilities=torch.softmax(terms,-1)
        score=responsibilities[:,0:1]*narrow.score+responsibilities[:,1:2]*broad.score
        return DensityValue(torch.logsumexp(terms,-1),score)
