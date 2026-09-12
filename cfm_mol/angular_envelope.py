"""Exact two-vMF envelope for a linear-plus-quadratic spherical score.

This is an implementation of an elementary rejection bound, not a claimed new
general directional-distribution theorem. Eigenvector signs exchange components.
"""
import math
import torch
from cfm_mol.spherical_proposal import vmf_log_normalizer,vmf_sample


def logcosh(x):return torch.logaddexp(x,-x)-math.log(2)


def envelope_parameters(eta,matrix):
    eigenvalues,eigenvectors=torch.linalg.eigh(matrix)
    gap=eigenvalues[:,-1]-eigenvalues[:,-2];axis=eigenvectors[:,:,-1]
    parameters=torch.stack([eta+gap[:,None]*axis,eta-gap[:,None]*axis],1)
    log_c=vmf_log_normalizer(parameters.reshape(-1,3)).reshape(len(eta),2)
    log_weights=(-log_c).log_softmax(1)
    bound=eigenvalues[:,-1]-logcosh(gap)
    return dict(gap=gap,axis=axis,parameters=parameters,log_weights=log_weights,bound=bound,
        log_base_partition=torch.logsumexp(-log_c,1)-math.log(2))


@torch.no_grad()
def envelope_draw(eta,matrix,*,generator):
    p=envelope_parameters(eta,matrix)
    component=torch.multinomial(p['log_weights'].exp(),1,generator=generator)[:,0]
    selected=p['parameters'][torch.arange(len(eta)),component]
    u,random=vmf_sample(selected,generator=generator)
    quad=torch.einsum('bi,bij,bj->b',u,matrix,u)
    log_accept=quad-logcosh(p['gap']*(p['axis']*u).sum(1))-p['bound']
    if (log_accept>1e-8).any():raise ValueError('Directional envelope bound violated')
    return u,log_accept,dict(**p,component=component,random=random)
