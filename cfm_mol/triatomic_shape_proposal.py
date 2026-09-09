"""Explicit shape proposals for independent three-atom thermodynamic integrals.

Coordinates are log(r01), log(r02), atanh(cos(angle102)). Gaussian densities in
these coordinates include the full Jacobian to the six-dimensional orthonormal
COM-free measure, after integrating uniform SO(3) orientation. These are standard
change-of-variables and mixture identities, not a learned flow or new theorem.
"""
import math

import numpy as np
from scipy.special import logsumexp,ndtri
from scipy.stats import qmc

from cfm_mol.triatomic_reference import centered_gaussian_mixture_log_density


def to_shape_coordinates(positions):
    x=np.asarray(positions,dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(3,3) or not np.isfinite(x).all():raise ValueError('Finite [batch,3,3] coordinates required')
    a=x[:,1]-x[:,0];b=x[:,2]-x[:,0]
    ra=np.linalg.norm(a,axis=-1);rb=np.linalg.norm(b,axis=-1)
    if (ra<=0).any() or (rb<=0).any():raise ValueError('Degenerate anchor distances')
    cosine=(a*b).sum(-1)/(ra*rb)
    if (np.abs(cosine)>=1).any():raise ValueError('Collinear boundary has singular shape coordinates')
    return np.stack([np.log(ra),np.log(rb),np.arctanh(cosine)],-1)


def from_shape_coordinates(values):
    u=np.asarray(values,dtype=np.float64)
    if u.ndim!=2 or u.shape[1]!=3 or not np.isfinite(u).all():raise ValueError('Finite [batch,3] shape coordinates required')
    ra,rb=np.exp(u[:,:2]).T;cosine=np.tanh(u[:,2])
    x=np.zeros((len(u),3,3),dtype=np.float64)
    x[:,1,0]=ra;x[:,2,0]=rb*cosine;x[:,2,1]=rb*np.sqrt(1-cosine**2)
    return x-x.mean(1,keepdims=True)


def log_shape_volume(u):
    """Orientation-integrated dH/du = 8*pi^2*r1^3*r2^3*(1-c^2)/(3*sqrt(3))."""
    u=np.asarray(u,dtype=np.float64)
    log_sech2=2*(math.log(2)-np.logaddexp(u[:,2],-u[:,2]))
    return math.log(8*math.pi**2/(3*math.sqrt(3)))+3*u[:,0]+3*u[:,1]+log_sech2


class TriatomicShapeMixture:
    def __init__(self,centers,stds=(.05,.05,.5)):
        self.centers=np.asarray(centers,dtype=np.float64).copy();self.stds=np.asarray(stds,dtype=np.float64)
        if self.centers.ndim!=2 or self.centers.shape[1]!=3 or len(self.centers)<1 or not np.isfinite(self.centers).all():raise ValueError('Invalid shape centers')
        if self.stds.shape!=(3,) or not np.isfinite(self.stds).all() or (self.stds<=0).any():raise ValueError('Positive shape scales required')

    def sample(self,power,seed):
        uniform=qmc.Sobol(4,scramble=True,seed=seed).random_base2(power)
        component=np.minimum((uniform[:,0]*len(self.centers)).astype(int),len(self.centers)-1)
        noise=ndtri(np.clip(uniform[:,1:],np.finfo(float).eps,1-np.finfo(float).eps))
        return from_shape_coordinates(self.centers[component]+noise*self.stds)

    def log_density(self,positions):
        u=to_shape_coordinates(positions)
        residual=(u[:,None,:]-self.centers[None,:,:])/self.stds
        log_components=-.5*np.sum(residual**2,-1)-np.log(self.stds).sum()-1.5*math.log(2*math.pi)
        return logsumexp(log_components,-1)-math.log(len(self.centers))-log_shape_volume(u)


def defensive_shape_log_density(positions,local,gaussian_std):
    if not math.isfinite(gaussian_std) or gaussian_std<=0:raise ValueError('Positive defensive scale required')
    return np.logaddexp(local.log_density(positions),
        centered_gaussian_mixture_log_density(positions,[gaussian_std]))-math.log(2)
