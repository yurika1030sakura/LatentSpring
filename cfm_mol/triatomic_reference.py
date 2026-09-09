"""Rotation-reduced Gaussian integration for a three-atom COM-free target.

For rotation-invariant observables, a centered isotropic Gaussian in six
intrinsic coordinates can be integrated using three shape variables. This
is a Gaussian conditioning identity, not a learned sampler or new theorem.
Randomized Sobol repetitions supply a statistical reference, not exact truth.
"""
import math

import numpy as np
from scipy.special import gammaincinv,ndtri,logsumexp
from scipy.stats import qmc


def gaussian_triatomic_shapes(power,std,seed):
    """Canonical representatives of iid-Gaussian shape law via scrambled Sobol.

    r1=y1-y0 ~ N(0,2*sigma^2 I), r2|r1 ~ N(r1/2,3*sigma^2 I/2).
    Integrating their common orientation leaves a Maxwell radius, a normal
    parallel component and a Rayleigh transverse radius. The orthonormal
    COM-free squared norm is sum_i ||x_i||^2; no missing volume factor is added.
    """
    if power<1 or not math.isfinite(std) or std<=0:raise ValueError('Invalid reference size or scale')
    u=qmc.Sobol(3,scramble=True,seed=seed).random_base2(power)
    u=np.clip(u,np.finfo(np.float64).eps,1-np.finfo(np.float64).eps)
    r1=2*std*np.sqrt(gammaincinv(1.5,u[:,0]))
    parallel=.5*r1+std*np.sqrt(1.5)*ndtri(u[:,1])
    transverse=std*np.sqrt(-3*np.log1p(-u[:,2]))
    x=np.zeros((len(u),3,3),dtype=np.float64)
    x[:,1,0]=r1;x[:,2,0]=parallel;x[:,2,1]=transverse
    return x-x.mean(1,keepdims=True)


def centered_gaussian_mixture_log_density(positions,scales):
    """Exact six-dimensional Gaussian mixture density, evaluated on representatives."""
    x=np.asarray(positions,dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(3,3):raise ValueError('Expected [batch,3,3] positions')
    norm2=(x*x).sum((1,2))
    components=np.stack([-norm2/(2*s*s)-6*math.log(s*math.sqrt(2*math.pi)) for s in scales],-1)
    return logsumexp(components,-1)-math.log(len(scales))


def invariant_observables(x):
    a=x[:,1]-x[:,0];b=x[:,2]-x[:,0]
    ra=np.linalg.norm(a,axis=-1);rb=np.linalg.norm(b,axis=-1)
    return {'minimum_anchor_distance_A':np.minimum(ra,rb),
            'maximum_anchor_distance_A':np.maximum(ra,rb),
            'outer_pair_distance_A':np.linalg.norm(x[:,2]-x[:,1],axis=-1),
            'angle_cosine':(a*b).sum(-1)/(ra*rb),
            'radius_gyration_A':np.sqrt((x*x).sum((1,2))/3)}
