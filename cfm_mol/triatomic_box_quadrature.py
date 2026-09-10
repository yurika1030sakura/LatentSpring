"""Direct labelled-coordinate cubature for a bounded three-atom shape domain."""
import math

import numpy as np
from numpy.polynomial.legendre import leggauss


def triatomic_box_nodes(order,lower,upper):
    if not isinstance(order,int) or order<2 or not 0<=lower<upper or not math.isfinite(upper):
        raise ValueError('Require an order >= 2 and finite nonnegative radial interval')
    nodes,weights=leggauss(order)
    radius=lower+(nodes+1)*(upper-lower)/2;radial_weight=weights*(upper-lower)/2
    r1,r2,cosine=np.meshgrid(radius,radius,nodes,indexing='ij')
    w1,w2,wc=np.meshgrid(radial_weight,radial_weight,weights,indexing='ij')
    r1,r2,cosine=[value.ravel() for value in [r1,r2,cosine]]
    x=np.zeros((len(r1),3,3));x[:,1,0]=r1;x[:,2,0]=r2*cosine;x[:,2,1]=r2*np.sqrt(1-cosine*cosine)
    x-=x.mean(1,keepdims=True)
    # No sorting, symmetry division, density estimation, or mixture factors.
    volume=(8*math.pi**2/(3*math.sqrt(3)))*r1*r1*r2*r2
    log_weight=np.log((w1*w2*wc).ravel()*volume)
    return x,log_weight
