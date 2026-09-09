"""Haar-rotation-averaged Gaussian proposals on intrinsic molecular coordinates.

The exact proposal averages normalized Gaussians over SO(3); its density uses
the standard matrix-Fisher normalizer (Lee, arXiv:1710.03746, Theorem II.1).
We evaluate that one-dimensional integral by refinement-checked deterministic
quadrature. The numerical answer is not an algebraically exact likelihood or
a certified interval bound. Failure to pass the refinement check raises.
No optimal-alignment density shortcut or neural-flow Jacobian is used.
"""
from functools import lru_cache
import itertools
import math

import numpy as np
import torch

from cfm_mol.tempered_smc import DensityValue


def symmetry_templates(centers,basis,numbers,*,max_permutations=64,reflect=True,seed=0):
    """Fixed finite mixture over identical-atom permutations and optional inversion.

    Enumerate the entire permutation group if small; otherwise retain a fixed
    seeded subset and explicitly report incomplete permutation coverage.
    Either finite mixture is normalized. Only full enumeration guarantees
    invariance to every identical-atom permutation.
    """
    if max_permutations<1:raise ValueError('Positive permutation cap required')
    numbers=list(map(int,numbers));groups=[[i for i,z in enumerate(numbers) if z==element] for element in sorted(set(numbers))]
    total=math.prod(math.factorial(len(group)) for group in groups)
    permutations=[]
    if total<=max_permutations:
        for choices in itertools.product(*(itertools.permutations(group) for group in groups)):
            order=list(range(len(numbers)))
            for group,choice in zip(groups,choices):
                for target,source in zip(group,choice):order[target]=source
            permutations.append(order)
    else:
        generator=torch.Generator().manual_seed(seed);seen={tuple(range(len(numbers)))}
        permutations=[list(range(len(numbers)))]
        for _ in range(100*max_permutations):
            if len(permutations)>=max_permutations:break
            order=list(range(len(numbers)))
            for group in groups:
                choice=torch.randperm(len(group),generator=generator).tolist()
                for target,source in zip(group,choice):order[target]=group[source]
            if tuple(order) not in seen:seen.add(tuple(order));permutations.append(order)
    z=centers.double().reshape(len(centers),-1,3)
    cartesian=torch.einsum('nk,bkd->bnd',basis,z)
    variants=[]
    for order in permutations:
        value=torch.einsum('nk,bnd->bkd',basis,cartesian[:,order,:]).reshape_as(centers)
        variants.append(value)
        if reflect:variants.append(-value)
    return torch.cat(variants),{'total_identical_atom_permutations':total,
        'used_permutations':len(permutations),'all_permutations_enumerated':len(permutations)==total,
        'reflections_included':reflect,'template_count':sum(len(value) for value in variants),
        'permutation_seed':seed}


@lru_cache(maxsize=8)
def _legendre(order):
    return np.polynomial.legendre.leggauss(order)


def random_rotations(count,generator,device=None):
    """Normalized iid Gaussian quaternions induce normalized Haar measure."""
    q=torch.randn((count,4),dtype=torch.float64,device=device,generator=generator)
    q=q/q.norm(dim=-1,keepdim=True)
    w,x,y,z=q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],dim=-1).reshape(count,3,3)


def _integral(s,order):
    points,weights=_legendre(order)
    u=torch.as_tensor(points,dtype=s.dtype,device=s.device)[None,:]
    lw=torch.as_tensor(weights,dtype=s.dtype,device=s.device).log()[None,:]-math.log(2)
    a=.5*(s[:,0:1]-s[:,1:2])*(1-u)
    b=.5*(s[:,0:1]+s[:,1:2])*(1+u)
    ia=torch.special.i0e(a);ib=torch.special.i0e(b)
    log_integrand=ia.log()+a.abs()+ib.log()+b.abs()+s[:,2:3]*u+lw
    logc=torch.logsumexp(log_integrand,-1)
    probability=torch.softmax(log_integrand,-1)
    ra=torch.special.i1e(a)/ia;rb=torch.special.i1e(b)/ib
    g1=(probability*(.5*(1-u)*ra+.5*(1+u)*rb)).sum(-1)
    g2=(probability*(-.5*(1-u)*ra+.5*(1+u)*rb)).sum(-1)
    g3=(probability*u).sum(-1)
    return logc,torch.stack([g1,g2,g3],-1)


@torch.no_grad()
def matrix_fisher_normalizer(matrix,*,tolerance=1e-10,start_order=32,max_order=1024):
    """Return log c(F), E[R] and observed quadrature-refinement diagnostics.

    Analytic first moments avoid differentiating singular vectors at repeated
    singular values. Negatively signed third singular values distinguish SO(3)
    from reflection averaging. Near-rank-one matrices with discarded singular
    values <=1e-12 use the sinh(k)/k limit; their log-normalizer perturbation is
    bounded by the sum of the discarded singular values, <=2e-12.
    """
    if matrix.shape[-2:]!=(3,3) or not torch.isfinite(matrix).all():raise ValueError('Invalid matrix-Fisher parameter')
    if not math.isfinite(tolerance) or tolerance<=0 or start_order<2 or max_order<2*start_order:
        raise ValueError('Invalid quadrature controls')
    shape=matrix.shape[:-2];f=matrix.double().reshape(-1,3,3)
    if not len(f):raise ValueError('Empty matrix-Fisher batch')
    u,s,vh=torch.linalg.svd(f)
    signs=torch.linalg.det(u@vh).sign()
    signed=s.clone();signed[:,2]*=signs
    logc=f.new_empty(len(f));g=f.new_zeros((len(f),3))
    rank_one=s[:,1]<=1e-12
    if rank_one.any():
        k=s[rank_one,0];safe=k.clamp_min(1e-10)
        small=k<1e-3
        logc[rank_one]=torch.where(small,k*k/6-k**4/180+k**6/2835,
            safe+torch.log1p(-torch.exp(-2*safe))-torch.log(2*safe))
        g[rank_one,0]=torch.where(small,k/3-k**3/45+2*k**5/945,1/torch.tanh(safe)-1/safe)
    active=~rank_one;order=0;log_difference=0.;moment_difference=0.
    if active.any():
        order=start_order;previous=_integral(signed[active],order)
        while order*2<=max_order:
            order*=2;current=_integral(signed[active],order)
            log_difference=float((current[0]-previous[0]).abs().max())
            moment_difference=float((current[1]-previous[1]).abs().max())
            if log_difference<=tolerance and moment_difference<=tolerance:
                logc[active]=current[0];g[active]=current[1];break
            previous=current
        else:
            raise RuntimeError(f'Matrix-Fisher quadrature did not converge: order={order}, log={log_difference}, moment={moment_difference}')
    g[:,2]*=signs
    mean=(u*g[:,None,:])@vh
    if not torch.isfinite(logc).all() or not torch.isfinite(mean).all():raise FloatingPointError('Non-finite rotational density')
    return logc.reshape(shape),mean.reshape(*shape,3,3),{
        'order':order,'last_log_refinement_change':log_difference,
        'last_moment_refinement_change':moment_difference,'rank_one_approximations':int(rank_one.sum()),
        'certified_error_bound':False}


class RotatedGaussianMixture:
    """Finite mixture of Gaussian templates, each averaged over all SO(3).

    Coordinates have shape [batch,3*(N-1)] in a fixed orthonormal COM basis.
    Each independent center is [N-1,3]. Rotations act on the Cartesian axis.
    Initial samples use actual random rotations; density is quadrature-evaluated.
    The analytic score follows from the posterior matrix-Fisher mean rotation.
    This proposal is rotationally invariant; it does not sum atom permutations.
    """
    def __init__(self,centers,std,*,tolerance=1e-10,max_order=1024):
        if centers.ndim!=2 or not len(centers) or centers.shape[1]%3 or not torch.isfinite(centers).all():
            raise ValueError('Centers must be finite [components,3*(N-1)] coordinates')
        if not math.isfinite(std) or std<=0:raise ValueError('Positive Gaussian width required')
        self.centers=centers.detach().double().reshape(len(centers),-1,3).clone()
        self.std=std;self.tolerance=tolerance;self.max_order=max_order
        self.max_observed_order=0;self.max_observed_refinement_change=0.

    def sample(self,n,generator):
        if n<1:raise ValueError('Positive sample count required')
        labels=torch.randint(len(self.centers),(n,),device=self.centers.device,generator=generator)
        rotations=random_rotations(n,generator,self.centers.device)
        noise=torch.randn((n,*self.centers.shape[1:]),device=self.centers.device,dtype=torch.float64,generator=generator)
        samples=self.centers[labels]@rotations+self.std*noise
        return samples.reshape(n,-1),labels

    @torch.no_grad()
    def __call__(self,x):
        if x.ndim!=2 or x.shape[1]!=self.centers.shape[1]*3 or not torch.isfinite(x).all():raise ValueError('Invalid intrinsic coordinates')
        z=x.double().reshape(len(x),-1,3);dimension=x.shape[1]
        f=torch.einsum('mki,bkj->bmij',self.centers,z)/self.std**2
        logc,mean,diagnostic=matrix_fisher_normalizer(f,tolerance=self.tolerance,max_order=self.max_order)
        self.max_observed_order=max(self.max_observed_order,diagnostic['order'])
        self.max_observed_refinement_change=max(self.max_observed_refinement_change,diagnostic['last_log_refinement_change'])
        square=z.square().sum((1,2))[:,None]+self.centers.square().sum((1,2))[None,:]
        component=-square/(2*self.std**2)-dimension*math.log(self.std*math.sqrt(2*math.pi))+logc
        log_value=torch.logsumexp(component,-1)-math.log(len(self.centers))
        expected_center=torch.einsum('mki,bmij->bmkj',self.centers,mean)
        score=((torch.softmax(component,-1)[:,:,None,None]*expected_center).sum(1)-z)/self.std**2
        return DensityValue(log_value,score.reshape_as(x))
