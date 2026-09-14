"""Geometry-preserving product-space coordinates for a fixed auxiliary tree.

Every sampled tree edge remains in a declared covalent-radius interval. This
ensures contact connectivity, not non-edge sterics, chemical bonds or equilibrium.
Riemannian flow matching and internal coordinates have established prior art.
"""
import math
import numpy as np
import torch
from rdkit import Chem
from cfm_mol.degree_tree import tree_degrees,ORGANIC_CAPS


def edge_incidence(edges,n,*,like):
    tree_degrees(edges,n)
    b=like.new_zeros((n-1,n))
    for e,(i,j) in enumerate(edges):b[e,i]=-1.;b[e,j]=1.
    return b


def edge_lengths(numbers,*,like):
    table=Chem.GetPeriodicTable();return like.new_tensor([table.GetRcovalent(int(z)) for z in numbers])


def tree_geometry(edges,numbers,*,like):
    b=edge_incidence(edges,len(numbers),like=like)
    # Unique COM-zero reconstruction of oriented edge vectors.
    inverse=b.T@torch.linalg.inv(b@b.T)
    radii=edge_lengths(numbers,like=like)
    lengths=torch.stack([radii[i]+radii[j] for i,j in edges])
    return b,inverse,lengths


def to_coordinates(y,u,inverse,lengths,lower=.65,upper=1.20):
    if not 0<lower<upper:raise ValueError('Invalid radial interval')
    r=lengths*(lower+(upper-lower)*y.sigmoid())
    return torch.einsum('ne,...ed->...nd',inverse,r[...,None]*u)


def from_coordinates(x,b,lengths,lower=.65,upper=1.20):
    e=torch.einsum('en,...nd->...ed',b,x);r=e.norm(dim=-1)
    fraction=(r/lengths-lower)/(upper-lower)
    if (fraction<=0).any() or (fraction>=1).any():raise ValueError('Reference edges outside the open radial support')
    return torch.logit(fraction),e/r[...,None]


def product_path(y0,u0,y1,u1,t):
    """Linear log-radius-logit path and short great-circle direction path."""
    dot=(u0*u1).sum(-1).clamp(-1.,1.)
    perpendicular=u1-dot[...,None]*u0;norm=perpendicular.norm(dim=-1)
    if ((norm<1e-12)&(dot<0)).any():raise ValueError('Numerically antipodal directions; path not uniquely defined')
    angle=torch.atan2(norm,dot);axis=perpendicular/norm.clamp_min(1e-15)[...,None]
    phase=t*angle
    u=phase.cos()[...,None]*u0+phase.sin()[...,None]*axis
    velocity=angle[...,None]*(-phase.sin()[...,None]*u0+phase.cos()[...,None]*axis)
    return (1-t)*y0+t*y1,u,y1-y0,velocity


def sphere_step(u,tangent,dt):
    tangent=tangent-(u*tangent).sum(-1,keepdim=True)*u
    speed=tangent.norm(dim=-1,keepdim=True);angle=dt*speed
    update=angle.cos()*u+dt*torch.sinc(angle/math.pi)*tangent
    return update/update.norm(dim=-1,keepdim=True).clamp_min(1e-15)


def pullback_velocity(v,b,lengths,u):
    """Neural node-vector channels parameterize logit and spherical velocities."""
    edge=torch.einsum('en,...nd->...ed',b,v.to(b))/lengths[:,None]
    radial=(edge*u).sum(-1)
    return radial,edge-radial[...,None]*u


def geometric_tree(x,numbers,lower=.65,upper=1.20):
    """Deterministic minimum normalized-distance tree, with declared degree caps.

    Reject unsupported references; this is a geometric coordinate scaffold and
    does not read, infer or train on chemical bond-order labels.
    """
    n=len(numbers);radii=edge_lengths(numbers,like=x)
    ratio=torch.cdist(x,x)/(radii[:,None]+radii[None,:])
    candidates=[(float(ratio[i,j]),i,j) for i in range(n) for j in range(i+1,n) if lower<float(ratio[i,j])<upper]
    parent=list(range(n));edges=[]
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    for _,i,j in sorted(candidates):
        ri,rj=root(i),root(j)
        if ri!=rj:parent[ri]=rj;edges.append((i,j))
        if len(edges)==n-1:break
    if len(edges)!=n-1:return None
    if any(d>ORGANIC_CAPS[int(z)] for d,z in zip(tree_degrees(edges,n),numbers)):return None
    return sorted(edges)


@torch.no_grad()
def sample_product_source(edges,numbers,*,generator,width=.2,lower=.65,upper=1.20):
    like=torch.empty((),dtype=torch.double)
    b,inverse,lengths=tree_geometry(edges,numbers,like=like)
    mu=-.5*width**2
    normal=torch.distributions.Normal(like.new_zeros(()),like.new_ones(()))
    low=normal.cdf(like.new_tensor((math.log(lower)-mu)/width))
    high=normal.cdf(like.new_tensor((math.log(upper)-mu)/width))
    quantile=low+(high-low)*torch.rand(len(edges),dtype=torch.double,generator=generator)
    ratio=torch.exp(mu+width*normal.icdf(quantile))
    fraction=((ratio-lower)/(upper-lower)).clamp(1e-12,1-1e-12)
    y=torch.logit(fraction)
    u=torch.randn(len(edges),3,dtype=torch.double,generator=generator);u=u/u.norm(dim=-1,keepdim=True)
    return y,u,b,inverse,lengths


def transport_tangent(tangent,source,target):
    denominator=1+(source*target).sum(-1,keepdim=True)
    if (denominator<1e-8).any():raise ValueError('Antipodal parallel transport is not defined')
    return tangent-(tangent*target).sum(-1,keepdim=True)/denominator*(source+target)


def cap_angular(tangent,maximum=math.pi):
    return tangent*(maximum/tangent.norm(dim=-1,keepdim=True).clamp_min(1e-15)).clamp_max(1.)


def midpoint_product(y,u,t,dt,field):
    """Product-space midpoint with spherical parallel transport; two field calls."""
    vy,vu=field(y,u,t);vu=cap_angular(vu)
    ym=y+.5*dt*vy;um=sphere_step(u,vu,.5*dt)
    my,mu=field(ym,um,t+.5*dt);mu=cap_angular(mu)
    transported=transport_tangent(mu,um,u)
    return y+dt*my,sphere_step(u,transported,dt)
