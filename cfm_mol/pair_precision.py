"""Bond-free equivariant Gaussian precision from bounded pairwise elastic terms.

Elastic-network matrices and anisotropic molecular noise are prior work. This
module is a candidate kernel component, not a demonstrated new learning method.
All precision matrices act in an orthonormal zero-centroid coordinate basis.
"""
import math

import torch
from torch import nn

from cfm_mol.nonequilibrium import centered_orthonormal_basis


def pair_precision_matrix(positions,weights,*,strength=32.,softening=.1):
    """I + strength/N * T^T L T, with eigenvalues in [1,1+strength].

    L is a block graph Laplacian of w_ij u_ij u_ij^T, with symmetric weights
    between zero and one and u_ij=(x_i-x_j)/sqrt(||x_i-x_j||^2+softening^2).
    Its geometric contribution vanishes on infinitesimal rigid rotations.
    """
    x=positions.double()
    if x.ndim!=3 or x.shape[-1]!=3 or not 2<=x.shape[1]<=200 or not torch.isfinite(x).all():raise ValueError('Invalid molecular positions')
    batch,n,_=x.shape
    if weights.shape!=(batch,n,n) or not torch.isfinite(weights).all() or (weights<0).any() or (weights>1).any():raise ValueError('Pair weights must lie in [0,1]')
    if not torch.allclose(weights,weights.transpose(1,2),rtol=1e-9,atol=1e-10):raise ValueError('Symmetric pair weights required')
    if not math.isfinite(strength) or strength<0 or not math.isfinite(softening) or softening<=0:raise ValueError('Invalid precision scales')
    delta=x[:,:,None,:]-x[:,None,:,:]
    unit=delta/torch.sqrt(delta.square().sum(-1,keepdim=True)+softening**2)
    blocks=weights.double()[...,None,None]*unit[...,None]*unit[...,None,:]
    diagonal=blocks.sum(2);identity=torch.eye(n,device=x.device,dtype=x.dtype)
    laplacian=torch.einsum('ij,bikl->bikjl',identity,diagonal)-blocks.permute(0,1,3,2,4)
    basis=centered_orthonormal_basis(n,device=x.device)
    tangent=torch.kron(basis.contiguous(),torch.eye(3,device=x.device,dtype=x.dtype))
    relative=tangent.T@laplacian.reshape(batch,3*n,3*n)@tangent
    return torch.eye(3*(n-1),device=x.device,dtype=x.dtype)+strength/n*relative


class LearnedPairPrecision(nn.Module):
    """Invariant pair weights; exact density/sampling is handled separately.

    mode='fixed' freezes w=.5 times the distance envelope. mode='trace' returns
    an isotropic covariance with the same trace as the full covariance at the
    same input, to control for overall noise reduction. No bond labels are used.
    """
    def __init__(self,*,strength=32.,softening=.1,length_scale=2.,mode='learned',embedding_dim=8):
        super().__init__()
        if mode not in ['learned','fixed','trace']:raise ValueError('Unknown precision mode')
        if not math.isfinite(length_scale) or length_scale<=0:raise ValueError('Positive length scale required')
        self.strength=strength;self.softening=softening;self.length_scale=length_scale;self.mode=mode
        self.embedding=nn.Embedding(101,embedding_dim)
        self.network=nn.Sequential(nn.Linear(2*embedding_dim+7,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        if mode=='fixed':
            for parameter in self.parameters():parameter.requires_grad_(False)

    def forward(self,positions,t,numbers,charge,spin,kT):
        x=positions;batch,n,_=x.shape;dtype=self.embedding.weight.dtype
        numbers=torch.as_tensor(numbers,device=x.device,dtype=torch.long)
        if numbers.shape!=(n,) or (numbers<1).any() or (numbers>100).any():raise ValueError('One valid atomic number per atom required')
        if not math.isfinite(kT) or kT<=0 or not math.isfinite(spin) or spin<1 or not math.isfinite(charge):raise ValueError('Invalid electronic conditioning')
        times=torch.as_tensor(t,device=x.device,dtype=dtype).reshape(-1)
        if times.numel()==1:times=times.expand(batch)
        if times.shape!=(batch,) or not torch.isfinite(times).all() or ((times<0)|(times>1)).any():raise ValueError('One valid time per molecule required')
        delta=x[:,:,None,:]-x[:,None,:,:];distance2=delta.square().sum(-1)
        envelope=torch.exp(-distance2/self.length_scale**2)
        if self.mode=='fixed':weights=.5*envelope
        else:
            embedded=self.embedding(numbers)
            pair=torch.cat([embedded[:,None]+embedded[None,:],embedded[:,None]*embedded[None,:]],-1)
            pair=pair[None].expand(batch,-1,-1,-1)
            scalar=torch.stack([torch.sqrt(distance2+self.softening**2)/self.length_scale,
                times[:,None,None].expand(batch,n,n),times[:,None,None].square().expand(batch,n,n),
                x.new_full((batch,n,n),charge/4),x.new_full((batch,n,n),math.log(spin)),
                x.new_full((batch,n,n),math.log(kT)),x.new_full((batch,n,n),math.log(n)/math.log(201))],-1).to(dtype)
            weights=torch.sigmoid(self.network(torch.cat([pair,scalar],-1)).squeeze(-1))*envelope
        precision=pair_precision_matrix(x,weights,strength=self.strength,softening=self.softening)
        if self.mode=='trace':
            inverse=torch.cholesky_inverse(torch.linalg.cholesky(precision))
            average=inverse.diagonal(dim1=-2,dim2=-1).mean(-1)
            precision=torch.eye(precision.shape[-1],device=x.device,dtype=precision.dtype)[None]/average[:,None,None]
        return precision
