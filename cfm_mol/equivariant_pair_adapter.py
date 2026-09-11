"""Global equivariant pair-potential transport with an exact COM determinant.

A bounded prototype for the homogeneous-context limitation. Convex-potential
and contractive-residual flows are prior art; no universality or novelty claim.
Unlike the element-group primitive, this uses dense3N Cholesky factorizations.
"""
import math

import torch
from torch import nn


class EquivariantPairAdapter(nn.Module):
    """Composition-conditioned pair forces, with position-independent coefficients.

    Each layer is grad Phi, where
      Phi = lambda/2 ||x||^2 + sum_{i<j,k} c_ijk sqrt(ell_k^2+||x_i-x_j||^2).
    Signed coefficients obey sum_k |c_ijk|/ell_k <= beta*lambda/N, beta=1/4.
    Thus ||D(grad Phi-lambda*x)|| <= beta*lambda and the map is invertible.
    Coefficients must not acquire coordinate dependence without a new Jacobian.
    """
    curvature_bound=.25
    scale_bound=.25
    permutation_equivariant=True

    def __init__(self,numbers,*,charge,spin_multiplicity,kT,sweeps=4,hidden=32,
                 length_scales=(.3,.6,1.2,2.4,4.8),affine=False,curvature_bound=.25):
        super().__init__()
        numbers=torch.as_tensor(numbers,dtype=torch.long)
        if numbers.ndim!=1 or not 2<=len(numbers)<=200 or ((numbers<1)|(numbers>118)).any():
            raise ValueError('Require2--200 declared atoms')
        if not isinstance(sweeps,int) or sweeps<1 or not isinstance(hidden,int) or hidden<1:
            raise ValueError('Positive layer count and hidden size required')
        if not math.isfinite(kT) or kT<=0 or spin_multiplicity<1:
            raise ValueError('Invalid electronic/temperature condition')
        if not math.isfinite(curvature_bound) or not 0<curvature_bound<1:
            raise ValueError('Curvature bound must be strictly between zero and one')
        self.curvature_bound=float(curvature_bound)
        scales=torch.as_tensor(length_scales,dtype=torch.float64)
        if scales.ndim!=1 or len(scales)<1 or not torch.isfinite(scales).all() or (scales<=0).any():
            raise ValueError('Positive finite radial lengths required')
        self.register_buffer('numbers',numbers)
        self.register_buffer('length_scales',scales)
        self.register_buffer('pairs',torch.triu_indices(len(numbers),len(numbers),offset=1))
        self.register_buffer('electronic',torch.tensor([charge/5.,(spin_multiplicity-1)/5.,math.log(kT)],dtype=torch.float64))
        self.elements=nn.Embedding(119,hidden)
        self.layer_embedding=nn.Embedding(sweeps,hidden)
        self.state=nn.Sequential(nn.Linear(3,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        self.pair_head=nn.Sequential(nn.Linear(3*hidden,hidden),nn.SiLU(),nn.Linear(hidden,len(scales)))
        self.scale_head=nn.Linear(hidden,1)
        for head in [self.pair_head[-1],self.scale_head]:
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        self.configuration={'charge':charge,'spin_multiplicity':spin_multiplicity,'kT':kT,
            'sweeps':sweeps,'hidden':hidden,'length_scales':scales.tolist(),'affine':bool(affine),
            'curvature_bound':self.curvature_bound}
        self.affine=bool(affine)
        self.sweeps=sweeps

    def coefficients(self,layer):
        elements=self.elements(self.numbers);i,j=self.pairs
        global_context=elements.mean(0)+self.state(self.electronic)+self.layer_embedding.weight[layer]
        pair_context=torch.cat([elements[i]+elements[j],(elements[i]-elements[j]).square(),
            global_context.expand(len(i),-1)],dim=-1)
        lam=torch.exp(self.scale_bound*torch.tanh(self.scale_head(global_context)[0]))
        raw=self.pair_head(pair_context)
        # Preserve the original beta=.25 parameterization and the SAME initial
        # parameter derivative when testing a larger spectral range.
        coefficient=self.curvature_bound*lam*self.length_scales*torch.tanh(raw*.25/self.curvature_bound)/(
            len(self.numbers)*len(self.length_scales))
        return lam,coefficient

    def residual(self,x,coefficient,*,with_blocks=False):
        i,j=self.pairs
        displacement=x[:,i]-x[:,j]
        radius=(displacement.square().sum(-1)[...,None]+self.length_scales.square()).sqrt()
        first=(coefficient[None]/radius).sum(-1)
        if self.affine:
            # Exact tangent at zero, with identical coefficients and NN parameters.
            first=(coefficient/self.length_scales).sum(-1)[None].expand(len(x),-1)
        pair_vectors=first[...,None]*displacement
        change=torch.zeros_like(x).index_add(1,i,pair_vectors).index_add(1,j,-pair_vectors)
        if not with_blocks:return change
        third=torch.zeros_like(first) if self.affine else (coefficient[None]/radius.pow(3)).sum(-1)
        block=first[...,None,None]*torch.eye(3,dtype=x.dtype,device=x.device)-third[...,None,None]*(
            displacement[...,None]*displacement[...,None,:])
        return change,block

    def layer_forward(self,x,layer):
        lam,coefficient=self.coefficients(layer)
        change,block=self.residual(x,coefficient,with_blocks=True)
        n=len(self.numbers);i,j=self.pairs
        matrix=x.new_zeros(len(x),n*n,3,3)
        for indices,values in [(i*n+i,block),(j*n+j,block),(i*n+j,-block),(j*n+i,-block)]:
            matrix=matrix.index_add(1,indices,values)
        matrix=matrix.reshape(len(x),n,n,3,3).permute(0,1,3,2,4).reshape(len(x),3*n,3*n)
        matrix=matrix+lam*torch.eye(3*n,dtype=x.dtype,device=x.device)
        factor=torch.linalg.cholesky(matrix)
        # Three translation directions have eigenvalue lambda. Remove them;
        # the remaining determinant is on the unweighted COM-free subspace H.
        volume=2*factor.diagonal(dim1=-2,dim2=-1).log().sum(-1)-3*lam.log()
        return lam*x+change,volume

    def _validate(self,x):
        if x.ndim!=3 or x.shape[1:]!=(len(self.numbers),3) or len(x)<1 or not torch.isfinite(x).all():
            raise ValueError('Invalid Cartesian coordinate batch')
        if float(x.mean(1).abs().max())>1e-8:raise ValueError('Input must be in the COM-free subspace')

    def forward(self,x):
        self._validate(x);volume=x.new_zeros(len(x))
        for layer in range(self.sweeps):
            x,increment=self.layer_forward(x,layer);volume=volume+increment
        return x,volume

    @torch.no_grad()
    def inverse(self,y,*,tolerance=1e-11,max_iterations=256):
        """Contractive reconstruction; not an inverse-likelihood gradient API."""
        self._validate(y)
        if not math.isfinite(tolerance) or tolerance<=0 or max_iterations<1:raise ValueError('Invalid inverse settings')
        x=y;inverse_volume=y.new_zeros(len(y));iterations=[]
        for layer in reversed(range(self.sweeps)):
            target=x;lam,coefficient=self.coefficients(layer);x=target/lam
            for step in range(max_iterations):
                following=(target-self.residual(x,coefficient))/lam
                error=float((following-x).abs().max());x=following
                if error<tolerance:break
            else:raise RuntimeError('Pair transport inverse did not converge')
            reconstructed,volume=self.layer_forward(x,layer)
            if float((reconstructed-target).abs().max())>max(1e-9,4*tolerance):
                raise RuntimeError('Pair transport inverse residual failed')
            inverse_volume-=volume;iterations.append(step+1)
        rebuilt,forward_volume=self(x)
        return x,inverse_volume,{'iterations':iterations,'maximum_residual':float((rebuilt-y).abs().max()),
            'volume_cancellation_error':float((inverse_volume+forward_volume).abs().max())}
