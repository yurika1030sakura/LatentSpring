"""Tree-conditioned Gaussian diffusion for transferring the harmonic source.

For a fixed auxiliary tree, B maps standard edge noise to centered atom noise.
C=B B^T is positive definite on the zero-centroid subspace. Forward marginals
are alpha*x_clean + sigma*B*z; every reverse innovation uses the same B.
The denoiser predicts physical-coordinate noise, conditioned on C through an
invariant edge channel. The latent tree is not an observed chemical bond graph.
Non-isotropic Gaussian diffusion is established prior work, not a new identity.
"""
import math
import numpy as np
import torch
from . import matched_egnn as base, connectivity_feedback as feedback
from .tree_prior_controls import cayley_tree, edge_embedding


class HarmonicNoise:
    def __init__(self,width=.2):
        self.source=base.HarmonicSource(width)

    def tree(self,numbers,rng):
        key=tuple(numbers);source=self.source
        if not 2<=len(key)<=200:raise ValueError('Between two and200 atoms required')
        if key not in source.cache:
            _,length=source.base.parameters_for(key,0,1)
            u=source.base.base_log_propensity[list(key)].exp().detach().numpy()
            source.cache[key]=(u,length.detach().numpy()*math.exp(.5*source.width**2)/math.sqrt(3))
        u,scales=source.cache[key];edges=np.asarray(cayley_tree(u,rng),dtype=np.int64)
        std=scales[edges[:,0],edges[:,1]]
        factor=edge_embedding(len(key),edges)*std[None,:]
        return factor,edges,std

    def batch(self,numbers,seed,*,device,dtype):
        rng=np.random.default_rng(seed)
        components=[self.tree(list(z),rng) for z in numbers]
        factor=torch.as_tensor(np.stack([c[0] for c in components]),device=device,dtype=dtype)
        edges=torch.as_tensor(np.stack([c[1] for c in components]),device=device,dtype=torch.long)
        std=torch.as_tensor(np.stack([c[2] for c in components]),device=device,dtype=dtype)
        covariance=factor@factor.transpose(-1,-2)
        diagonal=covariance.diagonal(dim1=-2,dim2=-1)
        pair_variance=(diagonal[:,:,None]+diagonal[:,None,:]-2*covariance).clamp_min(0)
        context=torch.log1p(3*pair_variance).reshape(-1,1)
        return dict(factor=factor,edges=edges,std=std,covariance=covariance,context=context)


def draw_noise(component,rng):
    b=component['factor'];z=torch.randn((b.shape[0],b.shape[-1],3),device=b.device,dtype=b.dtype,generator=rng)
    return base.center(b@z)


def mahalanobis_loss(error,component):
    edges=component['edges'];batch=torch.arange(len(error),device=error.device)[:,None]
    increments=(error[batch,edges[:,:,0]]-error[batch,edges[:,:,1]])/component['std'][:,:,None]
    return increments.square().mean()


def prediction(model,x,t,numbers,spec,component):
    with feedback.use_context(model,component['context']):
        return base.vector(model,x,t,numbers,spec)


def loss(model,clean,numbers,spec,noise,seed):
    if spec['kind'] not in ('edm','gaga'):raise ValueError(spec['kind'])
    if model.norm_values[0]!=1.:raise ValueError('Coordinates must be in Angstrom')
    clean=base.center(clean);rng=torch.Generator(device=clean.device).manual_seed(seed)
    component=noise.batch(numbers.detach().cpu().tolist(),seed+300000007,device=clean.device,dtype=clean.dtype)
    maximum=spec['gaga_max_t'] if spec['kind']=='gaga' else model.T
    t=torch.randint(0,maximum+1,(len(clean),1),device=clean.device,generator=rng).to(clean)/model.T
    eta=draw_noise(component,rng);gamma=model.gamma(t)
    x=model.alpha(gamma,clean)*clean+model.sigma(gamma,clean)*eta
    return mahalanobis_loss(prediction(model,x,t,numbers,spec,component)-eta,component)


def initial_state(model,spec,component,rng):
    eta=draw_noise(component,rng)
    if spec['kind']=='edm':return eta
    if spec['kind']!='gaga':raise ValueError(spec['kind'])
    t=eta.new_full((len(eta),1),spec['gaga_max_t']/model.T);gamma=model.gamma(t)
    # The same Gaussian data approximation as GAGA, plus harmonic corruption:
    # Cov(X_T|tree)=alpha_T^2*v*P + sigma_T^2*C_tree.
    data_noise=base.center(torch.randn(eta.shape,device=eta.device,dtype=eta.dtype,generator=rng))
    return base.center(model.alpha(gamma,eta)*math.sqrt(spec['data_variance_per_dof'])*data_noise+
                       model.sigma(gamma,eta)*eta)


@torch.no_grad()
def sample(model,numbers,spec,noise,seed,batch,calls=128,*,field_transform=None):
    device=next(model.parameters()).device;dtype=next(model.parameters()).dtype
    component=noise.batch([numbers]*batch,seed+300000007,device=device,dtype=dtype)
    rng=torch.Generator(device=device).manual_seed(seed)
    z=torch.tensor(numbers,device=device)[None].expand(batch,-1)
    x=initial_state(model,spec,component,rng);initial=x.clone()
    def field(x,t):
        value=prediction(model,x,t,z,spec,component)
        return value if field_transform is None else field_transform(x,t,z,value)
    maximum=spec['gaga_max_t'] if spec['kind']=='gaga' else model.T
    ticks=np.rint(np.linspace(0,maximum,calls)).astype(int)
    if len(np.unique(ticks))!=calls:raise ValueError('Repeated diffusion time points')
    for low,high in reversed(list(zip(ticks[:-1],ticks[1:]))):
        s,t=[x.new_full((batch,1),int(k)/model.T) for k in [low,high]]
        gs,gt=model.gamma(s),model.gamma(t)
        variance,std,ratio=model.sigma_and_alpha_t_given_s(gt,gs,x)
        ss,st=model.sigma(gs,x),model.sigma(gt,x)
        mean=x/ratio-variance/(ratio*st)*field(x,t)
        x=base.center(mean+std*ss/st*draw_noise(component,rng))
    t=x.new_zeros(batch,1);gamma=model.gamma(t)
    alpha,sigma=model.alpha(gamma,x),model.sigma(gamma,x)
    x=base.center((x-sigma*field(x,t))/alpha+sigma/alpha*draw_noise(component,rng))
    if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite harmonic-diffusion output')
    return x,initial,{k:v.detach().cpu() for k,v in component.items() if k!='context'}
