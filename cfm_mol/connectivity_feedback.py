"""Geometry and global auxiliary-connectivity feedback for the shared EGNN.

The second invariant edge channel normally repeats the input pair distance.
Replace that channel by a provisional-geometry distance or a regularized tree
connectivity cost. No parameter, supplied bond graph, or atom category is added.
"""
from contextlib import contextmanager
import numpy as np
import torch

from cfm_mol import matched_egnn as base


def _edge_context(module, arguments, keywords):
    context=getattr(module,'_geometry_context',None)
    if context is not None:
        keywords=dict(keywords,edge_attr=context)
    return arguments,keywords


def install(model):
    for index in range(model.dynamics.egnn.n_layers):
        block=model.dynamics.egnn._modules[f'e_block_{index}']
        if not hasattr(block,'_geometry_context'):
            block._geometry_context=None
            block.register_forward_pre_hook(_edge_context,with_kwargs=True)
    return model


@contextmanager
def use_context(model,values):
    blocks=[model.dynamics.egnn._modules[f'e_block_{i}'] for i in range(model.dynamics.egnn.n_layers)]
    old=[block._geometry_context for block in blocks]
    try:
        for block in blocks:block._geometry_context=values
        yield
    finally:
        for block,value in zip(blocks,old):block._geometry_context=value


def tree_marginals(weights):
    """Exact inclusion probabilities of the specified positive weighted-tree law."""
    n=weights.shape[-1]
    if n<2 or weights.shape[-2]!=n:raise ValueError('Require at least two nodes')
    identity=torch.eye(n,dtype=weights.dtype,device=weights.device)
    w=(weights+weights.transpose(-1,-2))/2*(1-identity)
    laplacian=torch.diag_embed(w.sum(-1))-w
    inverse=torch.cholesky_inverse(torch.linalg.cholesky(laplacian+torch.ones_like(w)/n))
    diagonal=inverse.diagonal(dim1=-2,dim2=-1)
    resistance=diagonal[...,None]+diagonal[...,None,:]-2*inverse
    probabilities=w*resistance
    return probabilities*(1-identity)


class GeometryContext:
    def __init__(self,source,kind,regularization=1e-3):
        if kind not in ['distance','tree']:raise ValueError(kind)
        self.source=source;self.kind=kind;self.regularization=regularization;self.cache={}

    @torch.no_grad()
    def __call__(self,coordinates,numbers):
        x=coordinates.detach()
        if self.kind=='distance':
            return (x[:,:,None]-x[:,None,:]).square().sum(-1).reshape(-1,1)
        x=x.double();distance=(x[:,:,None]-x[:,None,:]).square().sum(-1)
        batch,n=coordinates.shape[:2];loga=[];variances=[]
        for zz in numbers.detach().cpu().tolist():
            key=tuple(zz)
            if key not in self.cache:
                _,length=self.source.base.parameters_for(key,0,1)
                logu=self.source.base.base_log_propensity[list(key)].detach().double()
                self.cache[key]=(logu[:,None]+logu[None,:],length.detach().double().square()*np.exp(self.source.width**2)/3)
            a,v=self.cache[key];loga.append(a);variances.append(v)
        loga=torch.stack(loga).to(x);variance=torch.stack(variances).to(x)
        off=1-torch.eye(n,dtype=x.dtype,device=x.device)
        logw=loga-1.5*variance.log()-distance/(2*variance)
        logw=logw.masked_fill(~off.bool(),-torch.inf)
        maximum=logw.amax((-1,-2),keepdim=True)
        chemical=(loga-loga.amax((-1,-2),keepdim=True)).exp()*off
        weights=(logw-maximum).exp()+self.regularization*chemical/n
        marginal=tree_marginals(weights)
        if not torch.isfinite(marginal).all():raise FloatingPointError('Invalid tree marginals')
        if (marginal< -1e-7).any() or (marginal>1+1e-7).any():raise FloatingPointError('Tree probability outside its range')
        # A bounded-scale invariant cost: likely auxiliary edges have low cost.
        cost=-marginal.clamp(min=1e-8,max=1.).log()*off
        return cost.reshape(-1,1).to(coordinates)


def prediction(model,x,t,numbers,spec,context,*,two_pass=True,return_first=False):
    with use_context(model,context(x,numbers)):
        first=base.vector(model,x,t,numbers,spec)
    if not two_pass:return first
    if spec['kind']=='gaga':
        gamma=model.gamma(t);estimate=(x-model.sigma(gamma,x)*first)/model.alpha(gamma,x)
    else:estimate=x+(1-t[...,None])*first
    with use_context(model,context(base.center(estimate),numbers)):
        second=base.vector(model,x,t,numbers,spec)
    return (first,second) if return_first else second


def loss(model,clean,numbers,spec,source,context,seed):
    clean=base.center(clean)/model.norm_values[0];rng=torch.Generator(device=clean.device).manual_seed(seed)
    batch=len(clean)
    if spec['kind']=='gaga':
        t=torch.randint(0,spec['gaga_max_t']+1,(batch,1),generator=rng,device=clean.device).to(clean)/model.T
        target=base.center(torch.randn(clean.shape,generator=rng,device=clean.device));gamma=model.gamma(t)
        xt=model.alpha(gamma,clean)*clean+model.sigma(gamma,clean)*target
    else:
        x0,x1=base.fm_endpoints(clean,numbers,spec['kind'],source,seed)
        t=torch.rand((batch,1),generator=rng,device=clean.device);xt=(1-t[...,None])*x0+t[...,None]*x1;target=x1-x0
    values=prediction(model,xt,t,numbers,spec,context,two_pass=spec['two_pass'],return_first=True)
    if not spec['two_pass']:values=(values,)
    return sum((v-target).square().mean() for v in values)/len(values)


@torch.no_grad()
def sample(model,numbers,spec,source,context,seed,batch,calls=128,*,field_transform=None):
    if spec.get('cached_feedback'):
        if field_transform is not None:raise ValueError('Physical field transforms require the original sampler')
        from .cached_geometry_feedback import sample_cached
        return sample_cached(model,numbers,spec,source,context,seed,batch,calls)
    device=next(model.parameters()).device;z=torch.tensor(numbers,device=device)[None].expand(batch,-1)
    rng=torch.Generator(device=device).manual_seed(seed);kind=spec['kind'];passes=2 if spec['two_pass'] else 1
    if kind=='harmonic_fm':
        nrng=np.random.default_rng(seed);x=torch.stack([source.sample(numbers,nrng) for _ in range(batch)]).to(device).float()
    else:x=base.center(torch.randn((batch,len(numbers),3),generator=rng,device=device))
    def field(x,t):
        value=prediction(model,x,t,z,spec,context,two_pass=spec['two_pass'])
        return value if field_transform is None else field_transform(x,t,z,value)
    if kind in ['harmonic_fm','gaussian_fm']:
        assert calls%(2*passes)==0;initial=x.clone();steps=calls//(2*passes)
        for i in range(steps):
            t=x.new_full((batch,1),i/steps);first=field(x,t)
            middle=base.center(x+first/(2*steps));x=base.center(x+field(middle,t+.5/steps)/steps)
        return x*model.norm_values[0],initial
    maximum=spec['gaga_max_t'];t=x.new_full((batch,1),maximum/model.T);gamma=model.gamma(t)
    x*=((model.alpha(gamma,x)**2)*spec['data_variance_per_dof']+model.sigma(gamma,x)**2).sqrt();initial=x.clone()
    assert calls%passes==0;steps=calls//passes;ticks=np.rint(np.linspace(0,maximum,steps)).astype(int)
    assert len(np.unique(ticks))==steps
    for low,high in reversed(list(zip(ticks[:-1],ticks[1:]))):
        s,t=[x.new_full((batch,1),int(k)/model.T) for k in [low,high]];gs,gt=model.gamma(s),model.gamma(t)
        variance,std,ratio=model.sigma_and_alpha_t_given_s(gt,gs,x);ss,st=model.sigma(gs,x),model.sigma(gt,x)
        mean=x/ratio-variance/(ratio*st)*field(x,t)
        x=base.center(mean+std*ss/st*base.center(torch.randn(x.shape,generator=rng,device=device)))
    t=x.new_zeros(batch,1);gamma=model.gamma(t);alpha,sigma=model.alpha(gamma,x),model.sigma(gamma,x)
    x=base.center((x-sigma*field(x,t))/alpha+sigma/alpha*base.center(torch.randn(x.shape,generator=rng,device=device)))
    return x*model.norm_values[0],initial
