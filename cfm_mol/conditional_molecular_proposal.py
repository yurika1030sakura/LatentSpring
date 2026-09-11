"""State-conditioned molecular proposal with intrinsic density in both directions.

The current state conditions every atom; active noise is never used as context.
A low-rank collective map precedes per-atom centered convex point maps. The
inverse remains differentiable for reverse-proposal likelihood training. These
are established flow/MH principles; useful molecular sampling is not implied.
"""
import math
import torch
from torch import nn


def center(x):return x-x.mean(1,keepdim=True)


class ConditionalMolecularProposal(nn.Module):
    def __init__(self,*,hidden=16,radial=16,features=4,rank=4,step_size=.1,
                 shift_bound=.5,scale_bound=1.5,nonlinear=True,inverse_iterations=32):
        super().__init__()
        if (min(hidden,radial,features,rank,inverse_iterations)<1 or step_size<=0
                or not all(math.isfinite(v) for v in [step_size,shift_bound,scale_bound])):
            raise ValueError('Positive network dimensions, steps and inverse iterations required')
        self.configuration=dict(hidden=hidden,radial=radial,features=features,rank=rank,
            step_size=step_size,shift_bound=shift_bound,scale_bound=scale_bound,
            nonlinear=nonlinear,inverse_iterations=inverse_iterations)
        self.features=features;self.rank=rank;self.step_size=step_size;self.shift_bound=shift_bound
        self.scale_bound=scale_bound;self.nonlinear=nonlinear;self.inverse_iterations=inverse_iterations
        self.elements=nn.Embedding(119,hidden)
        self.state=nn.Sequential(nn.Linear(3,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        self.register_buffer('centers',torch.linspace(0,8,radial))
        self.messages=nn.ModuleList([nn.Sequential(nn.Linear(2*hidden+radial,hidden),nn.SiLU(),
            nn.Linear(hidden,hidden),nn.SiLU()) for _ in range(2)])
        self.updates=nn.ModuleList([nn.Sequential(nn.Linear(2*hidden,hidden),nn.SiLU(),nn.Linear(hidden,hidden)) for _ in range(2)])
        self.node_head=nn.Linear(hidden,2*features+1)
        self.group_head=nn.Linear(hidden,rank+1)
        self.vector_head=nn.Linear(hidden,features+rank+1)
        for head in [self.node_head,self.group_head]:
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        with torch.no_grad():
            self.vector_head.weight[-1].zero_();self.vector_head.bias[-1].zero_()

    def _context(self,x,numbers,electronic):
        if (x.ndim!=3 or x.shape[-1]!=3 or not 2<=x.shape[1]<=200
                or numbers.shape!=(x.shape[1],) or numbers.dtype!=torch.long
                or ((numbers<1)|(numbers>118)).any()):
            raise ValueError('Require2--200 declared atoms and integer atomic numbers')
        if electronic.shape==(3,):electronic=electronic.expand(len(x),-1)
        if electronic.shape!=(len(x),3) or not torch.isfinite(x).all() or not torch.isfinite(electronic).all():
            raise ValueError('Finite context and electronic features required')
        if float(x.mean(1).abs().max())>1e-8:raise ValueError('Context must have zero unweighted centroid')
        n=x.shape[1];difference=x[:,:,None,:]-x[:,None,:,:]
        squared=difference.square().sum(-1)
        radial=torch.exp(-.5*((torch.sqrt(squared+1e-8)[...,None]-self.centers)/.4)**2)
        nodes=self.elements(numbers)[None]+self.state(electronic)[:,None,:]
        mask=(~torch.eye(n,dtype=torch.bool,device=x.device))[None,:,:,None]
        for message,update in zip(self.messages,self.updates):
            a=nodes[:,:,None].expand(-1,-1,n,-1);b=nodes[:,None,:].expand(-1,n,-1,-1)
            pairs=message(torch.cat([a+b,(a-b).square(),radial],-1))
            nodes=nodes+update(torch.cat([nodes,(pairs*mask).sum(2)/n],-1))
        scalars=torch.tanh(self.vector_head(pairs))*mask
        vectors=torch.einsum('bijk,bijd->bikd',scalars,difference/(squared[...,None]+.01).sqrt())/n
        params=self.node_head(nodes);group=self.group_head(nodes.mean(1))
        directions=vectors[:,:,:self.features]
        directions=directions/(directions.square().sum(-1,keepdim=True)+.01).sqrt()
        collective=vectors[:,:,self.features:self.features+self.rank]
        collective=collective-collective.mean(1,keepdim=True)
        collective=collective/(collective.square().sum((1,3),keepdim=True)+.01).sqrt()
        scale=torch.exp(self.scale_bound*torch.tanh(group[:,-1]))
        return dict(weights=.5*scale[:,None,None]*torch.tanh(params[:,:,:self.features])/self.features,
            offsets=params[:,:,self.features:2*self.features],directions=directions,
            radial=.125*scale[:,None]*torch.tanh(params[:,:,-1]),scale=scale,
            collective=collective,coefficients=.75*torch.tanh(group[:,:self.rank])/self.rank,
            shift=self.shift_bound*center(vectors[:,:,-1]))

    def _point(self,u,context,*,jacobian=True):
        if not self.nonlinear:
            zero=torch.zeros_like(u)
            _,matrix=self._nonlinear_point(zero,context,jacobian=True)
            return torch.einsum('bnij,bnj->bni',matrix,u),matrix if jacobian else None
        return self._nonlinear_point(u,context,jacobian=jacobian)

    @staticmethod
    def _nonlinear_point(u,context,*,jacobian):
        b=context['directions'];a=context['weights'];c=context['offsets']
        sigmoid=torch.sigmoid(torch.einsum('bnd,bnkd->bnk',u,b)+c)
        radius=(1+u.square().sum(-1)).sqrt()
        mapped=(context['scale'][:,None,None]*u+
            torch.einsum('bnk,bnkd->bnd',a*(sigmoid-torch.sigmoid(c)),b)+
            context['radial'][...,None]*u/radius[...,None])
        if not jacobian:return mapped,None
        identity=torch.eye(3,dtype=u.dtype,device=u.device)
        matrix=context['scale'][:,None,None,None]*identity
        matrix=matrix+torch.einsum('bnk,bnki,bnkj->bnij',a*sigmoid*(1-sigmoid),b,b)
        matrix=matrix+context['radial'][...,None,None]*(identity/radius[...,None,None]-
            u[...,:,None]*u[...,None,:]/radius[...,None,None]**3)
        return mapped,matrix

    @staticmethod
    def _volume(matrix):
        chol=torch.linalg.cholesky(matrix)
        correction=torch.linalg.cholesky(torch.cholesky_inverse(chol).mean(1))
        return 2*(chol.diagonal(dim1=-2,dim2=-1).log().sum((1,2))+
                  correction.diagonal(dim1=-2,dim2=-1).log().sum(1))

    def _linear(self,u,context,*,inverse=False):
        a=context['collective'];c=context['coefficients']
        gram=torch.einsum('bnkd,bnld->bkl',a,a)
        small=torch.eye(self.rank,dtype=u.dtype,device=u.device)[None]+c[:,:,None]*gram
        sign,logdet=torch.linalg.slogdet(small)
        if (sign<=0).any():raise RuntimeError('Collective-map determinant is not positive')
        projection=torch.einsum('bnkd,bnd->bk',a,u)
        if inverse:
            coefficient=torch.linalg.solve(small,(c*projection)[...,None])[...,0]
            mapped=u-torch.einsum('bnkd,bk->bnd',a,coefficient)
        else:mapped=u+torch.einsum('bnkd,bk->bnd',a,c*projection)
        return center(mapped),logdet

    def transform(self,x,noise,numbers,electronic):
        """noise is a standard Gaussian on H; return proposal and log q(y|x)."""
        context=self._context(x,numbers,electronic)
        if noise.shape!=x.shape or not torch.isfinite(noise).all() or float(noise.mean(1).abs().max())>1e-8:
            raise ValueError('Noise must belong to the same COM subspace')
        u,linear_volume=self._linear(noise,context)
        mapped,matrix=self._point(u,context)
        y=x+context['shift']+self.step_size*center(mapped)
        d=3*(x.shape[1]-1)
        volume=linear_volume+self._volume(matrix)+d*math.log(self.step_size)
        log_density=-.5*noise.square().sum((1,2))-.5*d*math.log(2*math.pi)-volume
        return y,log_density

    def log_prob(self,y,x,numbers,electronic):
        """Numerically controlled differentiable inverse; includes context gradients."""
        context=self._context(x,numbers,electronic)
        if y.shape!=x.shape or not torch.isfinite(y).all() or float(y.mean(1).abs().max())>1e-8:
            raise ValueError('Proposal value must belong to the same COM subspace')
        value=(y-x-context['shift'])/self.step_size
        if self.nonlinear:
            u=value/context['scale'][:,None,None]
            for _ in range(self.inverse_iterations):
                mapped,_=self._point(u,context,jacobian=False)
                residual=mapped-context['scale'][:,None,None]*u
                u=center((value-center(residual))/context['scale'][:,None,None])
        else:
            _,matrix=self._point(torch.zeros_like(value),context)
            inverse=torch.linalg.inv(matrix)
            unconstrained=torch.einsum('bnij,bnj->bni',inverse,value)
            multiplier=-torch.linalg.solve(inverse.mean(1),unconstrained.mean(1)[...,None])[...,0]
            u=torch.einsum('bnij,bnj->bni',inverse,value+multiplier[:,None,:])
        mapped,matrix=self._point(u,context)
        error=(center(mapped)-value).abs().amax((1,2))
        if (error>1e-9*(1+value.abs().amax((1,2)))).any():
            raise RuntimeError('Conditional inverse fails residual tolerance')
        noise,linear_volume=self._linear(u,context,inverse=True)
        d=3*(x.shape[1]-1)
        volume=linear_volume+self._volume(matrix)+d*math.log(self.step_size)
        return -.5*noise.square().sum((1,2))-.5*d*math.log(2*math.pi)-volume


def invariant_jump_squared(x,y,numbers):
    """Typed sorted pair-distance change; excludes rigid and relabelling motion.

    This is an incomplete invariant observable, not a certificate of mode mixing.
    """
    i,j=torch.triu_indices(len(numbers),len(numbers),1,device=x.device)
    types=torch.minimum(numbers[i],numbers[j])*119+torch.maximum(numbers[i],numbers[j])
    dx=((x[:,i]-x[:,j]).square().sum(-1)+1e-12).sqrt()
    dy=((y[:,i]-y[:,j]).square().sum(-1)+1e-12).sqrt()
    differences=[]
    for kind in types.unique():
        mask=types==kind
        differences.append(dy[:,mask].sort(-1).values-dx[:,mask].sort(-1).values)
    return torch.cat(differences,-1).square().mean(-1)


@torch.no_grad()
def metropolis_transition(model,x,value,target,numbers,electronic,*,generator):
    noise=center(torch.randn(x.shape,dtype=x.dtype,device=x.device,generator=generator))
    proposed,forward=model.transform(x,noise,numbers,electronic)
    new_value=target(proposed)
    if value.shape!=(len(x),) or new_value.shape!=(len(x),):
        raise ValueError('Require one target log value per state')
    if not torch.isfinite(value).all() or torch.isnan(new_value).any() or torch.isposinf(new_value).any():
        raise ValueError('Current states must have finite density; proposed values may be minus infinity')
    valid=torch.isfinite(new_value)
    reverse=torch.zeros_like(new_value)
    if valid.any():
        state=electronic[valid] if electronic.ndim==2 else electronic
        reverse[valid]=model.log_prob(x[valid],proposed[valid],numbers,state)
    ratio=new_value-value+reverse-forward
    if torch.isnan(ratio).any() or torch.isposinf(ratio).any():raise ValueError('Invalid Metropolis ratio')
    take=torch.rand(len(x),dtype=x.dtype,device=x.device,generator=generator).log()<ratio.clamp_max(0)
    return torch.where(take[:,None,None],proposed,x),torch.where(take,new_value,value),dict(
        accepted=take,valid_proposals=valid,log_ratio=ratio,
        accepted_invariant_jump=invariant_jump_squared(x,proposed,numbers)*take)
