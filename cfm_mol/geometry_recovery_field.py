"""Bounded geometric endpoint recovery followed by the frozen physical field."""
import torch
from torch import nn
from .geometry_moment_context import GeometryMomentContext
from .matched_egnn import center
from .matched_physical_connection import endpoint_and_progress,native_correction


class GeometryRecoveryField(GeometryMomentContext):
    def __init__(self,atomic_numbers,embedding_dim=16,hidden_dim=64,mode='moments',
                 velocity_scale=4.,gate_power=2):
        super().__init__(atomic_numbers,embedding_dim,hidden_dim,amplitude=1.,mode=mode)
        if velocity_scale<=0 or gate_power<1:raise ValueError('Invalid field bound')
        self.configuration=dict(atomic_numbers=list(atomic_numbers),embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,mode=mode,velocity_scale=velocity_scale,gate_power=gate_power)
        self.network=nn.Sequential(nn.Linear(2*embedding_dim+12,hidden_dim),nn.SiLU(),
            nn.Linear(hidden_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,3))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        self.velocity_scale=float(velocity_scale);self.gate_power=gate_power;self.forward_calls=0

    def forward(self,endpoint,numbers,progress):
        self.forward_calls+=1
        if progress.shape!=(len(endpoint),) or ((progress<0)|(progress>1)).any():
            raise ValueError('Expected one normalized progress per molecule')
        d2,lengths,unit,contacts,degree,first,second,_,features=self.pair_features(endpoint,numbers)
        time=torch.stack([progress,progress.square()],-1)[:,None,None].expand(*features.shape[:-1],2)
        coefficients=torch.tanh(self.network(torch.cat([features,time],-1)))
        if self.mode=='moments':
            tensor_direction=torch.einsum('bijde,bije->bijd',second[:,:,None]+second[:,None,:],unit)/2
            mean_direction=(first[:,:,None]-first[:,None,:])/2
        else:
            distance=(d2+1e-12).sqrt()/lengths
            tensor_direction=torch.exp(-.5*((distance-1.)/.5).square())[...,None]*unit
            mean_direction=torch.tanh(distance)[...,None]*unit
        maximum=degree.amax(-1).clamp_min(1.)
        messages=contacts[...,None]*(coefficients[...,:1]*unit+
            coefficients[...,1:2]*tensor_direction+coefficients[...,2:]*mean_direction)/(3*maximum[:,None,None,None])
        result=center(messages.sum(2))*self.velocity_scale*progress[:,None,None].pow(self.gate_power)
        if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite geometry recovery field')
        return result


def denoising_targets(clean,progress,generator,velocity_scale=4.,gate_power=2):
    """Paired reference recovery with a bounded velocity target, in Angstrom."""
    if ((progress<=0)|(progress>=1)).any():raise ValueError('Interior progress required')
    sigma=.06+.30*(1-progress)
    noise=center(torch.randn(clean.shape,device=clean.device,dtype=clean.dtype,generator=generator))
    identity=torch.rand((len(clean),),device=clean.device,generator=generator)<.15
    noise[identity]=0
    noisy=center(clean+sigma[:,None,None]*noise)
    target=(center(clean)-noisy)/(1-progress[:,None,None])
    bound=velocity_scale*progress.pow(gate_power)
    scale=(bound/target.norm(dim=-1).amax(-1).clamp_min(1e-12)).clamp_max(1.)
    return noisy,target*scale[:,None,None],dict(identity=identity,sigma=sigma,scale=scale)


class GeometryThenPhysical:
    def __init__(self,physical,geometry,strength=1.):
        if not 0<=strength<=1:raise ValueError('Frozen pilot strength must be in[0,1]')
        self.physical=physical;self.geometry=geometry;self.strength=strength;self.calls=0

    def __call__(self,x,t,numbers,value):
        self.calls+=1
        endpoint,progress=endpoint_and_progress(self.physical.model,x,t,value,self.physical.spec)
        geometry_velocity=self.geometry(endpoint,numbers,progress)*self.strength
        modified=value+native_correction(self.physical.model,x,t,geometry_velocity,self.physical.spec)
        return self.physical(x,t,numbers,modified)
