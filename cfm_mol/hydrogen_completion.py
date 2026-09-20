"""A conditional coordinate flow for hydrogen placement with fixed heavy geometry.

Training uses only OMol25 coordinates and atom identities. Hydrogen decoration
is an established task; this module is an unqualified task-specific candidate.
"""
import torch
from scipy.optimize import linear_sum_assignment
from . import matched_egnn as base
from .chemical_moves import covalent_radii


def detached_hydrogens(x,numbers,contact=1.25):
    if x.ndim!=3 or numbers.shape!=x.shape[:2] or not torch.isfinite(x).all():raise ValueError('Invalid coordinate/atom batch')
    lookup=torch.cat([x.new_zeros(1),covalent_radii(range(1,84),dtype=x.dtype,device=x.device)])
    r=lookup[numbers];distance=(x[:,:,None]-x[:,None,:]).norm(dim=-1);length=r[:,:,None]+r[:,None,:]
    hydrogen=numbers==1;heavy=~hydrogen
    ratio=(distance/length).masked_fill(~heavy[:,None,:],torch.inf)
    return hydrogen&(ratio.amin(-1)>contact)&heavy.any(-1,keepdim=True)


def training_example(clean,numbers,*,seed,noise_std=.4):
    """Keep heavy-relative geometry; pair identical H positions by assignment."""
    clean=base.center(clean);rng=torch.Generator(device=clean.device).manual_seed(seed);h=numbers==1
    if not h.any(-1).all() or not (~h).any(-1).all():raise ValueError('Require both hydrogen and heavy atoms')
    # Half of molecules corrupt every H; half use independent half-probability
    # H corruptions. The mask is latent, not a target graph or inference input.
    whole=torch.rand((len(clean),1),generator=rng,device=clean.device)<.5
    active=h&(whole|(torch.rand(h.shape,generator=rng,device=clean.device)<.5))
    noise=torch.randn(clean.shape,generator=rng,device=clean.device)*noise_std*active[...,None]
    source=clean+noise;target=clean.clone()
    for b in range(len(clean)):
        ids=torch.where(h[b])[0];cost=torch.cdist(source[b,ids].double(),clean[b,ids].double()).square().detach().cpu().numpy()
        left,right=linear_sum_assignment(cost);assert (left==range(len(ids))).all()
        target[b,ids]=clean[b,ids[torch.as_tensor(right,device=ids.device)]]
    t=torch.rand((len(clean),1),generator=rng,device=clean.device)
    x=base.center((1-t[...,None])*source+t[...,None]*target)
    # The conditional velocity before the common translation is zero on heavy
    # atoms. Only hydrogen predictions are supervised and applied at inference.
    velocity=(target-source)*h[...,None]
    return x,t,velocity


def loss(model,clean,numbers,spec,*,seed):
    x,t,target=training_example(clean,numbers,seed=seed,noise_std=spec['noise_std_A'])
    pred=base.vector(model,x,t,numbers,spec['network_spec']);h=(numbers==1).to(pred)
    return (((pred-target).square().sum(-1)*h).sum(-1)/(3*h.sum(-1))).mean()


@torch.no_grad()
def complete(model,x,numbers,spec,*,mode='molecule',start_time=0.,steps=4,velocity_cap=2.):
    """No oracle or optimizer. Preserve unselected structures exactly."""
    if mode not in ['molecule','atom'] or not 0<=start_time<1 or steps<1 or velocity_cap<=0:raise ValueError('Invalid completion settings')
    original=x.clone();detached=detached_hydrogens(x.double(),numbers);changed=detached.any(-1)
    active_ids=torch.where(changed)[0]
    if not len(active_ids):return original,dict(changed=changed,active_atoms=detached,network_calls=0,network_example_calls=0)
    y=x[active_ids].clone();z=numbers[active_ids];active=detached[active_ids] if mode=='atom' else z==1
    dt=(1-start_time)/steps;calls=0
    def field(state,time):
        nonlocal calls
        t=state.new_full((len(state),1),time);v=base.vector(model,state,t,z,spec['network_spec']);calls+=1
        scale=(velocity_cap/v.norm(dim=-1).clamp_min(1e-12)).clamp_max(1.)
        return v*scale[...,None]*active[...,None]
    for step in range(steps):
        t=start_time+step*dt;first=field(y,t);middle=y+dt/2*first;y=y+dt*field(middle,t+dt/2)
    if not torch.isfinite(y).all():raise FloatingPointError('Nonfinite hydrogen completion')
    # Conditional integration fixes all inactive coordinates. A final common
    # translation preserves the input centroid and heavy relative geometry.
    assert torch.equal(y[~active],x[active_ids][~active])
    result=original.clone();result[active_ids]=base.center(y)+original[active_ids].mean(1,keepdim=True)
    torch.testing.assert_close(result[~changed],original[~changed],atol=0,rtol=0)
    full_active=torch.zeros_like(detached);full_active[active_ids]=active
    return result,dict(changed=changed,active_atoms=full_active,network_calls=calls,network_example_calls=calls*len(active_ids))
