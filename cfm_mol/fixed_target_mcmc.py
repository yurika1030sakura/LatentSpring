"""Fixed-budget MALA from arbitrary starts, without a stationarity assertion."""
import math

import torch

from cfm_mol.nonequilibrium import _states
from cfm_mol.tempered_smc import DensityValue,_clip_score,metropolis_log_acceptance


@torch.no_grad()
def mala_population(x0,target,*,steps,proposal_std,generator,max_score_norm=100.,callback=None):
    """Run target-invariant MH kernels; finite-time endpoint density is unknown.

    Initial states need not be at equilibrium. Acceptance and completed steps do
    not prove mixing. Values/forces are cached across rejections. Clipped drifts
    enter both the forward and reverse Gaussian proposal probabilities.
    """
    _states(x0)
    if not isinstance(steps,int) or steps<0:raise ValueError('Nonnegative integer step count required')
    if any(not math.isfinite(value) or value<=0 for value in [proposal_std,max_score_norm]):
        raise ValueError('Positive finite proposal scale and score cap required')
    x=x0.detach().double().clone();value=target(x).validate(x,True)
    n=len(x);evaluations=n;accepted=torch.zeros(n,dtype=torch.long,device=x.device)
    if callback is not None:callback(0,x,value,accepted.clone(),evaluations)
    for step in range(1,steps+1):
        mean=x+.5*proposal_std**2*_clip_score(value.score.double(),max_score_norm)
        y=mean+proposal_std*torch.randn(x.shape,dtype=x.dtype,device=x.device,generator=generator)
        _states(y);proposed=target(y).validate(y,True);evaluations+=n
        reverse=y+.5*proposal_std**2*_clip_score(proposed.score.double(),max_score_norm)
        ratio=metropolis_log_acceptance(x,y,value.log_value,proposed.log_value,mean,reverse,proposal_std)
        if not torch.isfinite(ratio).all():raise FloatingPointError('Non-finite MALA acceptance ratio')
        take=torch.rand(n,dtype=x.dtype,device=x.device,generator=generator).log()<ratio.clamp_max(0)
        accepted+=take.long();x=torch.where(take[:,None],y,x)
        value=DensityValue(torch.where(take,proposed.log_value,value.log_value),
            torch.where(take[:,None],proposed.score,value.score))
        if callback is not None:callback(step,x,value,accepted.clone(),evaluations)
    return x,value,{'steps':steps,'target_evaluations':evaluations,'accepted_per_chain':accepted.cpu().tolist(),
        'acceptance_fraction':float(accepted.sum())/(n*steps) if steps else None,
        'endpoint_density':'unknown; target-invariant transitions do not prove finite-time equilibrium'}


@torch.no_grad()
def hmc_population(x0,target,*,leapfrog_counts,step_size,generator,max_score_norm=100.,callback=None,initial_value=None):
    """Momentum-refreshed HMC with true Hamiltonian acceptance and cached forces.

    Clipped deterministic kicks remain volume preserving and reversible. The
    true target energy enters acceptance; clipping is not a target change.
    """
    _states(x0)
    counts=list(leapfrog_counts)
    if not counts or any(not isinstance(v,int) or v<1 for v in counts):raise ValueError('Positive integer trajectory lengths required')
    if any(not math.isfinite(v) or v<=0 for v in [step_size,max_score_norm]):raise ValueError('Positive finite HMC scales required')
    x=x0.detach().double().clone();n=len(x)
    value=(target(x) if initial_value is None else initial_value).validate(x,True)
    evaluations=n if initial_value is None else 0
    accepted=torch.zeros(n,dtype=torch.long,device=x.device)
    if callback is not None:callback(0,x,value,accepted.clone(),evaluations)
    for iteration,length in enumerate(counts,1):
        initial_p=torch.randn(x.shape,dtype=x.dtype,device=x.device,generator=generator)
        y=x.clone();proposed=value;momentum=initial_p+.5*step_size*_clip_score(value.score.double(),max_score_norm)
        for step in range(length):
            y=y+step_size*momentum;_states(y);proposed=target(y).validate(y,True);evaluations+=n
            factor=.5 if step==length-1 else 1.
            momentum=momentum+factor*step_size*_clip_score(proposed.score.double(),max_score_norm)
        momentum=-momentum
        ratio=proposed.log_value-value.log_value-.5*(momentum.square().sum(-1)-initial_p.square().sum(-1))
        if not torch.isfinite(ratio).all():raise FloatingPointError('Non-finite Hamiltonian acceptance ratio')
        take=torch.rand(n,dtype=x.dtype,device=x.device,generator=generator).log()<ratio.clamp_max(0)
        accepted+=take.long();x=torch.where(take[:,None],y,x)
        value=DensityValue(torch.where(take,proposed.log_value,value.log_value),
            torch.where(take[:,None],proposed.score,value.score))
        if callback is not None:callback(iteration,x,value,accepted.clone(),evaluations)
    return x,value,{'trajectories':len(counts),'leapfrog_steps':sum(counts),'target_evaluations':evaluations,
        'accepted_per_chain':accepted.cpu().tolist(),'acceptance_fraction':float(accepted.sum())/(n*len(counts)),
        'endpoint_density':'unknown; finite trajectories do not certify equilibrium'}
