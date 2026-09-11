"""MALA with explicit zero-density support; invalid proposals remain in counts."""
import math
import torch
from cfm_mol.nonequilibrium import _states
from cfm_mol.tempered_smc import DensityValue,_clip_score,metropolis_log_acceptance


@torch.no_grad()
def supported_mala_population(x0,target,*,steps,proposal_std,generator,max_score_norm=100.,initial_value=None):
    _states(x0)
    if not isinstance(steps,int) or steps<1 or any(not math.isfinite(v) or v<=0 for v in [proposal_std,max_score_norm]):
        raise ValueError('Positive update count and proposal parameters required')
    x=x0.double().clone();n=len(x)
    value=(target(x) if initial_value is None else initial_value).validate(x,True)
    calls=n if initial_value is None else 0
    accepted=torch.zeros(n,dtype=torch.long,device=x.device)
    valid_counts=torch.zeros_like(accepted)
    for _ in range(steps):
        mean=x+.5*proposal_std**2*_clip_score(value.score,max_score_norm)
        y=mean+proposal_std*torch.randn(x.shape,dtype=x.dtype,device=x.device,generator=generator)
        proposed=target(y);calls+=n
        if (proposed.log_value.shape!=(n,) or proposed.score is None or proposed.score.shape!=x.shape
                or torch.isnan(proposed.log_value).any() or torch.isposinf(proposed.log_value).any()
                or not torch.isfinite(proposed.score).all()):
            raise ValueError('Invalid supported-target response')
        valid=torch.isfinite(proposed.log_value);valid_counts+=valid.long()
        # The arbitrary finite score outside support is never used by an accepted
        # state. Its proposal ratio is multiplied by a zero target density.
        reverse=y+.5*proposal_std**2*_clip_score(proposed.score,max_score_norm)
        ratio=metropolis_log_acceptance(x,y,value.log_value,proposed.log_value,mean,reverse,proposal_std)
        if torch.isnan(ratio).any() or torch.isposinf(ratio).any():raise ValueError('Invalid MH ratio')
        take=valid&(torch.rand(n,dtype=x.dtype,device=x.device,generator=generator).log()<ratio.clamp_max(0))
        accepted+=take.long();x=torch.where(take[:,None],y,x)
        value=DensityValue(torch.where(take,proposed.log_value,value.log_value),
            torch.where(take[:,None],proposed.score,value.score))
    return x,value,dict(steps=steps,target_evaluations=calls,accepted_per_chain=accepted.cpu().tolist(),
        valid_proposals_per_chain=valid_counts.cpu().tolist(),
        endpoint_density='unknown finite-time conditional-target law')
