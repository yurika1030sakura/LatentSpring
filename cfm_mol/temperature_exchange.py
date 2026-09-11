"""Replica exchange in temperature-scaled coordinates, with cached target values.

For each slot, physical z=sqrt(kT)*w and log_value=-(U(z)-constant)/kT.
Physical coordinates are exchanged; exchanging w directly is a different move.
This is a standard parallel-tempering kernel, not a new sampling method.
"""
import torch
from cfm_mol.tempered_smc import DensityValue


@torch.no_grad()
def exchange_scaled_states(w,value,temperatures,labels,*,parity,generator):
    if w.ndim!=3 or temperatures.ndim!=1 or w.shape[1]!=len(temperatures):
        raise ValueError('Require[ladders,temperatures,dimensions] and one temperature grid')
    ladders,levels,dim=w.shape
    if (levels<2 or not torch.isfinite(temperatures).all() or (temperatures<=0).any()
            or not (temperatures[1:]>temperatures[:-1]).all() or parity not in (0,1)
            or labels.shape!=w.shape[:2]):
        raise ValueError('Invalid temperature grid, parity or walker labels')
    value.validate(w.reshape(-1,dim),True)
    logp=value.log_value.reshape(ladders,levels)
    score=value.score.reshape_as(w)
    left=torch.arange(parity,levels-1,2,device=w.device);right=left+1
    ti,tj=temperatures[left],temperatures[right]
    ui=-ti*logp[:,left];uj=-tj*logp[:,right]
    ratio=(ti.reciprocal()-tj.reciprocal())*(ui-uj)
    if not torch.isfinite(ratio).all():raise ValueError('Nonfinite exchange ratio')
    take=torch.rand(ratio.shape,dtype=w.dtype,device=w.device,generator=generator).log()<ratio.clamp_max(0)
    output=w.clone();out_logp=logp.clone();out_score=score.clone();out_labels=labels.clone()
    scale=(tj/ti).sqrt()[None,:,None]
    output[:,left]=torch.where(take[...,None],scale*w[:,right],w[:,left])
    output[:,right]=torch.where(take[...,None],w[:,left]/scale,w[:,right])
    out_logp[:,left]=torch.where(take,(tj/ti)*logp[:,right],logp[:,left])
    out_logp[:,right]=torch.where(take,(ti/tj)*logp[:,left],logp[:,right])
    out_score[:,left]=torch.where(take[...,None],scale*score[:,right],score[:,left])
    out_score[:,right]=torch.where(take[...,None],score[:,left]/scale,score[:,right])
    out_labels[:,left]=torch.where(take,labels[:,right],labels[:,left])
    out_labels[:,right]=torch.where(take,labels[:,left],labels[:,right])
    return output,DensityValue(out_logp.reshape(-1),out_score.reshape(-1,dim)),out_labels,dict(
        parity=parity,left_slots=left.cpu().tolist(),accepted=take.cpu(),log_ratios=ratio.cpu())
