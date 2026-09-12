"""Oracle-only labels for bounded gates that retain ordinary MH acceptance.

These use the ALREADY QUERIED true MH ratio. They are training/audit targets,
never a deployable pre-query gate or a source of actual query savings.
"""
import math
import torch


def oracle_gate_targets(log_ratio,bound=math.log(16)):
    if not 0<bound<float('inf') or not torch.isfinite(log_ratio).all() or log_ratio.requires_grad:
        raise ValueError('Finite fixed oracle log ratios and positive finite bound required')
    return log_ratio.clamp(min=-bound,max=0),(-log_ratio).clamp(min=-bound,max=0)


def log_retention_lower_bound(log_ratio,forward,reverse,bound=math.log(16)):
    if not (log_ratio.shape==forward.shape==reverse.shape):
        raise ValueError('Matched per-pair arrays required')
    if any(not torch.isfinite(g).all() or (g>0).any() or (g < -bound-1e-12).any() for g in [forward,reverse]):
        raise ValueError('Finite bounded log gate probabilities required')
    target_f,target_r=oracle_gate_targets(log_ratio,bound)
    deficit=torch.maximum((target_f-forward).clamp_min(0),(target_r-reverse).clamp_min(0))
    return -deficit
