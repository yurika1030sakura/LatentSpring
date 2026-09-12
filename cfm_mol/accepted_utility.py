"""Actual-MH accepted utility under recorded behavioral proposals."""
import torch


def accepted_importance_utility(log_forward,log_reverse,log_behavior,target_log_ratio,action_log_ratio,reward):
    """Return signed utility and its NONCLIPPED accepted-flow importance factor.

    Inputs concern supported, scored attempts. Unsupported attempts must remain
    in the external denominator with zero utility. For coordinate-only learning,
    densities are coordinate laws and action_log_ratio is the unchanged reverse
    minus forward action log probability. For learned action selection, pass FULL
    action-plus-coordinate learned/behavior densities and action_log_ratio=0.
    Never add the uniform action-count ratio a second time with full densities.
    The identity does not qualify behavioral support or importance-weight ESS.
    """
    if not all(t.shape==log_forward.shape for t in [log_reverse,log_behavior,target_log_ratio,action_log_ratio,reward]):
        raise ValueError('Matched per-attempt tensors required')
    if not all(torch.isfinite(t).all() for t in [log_behavior,target_log_ratio,action_log_ratio,reward]):
        raise ValueError('Finite behavioral density, target/action ratios and reward required')
    if any(torch.isnan(t).any() or torch.isposinf(t).any() for t in [log_forward,log_reverse]):
        raise ValueError('Learned log densities may be finite or minus infinity only')
    first=log_forward-log_behavior
    second=target_log_ratio+action_log_ratio+log_reverse-log_behavior
    log_flow=torch.minimum(first,second)
    flow=log_flow.exp()
    if not torch.isfinite(flow).all():raise ValueError('Importance overflow; do not silently clip it')
    return reward*flow,flow
