"""Score-function gradient of MH-accepted utility, including hard target support.

This follows the standard score identity and differentiation of min(1, ratio).
It is not a new thermodynamic identity. Proposal coordinates must be held fixed
when differentiating their log densities; a forward reparameterized log density
is NOT the required proposal score.
"""
import torch


def mh_utility_score_loss(log_forward,log_reverse,log_ratio,reward,*,baseline=0.):
    """Negative accepted-utility gradient via a surrogate scalar loss.

    log_forward/log_reverse must be evaluated at fixed sampled coordinates.
    The baseline may depend on the old state or past batches, never the current
    sampled proposal. It multiplies the forward score for ALL proposals,
    including those outside support. At positive-mass MH ties the usual
    differentiability caveat applies; this uses the forward branch.
    """
    if not (log_forward.shape==log_reverse.shape==log_ratio.shape==reward.shape):
        raise ValueError('One matching log density, ratio and utility per proposal required')
    if (not torch.isfinite(log_forward).all() or not torch.isfinite(log_reverse).all()
            or not torch.isfinite(reward).all() or torch.isnan(log_ratio).any() or torch.isposinf(log_ratio).any()):
        raise ValueError('Invalid proposal utility inputs; minus-infinite target ratios are allowed')
    ratio=log_ratio.detach();weight=ratio.clamp_max(0).exp()*reward.detach()
    selected=torch.where(ratio>=0,log_forward,log_reverse)
    baseline=torch.as_tensor(baseline,dtype=log_forward.dtype,device=log_forward.device).detach()
    if baseline.shape not in [torch.Size([]),log_forward.shape] or not torch.isfinite(baseline).all():
        raise ValueError('Baseline must be finite and scalar or per old state')
    return -(weight*selected-baseline*log_forward).mean()


def proposal_utility_score_loss(model,x,y,log_target_x,log_target_y,reward,numbers,electronic,*,baseline=0.):
    """Evaluate the necessary inverse-density scores at detached x/y samples."""
    x=x.detach();y=y.detach()
    forward=model.log_prob(y,x,numbers,electronic)
    valid=torch.isfinite(log_target_y)
    if not torch.isfinite(log_target_x).all() or torch.isnan(log_target_y).any() or torch.isposinf(log_target_y).any():
        raise ValueError('Old states must lie in support; proposed states may have zero density')
    reverse=torch.zeros_like(forward)
    if valid.any():
        state=electronic[valid] if electronic.ndim==2 else electronic
        reverse[valid]=model.log_prob(x[valid],y[valid],numbers,state)
    ratio=log_target_y.detach()-log_target_x.detach()+reverse-forward
    return mh_utility_score_loss(forward,reverse,ratio,reward,baseline=baseline)
