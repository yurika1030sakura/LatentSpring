"""Normalized S2 proposals for fixed-radius molecular internal-coordinate moves.

These are conditional angular densities, NOT full Cartesian endpoint densities.
The radial r^2 factor cancels only when the same radius and passive coordinates
are retained. Hard-support candidates still require correct MH rejection.
"""
import math
import torch


def vmf_log_normalizer(natural_parameter):
    k=natural_parameter.norm(dim=1);safe=k.clamp_min(1e-12)
    regular=safe.log()-math.log(2*math.pi)-safe-(-torch.expm1(-2*safe)).log()
    series=-math.log(4*math.pi)-k.square()/6+k.pow(4)/180-k.pow(6)/2835
    log_constant=torch.where(k<1e-3,series,regular)
    return log_constant


def vmf_log_prob(direction,natural_parameter):
    if direction.shape!=natural_parameter.shape or direction.ndim!=2 or direction.shape[1]!=3:
        raise ValueError('Matched batches of three-dimensional vectors required')
    if (not torch.isfinite(direction).all() or not torch.isfinite(natural_parameter).all()
            or not torch.allclose(direction.norm(dim=1),torch.ones(len(direction),dtype=direction.dtype,device=direction.device),atol=1e-8,rtol=0)):
        raise ValueError('Finite natural parameters and unit directions required')
    return vmf_log_normalizer(natural_parameter)+(natural_parameter*direction).sum(1)


@torch.no_grad()
def vmf_sample(natural_parameter,*,generator):
    if (natural_parameter.ndim!=2 or natural_parameter.shape[1]!=3
            or not torch.isfinite(natural_parameter).all()):raise ValueError('Finite [batch,3] natural parameters required')
    eta=natural_parameter;k=eta.norm(dim=1);positive=k>0
    safe=k.clamp_min(torch.finfo(eta.dtype).tiny);mean=eta/safe[:,None]
    uniform=torch.rand(len(eta),dtype=eta.dtype,device=eta.device,generator=generator)
    gaussian=torch.randn(eta.shape,dtype=eta.dtype,device=eta.device,generator=generator)
    # Two numerically stable forms of the same exact inverse CDF for cos(theta).
    small=1+torch.log1p((1-uniform)*torch.expm1(-2*safe))/safe
    large=1+torch.logaddexp(uniform.log(),torch.log1p(-uniform)-2*safe)/safe
    cosine=torch.where(k<1,small,large).clamp(-1,1)
    tangent=gaussian-(gaussian*mean).sum(1,keepdim=True)*mean
    norm=tangent.norm(dim=1,keepdim=True)
    if ((norm[:,0]<1e-14)&positive).any():raise ValueError('Degenerate angular noise draw')
    tangent=tangent/norm.clamp_min(1e-300)
    direction=cosine[:,None]*mean+(1-cosine.square()).clamp_min(0).sqrt()[:,None]*tangent
    sphere=gaussian/gaussian.norm(dim=1,keepdim=True)
    return torch.where(positive[:,None],direction,sphere),dict(uniform=uniform,gaussian=gaussian)


def force_vmf_parameter(direction,force,*,radius,kT,concentration):
    """A classical force-informed angular proposal with a bounded tangent drift.

    force must be the COM-projected physical force at the moved leaf, including
    the declared restraint. The same selected concentration label must be used
    when recomputing the reverse parameter at the proposed state.
    """
    if direction.shape!=force.shape or direction.ndim!=2 or direction.shape[1]!=3:
        raise ValueError('Matched direction/force batches required')
    if not math.isfinite(kT) or kT<=0:raise ValueError('Positive temperature required')
    c=torch.as_tensor(concentration,dtype=direction.dtype,device=direction.device).expand(len(direction))
    r=torch.as_tensor(radius,dtype=direction.dtype,device=direction.device).expand(len(direction))
    if (c<=0).any() or (r<=0).any() or not torch.isfinite(c).all() or not torch.isfinite(r).all():
        raise ValueError('Positive finite radii and concentration labels required')
    score=(force-(force*direction).sum(1,keepdim=True)*direction)*r[:,None]/kT
    bound=4*c.sqrt();norm=score.norm(dim=1)
    score=score*torch.minimum(torch.ones_like(norm),bound/norm.clamp_min(1e-300))[:,None]
    return c[:,None]*direction+.5*score
