"""Normalized great-circle proposals inside explicit angular distance constraints.

The score is linearly interpolated in angle and exponentiated on each segment.
Sampling and density evaluation use that implemented law, not an unnormalized
continuous-score approximation. A failed single circle draw is a self transition.
Geodesic walks are established MCMC; this module makes no novelty claim.
"""
import math
import torch

TAU=2*math.pi


def intersect_arcs(normals,limits,base,tangent):
    """Return all positive-length intervals satisfying w dot u(theta) <= c."""
    intervals=[(0.,TAU)]
    for aa,bb,cc in zip((normals@base).tolist(),(normals@tangent).tolist(),limits.tolist()):
        amplitude=math.hypot(aa,bb)
        if amplitude<1e-14:
            if cc<0:return []
            continue
        ratio=cc/amplitude
        if ratio>=1:continue
        if ratio<=-1:return []
        half=math.acos(ratio);start=(math.atan2(bb,aa)+half)%TAU
        width=TAU-2*half;end=start+width
        allowed=[(start,min(end,TAU))]
        if end>TAU:allowed.append((0.,end-TAU))
        next_intervals=[]
        for left,right in intervals:
            for low,high in allowed:
                a,b=max(left,low),min(right,high)
                if b>a:next_intervals.append((a,b))
        intervals=sorted(next_intervals)
        if not intervals:return []
    return intervals


def root_distance_constraints(x,root,radius,radii,*,contact_factor=1.25,overlap_factor=.6,margin=1e-8):
    """Sufficient distance constraints for a single root attached only to anchor.

The caller supplies the desired single-bond graph. Passive geometry and graph
qualification remain the caller's responsibility. These thresholds match this
repository's useVdw=True connectivity rule; final graph validation is retained.
"""
    leaf,anchor=root
    if leaf==anchor or x.shape!=(len(radii),3) or not torch.isfinite(x).all() or radius<=0:
        raise ValueError('Finite anchored geometry and positive radius required')
    bond=float(radii[leaf]+radii[anchor])
    if not overlap_factor*bond+margin<float(radius)<contact_factor*bond-margin:
        return x.new_zeros(1,3),x.new_tensor([-1.])
    indices=[i for i in range(len(x)) if i not in (leaf,anchor)]
    vectors=x[indices]-x[anchor]
    separation=contact_factor*(radii[leaf]+radii[indices])+margin
    limits=(radius**2+vectors.square().sum(1)-separation.square())/(2*radius)
    return vectors,limits


def log_exprel(value):
    """Stable log((exp(value)-1)/value), including its zero limit."""
    size=value.abs();safe=size.clamp_min(1e-12)
    regular=value.clamp_min(0)+torch.log(-torch.expm1(-safe))-safe.log()
    series=value/2+value.square()/24-value.pow(4)/2880
    return torch.where(size<1e-4,series,regular)


def circle_law(base,tangent,normals,limits,eta,*,max_segment_width=math.pi/64):
    if base.shape!=(3,) or tangent.shape!=(3,) or eta.shape!=(3,) or normals.shape!=(len(limits),3):
        raise ValueError('Three-dimensional directions, score and halfspaces required')
    if not all(torch.isfinite(t).all() for t in [base,tangent,normals,limits,eta]):
        raise ValueError('Finite proposal parameters required')
    if not 0<max_segment_width<=math.pi:raise ValueError('Positive bounded segment width required')
    torch.testing.assert_close(base.norm(),base.new_tensor(1.),atol=1e-9,rtol=0)
    torch.testing.assert_close(tangent.norm(),base.new_tensor(1.),atol=1e-9,rtol=0)
    if abs(float(base@tangent))>1e-9:raise ValueError('Orthogonal circle frame required')
    arcs=intersect_arcs(normals,limits,base,tangent)
    segments=[]
    for left,right in arcs:
        count=max(1,math.ceil((right-left)/max_segment_width))
        for i in range(count):segments.append((left+(right-left)*i/count,left+(right-left)*(i+1)/count))
    if not segments:return None
    edges=base.new_tensor(segments);width=edges[:,1]-edges[:,0]
    directions=edges.cos()[...,None]*base+edges.sin()[...,None]*tangent
    heights=directions@eta;delta=heights[:,1]-heights[:,0]
    log_masses=width.log()+heights[:,0]+log_exprel(delta)
    normalizer=torch.logsumexp(log_masses,0)
    return dict(arcs=arcs,edges=edges,width=width,heights=heights,delta=delta,
                log_masses=log_masses,log_normalizer=normalizer)


def angle_log_prob(theta,law):
    if law is None:return None
    theta=torch.as_tensor(theta,dtype=law['edges'].dtype,device=law['edges'].device)
    contained=(theta>=law['edges'][:,0])&(theta<=law['edges'][:,1])
    indices=contained.nonzero().flatten()
    if not len(indices):return theta.new_tensor(-torch.inf)
    i=indices[0];fraction=(theta-law['edges'][i,0])/law['width'][i]
    return law['heights'][i,0]+law['delta'][i]*fraction-law['log_normalizer']


def inverse_exponential_fraction(uniform,delta):
    safe=torch.where(delta.abs()<1e-5,torch.ones_like(delta),delta)
    regular=torch.logaddexp(torch.log1p(-uniform),uniform.log()+delta)/safe
    series=uniform+delta*uniform*(1-uniform)/2+delta.square()*uniform*(1-uniform)*(1-2*uniform)/6
    return torch.where(delta.abs()<1e-5,series,regular)


def draw_angle(law,*,generator):
    i=int(torch.multinomial(torch.softmax(law['log_masses'],0),1,generator=generator))
    uniform=torch.rand((),dtype=law['edges'].dtype,device=law['edges'].device,generator=generator)
    fraction=inverse_exponential_fraction(uniform,law['delta'][i])
    theta=law['edges'][i,0]+law['width'][i]*fraction
    return theta,dict(segment=i,uniform=uniform,fraction=fraction)


def exponential_fraction_mean(delta):
    """Mean coordinate under a density proportional to exp(delta*t), 0<t<1."""
    size=delta.abs().clamp_min(1e-12)
    positive=-1/torch.expm1(-size)-1/size
    regular=torch.where(delta>=0,positive,1-positive)
    series=.5+delta/12-delta.pow(3)/720+delta.pow(5)/30240
    return torch.where(delta.abs()<1e-3,series,regular)


def arc_teacher_kl(teacher,student):
    """Exact KL between two implemented angle laws on a common geometric grid.

    Averaging this conditional KL over the shared random-circle law bounds the
    KL of their marginal direction kernels by data processing. This is not a
    bound on the molecular target error or on a coupled graph-exchange kernel.
    """
    if teacher is None or student is None:raise ValueError('Nonempty common angle support required')
    torch.testing.assert_close(teacher['edges'],student['edges'],atol=0,rtol=0)
    probabilities=torch.softmax(teacher['log_masses'].detach(),0)
    mean=exponential_fraction_mean(teacher['delta'].detach())
    difference=teacher['heights'][:,0].detach()-student['heights'][:,0]
    difference=difference+(teacher['delta'].detach()-student['delta'])*mean
    return (probabilities*difference).sum()-teacher['log_normalizer'].detach()+student['log_normalizer']


def direction_log_prob(observed,base,normals,limits,eta,*,max_segment_width=math.pi/64,chart_tolerance=1e-10):
    """Sphere density includes BOTH oriented-circle preimages and |sin theta|."""
    torch.testing.assert_close(observed.norm(),observed.new_tensor(1.),atol=1e-9,rtol=0)
    cosine=base@observed;perpendicular=observed-cosine*base;sine=perpendicular.norm()
    if float(sine)<=chart_tolerance:return observed.new_tensor(-torch.inf),dict(chart_rejected=True)
    tangent=perpendicular/sine;theta=torch.atan2(sine,cosine)
    logs=[]
    for sign,angle in [(1,theta),(-1,TAU-theta)]:
        law=circle_law(base,sign*tangent,normals,limits,eta,max_segment_width=max_segment_width)
        value=angle_log_prob(angle,law)
        logs.append(observed.new_tensor(-torch.inf) if value is None else value)
    result=torch.logsumexp(torch.stack(logs),0)-math.log(TAU)-sine.log()
    return result,dict(theta=theta,tangent=tangent,oriented_angle_log_probabilities=torch.stack(logs),
                       log_spherical_jacobian=sine.log(),chart_rejected=False)


@torch.no_grad()
def draw_direction(base,normals,limits,eta,*,generator,max_segment_width=math.pi/64,chart_tolerance=1e-10):
    noise=torch.randn(3,dtype=base.dtype,device=base.device,generator=generator)
    tangent=noise-(noise@base)*base;length=tangent.norm()
    if float(length)<1e-12:return None,base.new_tensor(-torch.inf),dict(failure='Degenerate circle frame',noise=noise)
    tangent=tangent/length
    law=circle_law(base,tangent,normals,limits,eta,max_segment_width=max_segment_width)
    if law is None:return None,base.new_tensor(-torch.inf),dict(failure='Empty admissible circle',noise=noise,tangent=tangent)
    theta,random=draw_angle(law,generator=generator)
    direction=theta.cos()*base+theta.sin()*tangent
    logq,density=direction_log_prob(direction,base,normals,limits,eta,
        max_segment_width=max_segment_width,chart_tolerance=chart_tolerance)
    trace=dict(noise=noise,tangent=tangent,theta=theta,random=random,arcs=law['arcs'],
               log_circle_normalizer=law['log_normalizer'],density=density)
    if not torch.isfinite(logq):
        trace['failure']='Numerically excluded density chart';return None,logq,trace
    return direction,logq,trace
