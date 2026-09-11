"""Symmetric stochastic relaxation around an invertible coordinate exchange.

An augmented-path Metropolis proposal, not a finite-time endpoint density or
Jarzynski estimator. The caller supplies endpoint support and action ratios.
"""
import math
import torch
from cfm_mol.nonequilibrium import _states
from cfm_mol.tempered_smc import DensityValue,_clip_score


def gaussian_path_logs(x,y,score_x,score_y,*,std,max_score_norm):
    mean_forward=x+.5*std**2*_clip_score(score_x,max_score_norm)
    mean_reverse=y+.5*std**2*_clip_score(score_y,max_score_norm)
    constant=x.shape[-1]*math.log(std*math.sqrt(2*math.pi))
    forward=-.5*((y-mean_forward)/std).square().sum(-1)-constant
    reverse=-.5*((x-mean_reverse)/std).square().sum(-1)-constant
    return forward,reverse,mean_forward,mean_reverse


@torch.no_grad()
def escorted_path(x0,smooth,transform,*,steps_per_side,std,max_score_norm,generator,
                  initial_value=None,phase='escorted'):
    """Propose k Gaussian steps, one map, then k Gaussian steps.

    smooth(x, phase=...) must return a finite smooth DensityValue everywhere
    visited, including outside the final hard support. transform(x) returns
    (y, log_abs_intrinsic_Jacobian). The reverse uses the inverse map and the
    reversed Gaussian path, with the same scalar std and force clipping.
    """
    _states(x0)
    if (not isinstance(steps_per_side,int) or steps_per_side<0
            or any(not math.isfinite(t) or t<=0 for t in [std,max_score_norm])):
        raise ValueError('Nonnegative path length and positive finite scales required')
    x=x0.double().clone()
    value=(smooth(x,phase=phase+'/initial') if initial_value is None else initial_value).validate(x,True)
    initial_log=value.log_value.clone();vertices=[x.clone()];values=[value.log_value.clone()];scores=[value.score.clone()]
    edges=[];ratio=torch.zeros(len(x),dtype=x.dtype,device=x.device)
    def propagate(label):
        nonlocal x,value,ratio
        mean=x+.5*std**2*_clip_score(value.score,max_score_norm)
        noise=torch.randn(x.shape,dtype=x.dtype,device=x.device,generator=generator)
        y=mean+std*noise;new=smooth(y,phase=phase+'/'+label).validate(y,True)
        forward,reverse,mf,mr=gaussian_path_logs(x,y,value.score,new.score,std=std,max_score_norm=max_score_norm)
        edges.append(dict(source=len(vertices)-1,destination=len(vertices),noise=noise,
            forward_mean=mf,reverse_mean=mr,log_forward=forward,log_reverse=reverse))
        ratio+=reverse-forward;x=y;value=new
        vertices.append(x.clone());values.append(value.log_value.clone());scores.append(value.score.clone())
    for i in range(steps_per_side):propagate(f'pre_{i}')
    mapped,log_volume=transform(x)
    if mapped.shape!=x.shape or log_volume.shape!=(len(x),) or not torch.isfinite(log_volume).all():
        raise ValueError('Invalid map result or intrinsic volume')
    _states(mapped);value=smooth(mapped,phase=phase+'/map').validate(mapped,True);x=mapped
    vertices.append(x.clone());values.append(value.log_value.clone());scores.append(value.score.clone())
    for i in range(steps_per_side):propagate(f'post_{i}')
    return x,value,dict(vertices=torch.stack(vertices),smooth_log_values=torch.stack(values),
        smooth_scores=torch.stack(scores),edges=edges,map_source=steps_per_side,map_destination=steps_per_side+1,
        log_volume=log_volume,path_log_ratio=ratio,
        smooth_log_acceptance_ratio=value.log_value-initial_log+ratio+log_volume,
        steps_per_side=steps_per_side,std=std,max_score_norm=max_score_norm,
        evaluated_vertices=(2*steps_per_side+1)+(initial_value is None),
        endpoint_density='unknown finite-time law; augmented path MH only')


def reverse_augmented_vertices(free_vertices,steps_per_side,transform):
    """Reverse free path coordinates, including the nontrivial middle Jacobian.

    free_vertices has the start and every stochastic destination, but excludes
    the deterministic map destination. This helper remains differentiable for
    full augmented Jacobian/involution tests.
    """
    if len(free_vertices)!=2*steps_per_side+1:raise ValueError('Incorrect augmented path shape')
    full=list(free_vertices.unbind(0));full.insert(steps_per_side+1,transform(free_vertices[steps_per_side]))
    reverse=list(reversed(full));del reverse[steps_per_side+1]
    return torch.stack(reverse)
