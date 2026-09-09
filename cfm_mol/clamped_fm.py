"""Flow matching for the explicitly defined composition-clamped q_T sampler.

For scheduler alpha, tau(t)=alpha(t)/alpha(T). An independent COM-free Gaussian
X0 and a data endpoint X1 define Xt=X0+tau(t)*(X1-X0). The existing endpoint
head must predict D*=X0+(X1-X0)/alpha(T), because its velocity conversion is
alpha'(t)/(1-alpha(t))*(D-Xt). This trains data to time T, not time one.
The separately trained displacement head defines v=D-Xt and predicts
D*=Xt+alpha'(t)/alpha(T)*(X1-X0), without an endpoint conversion singularity.
"""
import math
import torch
from .clamped_density import center_by_graph,deterministic_field


def clamped_fm_path(graph,node_batch_idx,scheduler,*,terminal_time=0.8,
                    prior_std=1.,generator=None,parameterization='endpoint'):
    if 'has_reference_geometry' in graph.ndata and not graph.ndata['has_reference_geometry'].all():
        raise ValueError('Condition-only placeholder coordinates cannot be used as FM targets')
    if not math.isfinite(terminal_time) or not 0<terminal_time<=1:
        raise ValueError('terminal_time must be in (0,1]')
    if not math.isfinite(prior_std) or prior_std<=0:
        raise ValueError('prior_std must be positive and finite')
    x1=center_by_graph(graph.ndata['x_1_true'],node_batch_idx,graph.batch_size)
    x0=torch.randn(x1.shape,device=x1.device,dtype=x1.dtype,generator=generator)*prior_std
    x0=center_by_graph(x0,node_batch_idx,graph.batch_size)
    t=torch.rand((graph.batch_size,),device=x1.device,dtype=x1.dtype,generator=generator)*terminal_time
    feature=list(scheduler.feats).index('x')
    alpha=scheduler.alpha_t(t.clone())[:,feature]
    final_alpha=scheduler.alpha_t(t.new_full(t.shape,terminal_time))[:,feature]
    if (final_alpha<=0).any():raise ValueError('Position schedule must have alpha(T)>0')
    delta=x1-x0
    xt=x0+(alpha/final_alpha)[node_batch_idx,None]*delta
    endpoint_target=x0+delta/final_alpha[node_batch_idx,None]
    if parameterization=='displacement':
        prime=scheduler.alpha_t_prime(t.clone())[:,feature]
        endpoint_target=xt+(prime/final_alpha)[node_batch_idx,None]*delta
    elif parameterization!='endpoint':
        raise ValueError('Training supports endpoint or displacement heads')
    return xt,t,endpoint_target,{'x0':x0,'x1':x1,'alpha_T':final_alpha}


def clamped_fm_loss(model,graph,node_batch_idx,upper_edge_mask,*,terminal_time=0.8,
                    prior_std=1.,generator=None,parameterization='endpoint'):
    """Equal-molecule head MSE for a memoryless, unaligned clamped flow.

    It is a positive time-weighting of the velocity FM loss and has the same
    conditional regression optimum for endpoint heads; for displacement heads
    it is exactly velocity MSE. The field's stochastic/history branch is
    disabled exactly as in the density and sampler. No steric retraction occurs.
    """
    xt,t,target,_=clamped_fm_path(graph,node_batch_idx,
        model.vector_field.interpolant_scheduler,terminal_time=terminal_time,
        prior_std=prior_std,generator=generator,parameterization=parameterization)
    with graph.local_scope(),deterministic_field(model.vector_field):
        graph.ndata['x_t']=xt
        for key in ('a','c'):graph.ndata[f'{key}_t']=graph.ndata[f'{key}_1_true']
        graph.edata['e_t']=graph.edata['e_1_true']
        prediction=model.vector_field(graph,t,node_batch_idx=node_batch_idx,upper_edge_mask=upper_edge_mask)['x']
        prediction=center_by_graph(prediction,node_batch_idx,graph.batch_size)
        atom_loss=(prediction-target).square().mean(-1)
        counts=torch.bincount(node_batch_idx,minlength=graph.batch_size).to(atom_loss)
        loss=atom_loss.new_zeros(graph.batch_size).index_add(0,node_batch_idx,atom_loss)/counts
        return loss.mean()
