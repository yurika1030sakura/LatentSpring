"""Flow matching for the explicitly defined composition-clamped q_T sampler.

For scheduler alpha, tau(t)=alpha(t)/alpha(T). An independent COM-free Gaussian
X0 and a data endpoint X1 define Xt=X0+tau(t)*(X1-X0). The existing endpoint
head must predict D*=X0+(X1-X0)/alpha(T), because its velocity conversion is
alpha'(t)/(1-alpha(t))*(D-Xt). This trains data to time T, not time one.
The separately trained displacement head defines v=D-Xt and predicts
D*=Xt+alpha'(t)/alpha(T)*(X1-X0), without an endpoint conversion singularity.

Optional SO(3)-orbit pairing restores the Gaussian source with a shared Haar
rotation and keeps the rotation-symmetrized data endpoint law. The endpoints
are then correlated: an independent-Gaussian velocity-to-score identity does
not apply. This option changes the FM coupling, never the inference/density API.
"""
import math
import torch
from .clamped_density import center_by_graph,deterministic_field


def clamped_fm_path(graph,node_batch_idx,scheduler,*,terminal_time=0.8,
                    prior_std=1.,generator=None,parameterization='endpoint',
                    pairing=None,pairing_radii=None,pairing_generator=None,prior_positions=None):
    if 'has_reference_geometry' in graph.ndata and not graph.ndata['has_reference_geometry'].all():
        raise ValueError('Condition-only placeholder coordinates cannot be used as FM targets')
    if not math.isfinite(terminal_time) or not 0<terminal_time<=1:
        raise ValueError('terminal_time must be in (0,1]')
    if not math.isfinite(prior_std) or prior_std<=0:
        raise ValueError('prior_std must be positive and finite')
    x1=center_by_graph(graph.ndata['x_1_true'],node_batch_idx,graph.batch_size)
    if prior_positions is None:
        x0=torch.randn(x1.shape,device=x1.device,dtype=x1.dtype,generator=generator)*prior_std
    else:
        if prior_positions.shape!=x1.shape or not torch.isfinite(prior_positions).all():
            raise ValueError('Declared prior positions must be finite and match the graph')
        x0=prior_positions.to(x1).detach().clone()
    x0=center_by_graph(x0,node_batch_idx,graph.batch_size)
    t=torch.rand((graph.batch_size,),device=x1.device,dtype=x1.dtype,generator=generator)*terminal_time
    pairing_records=[]
    if pairing is not None:
        from .orbit_pairing import orbit_pair
        if pairing_radii is None or pairing_radii.shape!=(len(x1),) or pairing_generator is None:
            raise ValueError('Orbit pairing requires radii and an explicit separate generator')
        x0,x1=x0.clone(),x1.clone()
        for i in range(graph.batch_size):
            selected=node_batch_idx==i
            if pairing=='typed_rotation':
                from .orbit_pairing import typed_orbit_pair
                src,_=graph.edges();edges=graph.edata['e_1_true'][node_batch_idx[src]==i]
                if len(edges) and not torch.equal(edges,edges[:1].expand_as(edges)):
                    raise ValueError('Typed matching requires permutation-invariant edge conditioning')
                a=graph.ndata['a_1_true'][selected].argmax(-1)
                c=graph.ndata['c_1_true'][selected].argmax(-1)
                groups=a*graph.ndata['c_1_true'].shape[-1]+c
                x0[selected],x1[selected],record=typed_orbit_pair(x0[selected],x1[selected],pairing_radii[selected],groups,generator=pairing_generator)
            else:
                x0[selected],x1[selected],record=orbit_pair(x0[selected],x1[selected],pairing_radii[selected],
                    mode=pairing,generator=pairing_generator)
            pairing_records.append(record)
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
    return xt,t,endpoint_target,{'x0':x0,'x1':x1,'alpha_T':final_alpha,'pairing_records':pairing_records}


def clamped_fm_loss(model,graph,node_batch_idx,upper_edge_mask,*,terminal_time=0.8,
                    prior_std=1.,generator=None,parameterization='endpoint',
                    pairing=None,pairing_radii=None,pairing_generator=None,pairing_diagnostics=None,prior_positions=None):
    """Equal-molecule head MSE for a memoryless clamped flow.

    It is a positive time-weighting of the velocity FM loss and has the same
    conditional regression optimum for endpoint heads; for displacement heads
    it is exactly velocity MSE. The field's stochastic/history branch is
    disabled exactly as in the density and sampler. No steric retraction occurs.
    """
    if prior_positions is None and getattr(model,'_research_prior_kind','gaussian')!='gaussian':
        raise ValueError('Declared non-Gaussian source requires explicit prior training samples')
    xt,t,target,info=clamped_fm_path(graph,node_batch_idx,
        model.vector_field.interpolant_scheduler,terminal_time=terminal_time,
        prior_std=prior_std,generator=generator,parameterization=parameterization,
        pairing=pairing,pairing_radii=pairing_radii,pairing_generator=pairing_generator,prior_positions=prior_positions)
    if pairing_diagnostics is not None:pairing_diagnostics.extend(info['pairing_records'])
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
