"""Differentiable density of a deterministic, composition-clamped position ODE.

This is a corrected diagnostic/training path, not a reinterpretation of archived
BGFM results. FlowMol3's endpoint head D(x,t) is converted to velocity using its
position schedule. Coordinates AND vector-field outputs are projected onto the
zero-centroid subspace before taking derivatives, so the ambient trace equals
the intrinsic trace. The midpoint solver differentiates the state and prior.

The default terminal time is 0.95: q_0.95 is a separate, explicitly defined
clamped-flow density, not the endpoint or the joint CTMC generator likelihood.
Passing terminal_time=1 is allowed with interior-stage midpoint evaluations,
but requires a convergence study for the particular learned endpoint field.
Steric retractions, guidance, and history-dependent self-conditioning are not
part of this smooth ODE; their sample law cannot be scored by this function.
"""
from __future__ import annotations

from contextlib import contextmanager
import math

import torch


def center_by_graph(x, node_batch_idx, n_graphs):
    sums = x.new_zeros((n_graphs, x.shape[-1])).index_add(0, node_batch_idx, x)
    counts = torch.bincount(node_batch_idx, minlength=n_graphs).to(x).unsqueeze(-1)
    if (counts == 0).any():
        raise ValueError("Every graph must have at least one atom")
    return x - (sums / counts)[node_batch_idx]


@contextmanager
def deterministic_field(vector_field):
    """Disable dropout/random self-conditioning without disabling autograd.

    Restore each module's training flag, including mixed train/eval states.
    Calls use no cached previous prediction, defining a memoryless field.
    """
    states = [(module, module.training) for module in vector_field.modules()]
    # eval() alone still invokes FlowMol's bootstrap self-conditioning exactly
    # at t=0. Adaptive solvers evaluate that boundary; a memoryless field must
    # also disable this special branch, then restore the caller's setting.
    conditioning = getattr(vector_field, 'self_conditioning', None)
    try:
        vector_field.eval()
        if conditioning is not None:
            vector_field.self_conditioning = False
        yield
    finally:
        if conditioning is not None:
            vector_field.self_conditioning = conditioning
        for module, training in states:
            module.training = training


def position_velocity(model, graph, x, t, node_batch_idx, upper_edge_mask,
                      *, parameterization="endpoint", kT=None):
    """Convert a position head to P v(P x,t), using the configured schedule.

    ``parameterization`` is explicit to avoid guessing the semantics of a
    tensor called ``x``. Endpoint heads require a FlowMol3-style scheduler;
    a true velocity head is useful for analytic tests and alternative models.
    The helper deliberately avoids patched sampling methods and stale steric
    context: nonsmooth retractions have no ordinary CNF change of variables.
    """
    if parameterization not in {"endpoint", "velocity", "displacement"}:
        raise ValueError("parameterization must be endpoint, displacement or velocity")
    n_graphs = graph.batch_size
    x_centered = center_by_graph(x, node_batch_idx, n_graphs)
    graph.ndata["x_t"] = x_centered
    kwargs = dict(node_batch_idx=node_batch_idx, upper_edge_mask=upper_edge_mask)
    if kT is not None:
        kwargs["kT"] = kT
    prediction = model.vector_field(graph, t, **kwargs)["x"]
    if parameterization == 'displacement':
        # A separately trained research head: its raw coordinate increment is
        # the velocity. This is not the semantics of archived endpoint weights.
        prediction = prediction-x_centered
    if parameterization == "endpoint":
        scheduler = model.vector_field.interpolant_scheduler
        feature = list(scheduler.feats).index("x")
        alpha = scheduler.alpha_t(t.clone())[:, feature]
        alpha_prime = scheduler.alpha_t_prime(t.clone())[:, feature]
        if ((1-alpha) <= 0).any():
            raise ValueError("Endpoint-to-velocity conversion requires alpha_x(t) < 1")
        prediction = ((alpha_prime / (1-alpha))[node_batch_idx, None]
                      * (prediction - x_centered))
    return center_by_graph(prediction, node_batch_idx, n_graphs)


def _trace(v, x, node_batch_idx, n_graphs, probes, create_graph):
    """Trace of one already-computed field, also retaining it for the ODE step."""
    per_atom = x.new_zeros(x.shape[0])
    if not v.requires_grad:
        return x.new_zeros(n_graphs)
    if probes is None:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                grad = torch.autograd.grad(v[i, j], x, retain_graph=True,
                                           create_graph=create_graph, allow_unused=True)[0]
                if grad is not None:
                    basis = x.new_zeros(x.shape[0])
                    basis[i] = 1
                    per_atom = per_atom + basis * grad[i, j]
    else:
        for probe in probes:
            grad = torch.autograd.grad((v*probe).sum(), x, retain_graph=True,
                                       create_graph=create_graph, allow_unused=True)[0]
            if grad is not None:
                per_atom = per_atom + (grad*probe).sum(-1) / len(probes)
    return x.new_zeros(n_graphs).index_add(0, node_batch_idx, per_atom)


def _midpoint_step(model, graph, state, time, node_batch_idx, upper_edge_mask,
                   *, dt, n_hutchinson, n_trace_replicates, parameterization,
                   kT, xi_fn, step_index, probe_cache, for_training):
    """One deterministic coordinate step and its stochastic divergence value.

    The probe cache is populated once and reused during gradient recomputation.
    All graph/model context is re-established on every call, including backward.
    """
    n_graphs=graph.batch_size
    with graph.local_scope(), deterministic_field(model.vector_field):
        for key in ('a','c'):
            graph.ndata[f'{key}_t']=graph.ndata[f'{key}_1_true']
        graph.edata['e_t']=graph.edata['e_1_true']
        v_start=position_velocity(model,graph,state,time,node_batch_idx,
            upper_edge_mask,parameterization=parameterization,kT=kT)
        midpoint=state-0.5*dt*v_start
        if not for_training:
            midpoint=midpoint.detach().requires_grad_(True)
        v_mid=position_velocity(model,graph,midpoint,time,node_batch_idx,
            upper_edge_mask,parameterization=parameterization,kT=kT)
        if not probe_cache:
            for replica in range(n_trace_replicates):
                probes=None if n_hutchinson==0 else [
                    (xi_fn(step_index,replica*n_hutchinson+k,midpoint) if xi_fn is not None else
                     torch.randint(0,2,midpoint.shape,device=state.device).to(state)*2-1)
                    for k in range(n_hutchinson)]
                if probes is not None:
                    if any(probe.shape!=state.shape for probe in probes):
                        raise ValueError('Trace probe shape must match coordinates')
                    probes=[probe.to(state).detach() for probe in probes]
                probe_cache.append(probes)
        traces=[_trace(v_mid,midpoint,node_batch_idx,n_graphs,probes,for_training)
                for probes in probe_cache]
        return state-dt*v_mid,torch.stack(traces)


def _rk4_step(model, graph, state, time, node_batch_idx, upper_edge_mask,
              *, dt, n_hutchinson, n_trace_replicates, parameterization,
              kT, xi_fn, step_index, probe_cache, for_training):
    """Classical RK4 for the augmented coordinate/divergence ODE.

    Each replica shares its probe across the four stages of one step. Replicas
    remain independent, and cached probes are reused in the discrete adjoint.
    ``time`` is the step's midpoint time, matching the midpoint backend API.
    """
    n_graphs=graph.batch_size
    with graph.local_scope(),deterministic_field(model.vector_field):
        for key in ('a','c'):graph.ndata[f'{key}_t']=graph.ndata[f'{key}_1_true']
        graph.edata['e_t']=graph.edata['e_1_true']
        if not probe_cache:
            for replica in range(n_trace_replicates):
                probes=None if n_hutchinson==0 else [
                    (xi_fn(step_index,replica*n_hutchinson+k,state) if xi_fn is not None else
                     torch.randint(0,2,state.shape,device=state.device).to(state)*2-1)
                    for k in range(n_hutchinson)]
                if probes is not None:
                    if any(probe.shape!=state.shape for probe in probes):
                        raise ValueError('Trace probe shape must match coordinates')
                    probes=[probe.to(state).detach() for probe in probes]
                probe_cache.append(probes)
        velocities=[];divergences=[]
        for stage,(offset,fraction) in enumerate([(0.,.5),(.5,0.),(.5,0.),(1.,-.5)]):
            current=state if stage==0 else state-offset*dt*velocities[-1]
            if not for_training:current=current.detach().requires_grad_(True)
            velocity=position_velocity(model,graph,current,time+fraction*dt,node_batch_idx,
                upper_edge_mask,parameterization=parameterization,kT=kT)
            velocities.append(velocity)
            divergences.append(torch.stack([_trace(velocity,current,node_batch_idx,n_graphs,
                probes,for_training) for probes in probe_cache]))
        following=state-dt*(velocities[0]+2*velocities[1]+2*velocities[2]+velocities[3])/6
        divergence=(divergences[0]+2*divergences[1]+2*divergences[2]+divergences[3])/6
        return following,divergence


def log_density_clamped_flow(model, graph, node_batch_idx, upper_edge_mask,
                             *, n_ode_steps=32, n_hutchinson=2, prior_std=1.0,
                             terminal_time=0.95, for_training=False,
                             parameterization="endpoint", kT=None, xi_fn=None,
                             n_trace_replicates=1, checkpoint_steps=False,
                             discrete_adjoint=False, solver='midpoint'):
    """Log q_T(x), midpoint or RK4 integration of state AND log-Jacobian.

    With n_trace_replicates>1, return (replicates, graphs) estimates from
    independent traces on ONE deterministic trajectory. xi_fn sample indices
    run across all replicates, so callbacks must supply independent probes
    when independence is needed. The default returns (graphs,).

    ``xi_fn(step, sample, x)`` permits fixed probes for finite differences and
    comparisons. Zero probes selects exact divergence. No trajectory detach
    occurs during training: autograd gives the derivative of this discretised
    objective (not a claim of an unbiased continuous-likelihood gradient).
    Autocast is disabled; float16/bfloat16 inputs are promoted to float32.
    A local graph scope and model-mode restoration prevent side effects.
    Optional non-reentrant step checkpointing trades recomputation for memory;
    its per-step probe caches preserve both global and dedicated RNG streams.
    """
    if not isinstance(n_trace_replicates, int) or n_trace_replicates < 1:
        raise ValueError("n_trace_replicates must be a positive integer")
    if n_ode_steps < 1 or n_hutchinson < 0:
        raise ValueError("Require n_ode_steps >= 1 and n_hutchinson >= 0")
    if not math.isfinite(terminal_time) or not 0 < terminal_time <= 1:
        raise ValueError("terminal_time must be in (0, 1]")
    if not math.isfinite(prior_std) or prior_std <= 0:
        raise ValueError("prior_std must be positive and finite")
    if solver not in {'midpoint','rk4'}:raise ValueError('Unknown density solver')
    if solver=='rk4' and parameterization=='endpoint' and terminal_time==1:
        raise ValueError('RK4 endpoint-head evaluation requires terminal_time<1')
    if discrete_adjoint and for_training:
        if checkpoint_steps:
            raise ValueError('Select step checkpointing or the discrete adjoint, not both')
        from .clamped_adjoint import log_density_discrete_adjoint
        return log_density_discrete_adjoint(model,graph,node_batch_idx,upper_edge_mask,
            n_ode_steps=n_ode_steps,n_hutchinson=n_hutchinson,prior_std=prior_std,
            terminal_time=terminal_time,parameterization=parameterization,kT=kT,
            xi_fn=xi_fn,n_trace_replicates=n_trace_replicates,solver=solver)
    x = graph.ndata["x_1_true"].detach().clone()
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    if not torch.isfinite(x).all():
        raise ValueError("Non-finite input coordinates")
    n_graphs = graph.batch_size
    dt = terminal_time / n_ode_steps
    n_atoms = torch.bincount(node_batch_idx, minlength=n_graphs).to(x)
    integral = x.new_zeros((n_trace_replicates, n_graphs))
    with graph.local_scope(), deterministic_field(model.vector_field), \
            torch.enable_grad(), torch.autocast(device_type=x.device.type, enabled=False):
        # Clamp the discrete inputs explicitly, irrespective of prior graph state.
        for key in ("a", "c"):
            graph.ndata[f"{key}_t"] = graph.ndata[f"{key}_1_true"]
        graph.edata["e_t"] = graph.edata["e_1_true"]
        x = center_by_graph(x, node_batch_idx, n_graphs).requires_grad_(True)
        def make_step(step_index):
            probe_cache=[]
            def advance(state,time):
                step_function=_midpoint_step if solver=='midpoint' else _rk4_step
                return step_function(model,graph,state,time,node_batch_idx,upper_edge_mask,
                    dt=dt,n_hutchinson=n_hutchinson,n_trace_replicates=n_trace_replicates,
                    parameterization=parameterization,kT=kT,xi_fn=xi_fn,step_index=step_index,
                    probe_cache=probe_cache,for_training=for_training)
            return advance

        for step in range(n_ode_steps):
            # Both stages use an interior midpoint time. This is second-order
            # for smooth non-autonomous fields and avoids the t=1 singularity.
            t_mid = x.new_full((n_graphs,), terminal_time-(step+0.5)*dt)
            advance = make_step(step)
            if checkpoint_steps and for_training:
                from torch.utils.checkpoint import checkpoint
                x, divergence = checkpoint(advance, x, t_mid, use_reentrant=False,
                                            preserve_rng_state=False)
            else:
                x, divergence = advance(x, t_mid)
            integral = integral + dt*divergence
            if not for_training:
                x = x.detach().requires_grad_(True)
                integral = integral.detach()
        sq = x.new_zeros(n_graphs).index_add(0, node_batch_idx, x.square().sum(-1))
        logp = -0.5*sq/prior_std**2 - 1.5*(n_atoms-1)*math.log(2*math.pi*prior_std**2) - integral
        if not torch.isfinite(logp).all():
            raise FloatingPointError("Non-finite clamped-flow density; inspect field and resolution")
        result = logp[0] if n_trace_replicates == 1 else logp
        return result if for_training else result.detach()


@torch.no_grad()
def sample_clamped_flow(model, graph, node_batch_idx, upper_edge_mask, *,
                        n_ode_steps=128, terminal_time=0.95, prior_std=1.0,
                        parameterization="endpoint", kT=None, x0=None,
                        generator=None, solver='midpoint'):
    """Sample the SAME memoryless COM-free ODE used by the density diagnostic.

    The condition is the graph's fixed discrete composition; no bonds, history
    self-conditioning, retraction or force guidance is introduced. The output
    law approaches q_T as the fixed-step solver is refined. Finite-step CNF
    trace quadrature is not the exact Jacobian of this discrete solver.
    ``x0`` permits matched-prior comparisons and analytic validation.
    """
    if n_ode_steps < 1 or not math.isfinite(terminal_time) or not 0 < terminal_time <= 1:
        raise ValueError("Require positive steps and terminal_time in (0,1]")
    if not math.isfinite(prior_std) or prior_std <= 0:
        raise ValueError("prior_std must be positive and finite")
    if solver not in {'midpoint','rk4'}:raise ValueError('Unknown sampling solver')
    if solver=='rk4' and parameterization=='endpoint' and terminal_time==1:
        raise ValueError('RK4 endpoint-head evaluation requires terminal_time<1')
    reference = graph.ndata['x_1_true']
    if reference.dtype in (torch.float16, torch.bfloat16):
        reference = reference.float()
    if x0 is None:
        x = torch.randn(reference.shape, device=reference.device,
            dtype=reference.dtype, generator=generator)*prior_std
    else:
        if x0.shape != reference.shape or not torch.isfinite(x0).all():
            raise ValueError("x0 must be finite with the graph coordinate shape")
        x = x0.detach().to(reference).clone()
    dt = terminal_time/n_ode_steps
    with graph.local_scope(), deterministic_field(model.vector_field), \
            torch.autocast(device_type=x.device.type, enabled=False):
        for key in ('a', 'c'):
            graph.ndata[f'{key}_t'] = graph.ndata[f'{key}_1_true']
        graph.edata['e_t'] = graph.edata['e_1_true']
        x = center_by_graph(x, node_batch_idx, graph.batch_size)
        for step in range(n_ode_steps):
            t = x.new_full((graph.batch_size,), (step+0.5)*dt)
            if solver=='rk4':
                velocities=[]
                for stage,(offset,fraction) in enumerate([(0.,-.5),(.5,0.),(.5,0.),(1.,.5)]):
                    current=x if stage==0 else x+offset*dt*velocities[-1]
                    velocities.append(position_velocity(model,graph,current,t+fraction*dt,
                        node_batch_idx,upper_edge_mask,parameterization=parameterization,kT=kT))
                x=x+dt*(velocities[0]+2*velocities[1]+2*velocities[2]+velocities[3])/6
                continue
            v = position_velocity(model,graph,x,t,node_batch_idx,upper_edge_mask,
                parameterization=parameterization,kT=kT)
            midpoint = x+0.5*dt*v
            v_mid = position_velocity(model,graph,midpoint,t,node_batch_idx,upper_edge_mask,
                parameterization=parameterization,kT=kT)
            x = x+dt*v_mid
        if not torch.isfinite(x).all():
            raise FloatingPointError('Non-finite clamped-flow sample')
    return x
