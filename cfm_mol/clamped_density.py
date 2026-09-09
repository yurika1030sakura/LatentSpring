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
    try:
        vector_field.eval()
        yield
    finally:
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
    if parameterization not in {"endpoint", "velocity"}:
        raise ValueError("parameterization must be endpoint or velocity")
    n_graphs = graph.batch_size
    x_centered = center_by_graph(x, node_batch_idx, n_graphs)
    graph.ndata["x_t"] = x_centered
    kwargs = dict(node_batch_idx=node_batch_idx, upper_edge_mask=upper_edge_mask)
    if kT is not None:
        kwargs["kT"] = kT
    prediction = model.vector_field(graph, t, **kwargs)["x"]
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


def log_density_clamped_flow(model, graph, node_batch_idx, upper_edge_mask,
                             *, n_ode_steps=32, n_hutchinson=2, prior_std=1.0,
                             terminal_time=0.95, for_training=False,
                             parameterization="endpoint", kT=None, xi_fn=None,
                             n_trace_replicates=1):
    """Log q_T(x), explicit midpoint integration of state AND log-Jacobian.

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
    """
    if not isinstance(n_trace_replicates, int) or n_trace_replicates < 1:
        raise ValueError("n_trace_replicates must be a positive integer")
    if n_ode_steps < 1 or n_hutchinson < 0:
        raise ValueError("Require n_ode_steps >= 1 and n_hutchinson >= 0")
    if not math.isfinite(terminal_time) or not 0 < terminal_time <= 1:
        raise ValueError("terminal_time must be in (0, 1]")
    if not math.isfinite(prior_std) or prior_std <= 0:
        raise ValueError("prior_std must be positive and finite")
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
        for step in range(n_ode_steps):
            # A predictor at an interior time avoids evaluating a learned
            # endpoint head exactly at its singular t=1. With the same midpoint
            # time in both stages this is second-order for smooth v(x,t).
            t_mid = x.new_full((n_graphs,), terminal_time-(step+0.5)*dt)
            v_start = position_velocity(model, graph, x, t_mid, node_batch_idx,
                upper_edge_mask, parameterization=parameterization, kT=kT)
            x_mid = x - 0.5*dt*v_start
            if not for_training:
                x_mid = x_mid.detach().requires_grad_(True)
            v_mid = position_velocity(model, graph, x_mid, t_mid, node_batch_idx,
                upper_edge_mask, parameterization=parameterization, kT=kT)
            traces = []
            for replica in range(n_trace_replicates):
                probes = None if n_hutchinson == 0 else [
                    (xi_fn(step, replica*n_hutchinson+k, x_mid) if xi_fn is not None else
                     torch.randint(0, 2, x_mid.shape, device=x.device).to(x)*2-1)
                    for k in range(n_hutchinson)]
                if probes is not None:
                    if any(probe.shape != x.shape for probe in probes):
                        raise ValueError("Trace probe shape must match coordinates")
                    probes = [probe.to(x).detach() for probe in probes]
                traces.append(_trace(v_mid, x_mid, node_batch_idx, n_graphs, probes, for_training))
            divergence = torch.stack(traces)
            integral = integral + dt*divergence
            x = x - dt*v_mid
            if not for_training:
                x = x.detach().requires_grad_(True)
                integral = integral.detach()
        sq = x.new_zeros(n_graphs).index_add(0, node_batch_idx, x.square().sum(-1))
        logp = -0.5*sq/prior_std**2 - 1.5*(n_atoms-1)*math.log(2*math.pi*prior_std**2) - integral
        if not torch.isfinite(logp).all():
            raise FloatingPointError("Non-finite clamped-flow density; inspect field and resolution")
        result = logp[0] if n_trace_replicates == 1 else logp
        return result if for_training else result.detach()
