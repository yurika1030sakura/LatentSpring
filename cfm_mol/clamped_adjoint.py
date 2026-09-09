"""Exact first parameter derivative of the fixed-step clamped density objective.

Store forward coordinate states and trace probes. Backward recomputes one step
at a time and propagates the discrete adjoint, including divergence and Gaussian
prior contributions. It does not reconstruct an unstable trajectory in reverse,
and it does not freeze trajectory derivatives. Higher parameter derivatives and
coordinate/temperature input derivatives are deliberately unsupported.
"""
import math
import torch
from torch.autograd.function import once_differentiable
from .clamped_density import center_by_graph,_midpoint_step,_rk4_step


class _DensityAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx,model,graph,node_batch_idx,upper_edge_mask,options,*parameters):
        graph=graph.clone()
        for key in ('x_1_true','a_1_true','c_1_true'):
            graph.ndata[key]=graph.ndata[key].detach().clone()
        graph.edata['e_1_true']=graph.edata['e_1_true'].detach().clone()
        x=graph.ndata['x_1_true']
        if x.dtype in (torch.float16,torch.bfloat16):x=x.float()
        if not torch.isfinite(x).all():raise ValueError('Non-finite input coordinates')
        x=center_by_graph(x,node_batch_idx,graph.batch_size).detach()
        dt=options['terminal_time']/options['n_ode_steps']
        integral=x.new_zeros(options['n_trace_replicates'],graph.batch_size)
        states=[];probes=[]
        with torch.enable_grad(),torch.autocast(device_type=x.device.type,enabled=False):
            for step in range(options['n_ode_steps']):
                states.append(x.detach())
                cache=[];probes.append(cache)
                time=x.new_full((graph.batch_size,),options['terminal_time']-(step+.5)*dt)
                step_function=_midpoint_step if options['solver']=='midpoint' else _rk4_step
                following,divergence=step_function(model,graph,x.detach().requires_grad_(True),
                    time,node_batch_idx,upper_edge_mask,dt=dt,
                    n_hutchinson=options['n_hutchinson'],n_trace_replicates=options['n_trace_replicates'],
                    parameterization=options['parameterization'],kT=options['kT'],xi_fn=options['xi_fn'],
                    step_index=step,probe_cache=cache,for_training=False)
                x=following.detach();integral=integral+dt*divergence.detach()
            counts=torch.bincount(node_batch_idx,minlength=graph.batch_size).to(x)
            sq=x.new_zeros(graph.batch_size).index_add(0,node_batch_idx,x.square().sum(-1))
            sigma=options['prior_std']
            prior=-.5*sq/sigma**2-1.5*(counts-1)*math.log(2*math.pi*sigma**2)
            result=prior[None,:]-integral
        if not torch.isfinite(result).all():
            raise FloatingPointError('Non-finite clamped-flow density in discrete adjoint')
        ctx.model=model;ctx.graph=graph;ctx.nbi=node_batch_idx;ctx.uem=upper_edge_mask
        ctx.options=dict(options);ctx.states=states;ctx.probes=probes;ctx.final_state=x
        ctx.save_for_backward(*parameters)  # checks parameter versions before backward
        return result.detach()

    @staticmethod
    @once_differentiable
    def backward(ctx,grad_output):
        options=ctx.options;parameters=ctx.saved_tensors
        dt=options['terminal_time']/options['n_ode_steps']
        # Every trace replica has the same exact Gaussian prior contribution.
        adjoint=-ctx.final_state/options['prior_std']**2*grad_output.sum(0)[ctx.nbi,None]
        totals=[None]*len(parameters)
        with torch.enable_grad(),torch.autocast(device_type=adjoint.device.type,enabled=False):
            for step in reversed(range(options['n_ode_steps'])):
                state=ctx.states[step].detach().requires_grad_(True)
                time=state.new_full((ctx.graph.batch_size,),options['terminal_time']-(step+.5)*dt)
                step_function=_midpoint_step if options['solver']=='midpoint' else _rk4_step
                following,divergence=step_function(ctx.model,ctx.graph,state,time,ctx.nbi,ctx.uem,
                    dt=dt,n_hutchinson=options['n_hutchinson'],
                    n_trace_replicates=options['n_trace_replicates'],
                    parameterization=options['parameterization'],kT=options['kT'],
                    xi_fn=None,step_index=step,probe_cache=ctx.probes[step],for_training=True)
                outputs=[following];weights=[adjoint]
                if divergence.requires_grad:
                    outputs.append(divergence);weights.append(-dt*grad_output)
                derivatives=torch.autograd.grad(outputs,(state,*parameters),
                    grad_outputs=weights,allow_unused=True)
                adjoint=derivatives[0].detach()
                for index,derivative in enumerate(derivatives[1:]):
                    if derivative is not None:
                        derivative=derivative.detach()
                        totals[index]=derivative if totals[index] is None else totals[index]+derivative
        # Non-tensor arguments and fixed graph labels have no input derivative.
        return (None,None,None,None,None,*totals)


def log_density_discrete_adjoint(model,graph,node_batch_idx,upper_edge_mask,**options):
    if isinstance(options.get('kT'),torch.Tensor) and options['kT'].requires_grad:
        raise ValueError('Discrete adjoint supports model-parameter gradients, not temperature gradients')
    owner=model if isinstance(model,torch.nn.Module) else model.vector_field
    parameters=tuple(parameter for parameter in owner.parameters() if parameter.requires_grad)
    value=_DensityAdjoint.apply(model,graph,node_batch_idx,upper_edge_mask,options,*parameters)
    return value[0] if options['n_trace_replicates']==1 else value
