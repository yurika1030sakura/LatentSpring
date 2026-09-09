"""Evaluation reference: adaptive position solve, independent trace quadrature.

The adaptive position grid is independent of trace probes. DOP853 dense output
is integrated with fixed Gauss-Legendre rules on each accepted time interval.
Comparing tolerances AND quadrature orders checks two distinct error sources.
This is evaluation-only; use the differentiable solver for training.
"""
import math
import time
import numpy as np
from scipy.integrate import solve_ivp
import torch

from .clamped_density import center_by_graph, deterministic_field, position_velocity, _trace


class ReferenceBudgetExceeded(RuntimeError):
    """A recorded convergence failure, not permission to use partial densities."""


def log_density_clamped_reference(model, graph, node_batch_idx, upper_edge_mask,
        *, terminal_time=0.95, prior_std=1., rtol=1e-5, atol=1e-7,
        quadrature_orders=(2,4), n_replicates=8, seed=9003, max_nfe=10000,
        parameterization='endpoint'):
    if not 0 < terminal_time <= 1 or not math.isfinite(prior_std) or prior_std <= 0:
        raise ValueError('Require T in (0,1] and positive finite prior std')
    if terminal_time==1 and parameterization=='endpoint':
        raise ValueError('Adaptive endpoint evaluation requires T<1')
    if n_replicates < 1 or any(order < 1 for order in quadrature_orders):
        raise ValueError('Replicates and quadrature orders must be positive')
    x_template = graph.ndata['x_1_true']
    if x_template.dtype not in (torch.float32,torch.float64):
        raise ValueError('Reference solver requires float32 or float64 model inputs')
    n_graphs = graph.batch_size
    n_atoms = torch.bincount(node_batch_idx,minlength=n_graphs).double()
    start = time.monotonic()
    nfe = 0
    def state_tensor(array):
        return torch.as_tensor(np.array(array,copy=True),device=x_template.device,
                               dtype=x_template.dtype).reshape_as(x_template)
    with graph.local_scope(), deterministic_field(model.vector_field), torch.no_grad():
        for key in ('a','c'):
            graph.ndata[f'{key}_t'] = graph.ndata[f'{key}_1_true']
        graph.edata['e_t'] = graph.edata['e_1_true']
        terminal = center_by_graph(x_template,node_batch_idx,n_graphs)
        def rhs(t,y):
            nonlocal nfe
            nfe += 1
            if nfe > max_nfe:
                raise ReferenceBudgetExceeded(f'Reference position solve exceeded {max_nfe} evaluations')
            x = state_tensor(y)
            velocity = position_velocity(model,graph,x,x.new_full((n_graphs,),t),
                                          node_batch_idx,upper_edge_mask,parameterization=parameterization)
            if not torch.isfinite(velocity).all():
                raise FloatingPointError('Non-finite reference velocity')
            return velocity.double().cpu().numpy().reshape(-1)
        solution = solve_ivp(rhs,(terminal_time,0.),terminal.double().cpu().numpy().reshape(-1),
                             method='DOP853',rtol=rtol,atol=atol,dense_output=True)
        if not solution.success:
            raise RuntimeError(solution.message)
        position_seconds = time.monotonic()-start
        prior_x = state_tensor(solution.y[:,-1]).double()
        prior_x = center_by_graph(prior_x,node_batch_idx,n_graphs)
        sq = prior_x.new_zeros(n_graphs).index_add(0,node_batch_idx,prior_x.square().sum(-1))
        prior = -0.5*sq/prior_std**2-1.5*(n_atoms-1)*math.log(2*math.pi*prior_std**2)
        # CPU probe generation matches checkpoint_panel across CPU/GPU and
        # float32/float64, rather than assuming identical RNGs on each device.
        generator = torch.Generator().manual_seed(seed)
        probes = [(2*torch.randint(0,2,x_template.shape,generator=generator)-1).to(x_template)
                  for _ in range(n_replicates)]
        estimates = {}
        for order in quadrature_orders:
            q_start = time.monotonic()
            nodes,weights = np.polynomial.legendre.leggauss(order)
            integral = prior.new_zeros((n_replicates,n_graphs))
            for high,low in zip(solution.t[:-1],solution.t[1:]):
                half=(high-low)/2; middle=(high+low)/2
                for node,weight in zip(nodes,weights):
                    t=middle+half*node
                    with torch.enable_grad():
                        x=state_tensor(solution.sol(t)).requires_grad_(True)
                        velocity=position_velocity(model,graph,x,x.new_full((n_graphs,),t),
                                                   node_batch_idx,upper_edge_mask,parameterization=parameterization)
                        traces=[_trace(velocity,x,node_batch_idx,n_graphs,[probe],False) for probe in probes]
                    integral += half*weight*torch.stack(traces).double()
            value=prior[None,:]-integral
            if not torch.isfinite(value).all():
                raise FloatingPointError('Non-finite reference log density')
            estimates[str(order)]={'log_q':value.cpu().tolist(),
                'seconds':time.monotonic()-q_start,'trace_evaluations':order*(len(solution.t)-1)}
    return {'estimates':estimates,'rtol':rtol,'atol':atol,'dtype':str(x_template.dtype),
        'nfe':nfe,'accepted_intervals':len(solution.t)-1,'prior_log_q':prior.cpu().tolist(),
        'position_seconds':position_seconds,'total_seconds':time.monotonic()-start,
        'probe_seed':seed,'n_replicates':n_replicates,
        'scope':'adaptive deterministic coordinate trajectory; fixed-probe trace quadrature; no training gradient'}
