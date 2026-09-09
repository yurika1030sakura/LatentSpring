"""Frozen molecular FM velocities as discrete stochastic path proposals.

This changes the sampling law. It does not assign work weights to archived ODE
samples. Energy evaluation may run later in envs/omol25: add -beta*U(x) to the
returned proposal_log_factor, then normalize within this fixed condition.
The auxiliary backward drift is deliberately simple (-v); poor ESS is possible.
"""
import math

import torch

from cfm_mol.clamped_density import deterministic_field, position_velocity
from cfm_mol.nonequilibrium import centered_orthonormal_basis, gaussian_path_sample


@torch.no_grad()
def sample_clamped_work_proposal(model, graph, node_batch_idx, upper_edge_mask, *,
        steps: int, terminal_time: float, noise_scale: float, generator: torch.Generator,
        parameterization: str, prior_std: float = 1.) -> dict:
    if steps < 1 or not math.isfinite(terminal_time) or not 0 < terminal_time <= 1:
        raise ValueError('Require positive steps and terminal_time in (0,1]')
    if parameterization == 'endpoint' and terminal_time == 1:
        raise ValueError('Backward endpoint-head evaluation requires terminal_time<1')
    if not math.isfinite(prior_std) or prior_std <= 0:
        raise ValueError('prior_std must be finite and positive')
    counts=graph.batch_num_nodes()
    if not torch.all(counts==counts[0]) or int(counts[0])<2:
        raise ValueError('A work cloud must share one fixed atom count >=2')
    batch=graph.batch_size;n=int(counts[0]);dimension=3*(n-1)
    for key in ['a','c']:
        labels=graph.ndata[f'{key}_1_true'].reshape(batch,n,-1)
        if not torch.equal(labels,labels[:1].expand_as(labels)):
            raise ValueError('A work cloud must share one ordered composition/charge condition')
    if graph.edata['e_1_true'].argmax(-1).any():
        raise ValueError('This branch is bond-free')
    reference=graph.ndata['x_1_true']
    basis=centered_orthonormal_basis(n,device=reference.device)
    x0=torch.randn((batch,dimension),dtype=torch.float64,device=reference.device,generator=generator)*prior_std
    def log_initial(z):
        return -.5*(z/prior_std).square().sum(-1)-dimension*math.log(prior_std*math.sqrt(2*math.pi))
    def velocity(z,t):
        cartesian=torch.einsum('nk,bkd->bnd',basis,z.reshape(batch,n-1,3))
        current=cartesian.reshape(batch*n,3).to(reference)
        times=current.new_full((batch,),terminal_time*t)
        v=position_velocity(model,graph,current,times,node_batch_idx,upper_edge_mask,
                            parameterization=parameterization)
        return terminal_time*torch.einsum('nk,bnd->bkd',basis,v.reshape(batch,n,3).double()).reshape(batch,dimension)
    with graph.local_scope(), deterministic_field(model.vector_field), torch.autocast(device_type=reference.device.type,enabled=False):
        for key in ['a','c']:graph.ndata[f'{key}_t']=graph.ndata[f'{key}_1_true']
        graph.edata['e_t']=graph.edata['e_1_true']
        # Constant terminal log value extracts a proposal factor only; it is
        # not a normalizable target, so do not interpret its summary as Z/ESS.
        paths=gaussian_path_sample(x0,log_initial,lambda z:z.new_zeros(len(z)),velocity,
            lambda z,t:-velocity(z,t),torch.linspace(0,1,steps+1,dtype=torch.float64),noise_scale,generator)
    positions=torch.einsum('nk,bkd->bnd',basis,paths.positions.reshape(batch,n-1,3))
    return {'positions':positions,'proposal_log_factor':paths.log_weights,
            'dimension':dimension,'steps':steps,'noise_scale':noise_scale,'prior_std':prior_std,
            'terminal_time':terminal_time,'parameterization':parameterization,
            'backward_kernel':'Gaussian with negative learned velocity; auxiliary, not exact time reversal',
            'scope':'proposal factors only; target energy and any declared restraint must still be applied'}
