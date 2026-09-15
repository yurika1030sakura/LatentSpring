"""Force-escorted local quench with explicit Jarzynski/importance work.

Conditional on anchor A, q0=N_H(A,sigma^2 I). The escort is translation by a
fixed centered displacement, so its intrinsic determinant is one. The target
is q0(Y|A) exp[-(E(Y)-E(A))/kT] on the declared same-graph support. This is a
local restrained ensemble, not the generator's global Boltzmann distribution.
"""
import math
import torch
from cfm_mol.chemical_moves import infer_chemical_graph


def graph_key(graph):
    return (tuple(graph['bond_orders'].flatten().tolist()),tuple(graph['formal_charges']),tuple(graph['radical_electrons']))


def local_work_log_weights(anchor,source,proposal,anchor_energy,proposal_energy,sigma,kT,valid):
    source2=(source-anchor[:,None]).square().sum((-1,-2))
    proposal2=(proposal-anchor[:,None]).square().sum((-1,-2))
    log_ratio=-(proposal2-source2)/(2*sigma[:,None].square())
    logw=-(proposal_energy-anchor_energy[:,None])/kT+log_ratio
    return torch.where(valid,logw,torch.full_like(logw,-torch.inf))


def escorted_candidates(anchor,force,*,kT,particles,max_sigma,max_shift,seed):
    force=force-force.mean(1,keepdim=True)
    norm=force.square().sum((1,2)).sqrt()
    sigma=torch.minimum(torch.full_like(norm,max_sigma),torch.sqrt(max_shift*kT/norm.clamp_min(1e-12)))
    shift=sigma[:,None,None].square()*force/kT
    noise=torch.randn((len(anchor),particles,*anchor.shape[1:]),dtype=anchor.dtype,generator=torch.Generator().manual_seed(seed))
    noise-=noise.mean(-2,keepdim=True)
    source=anchor[:,None]+sigma[:,None,None,None]*noise
    return source,source+shift[:,None],sigma,shift


def make_escorted_teacher(anchor,condition,oracle,*,kT,particles,max_sigma,max_shift,seed):
    if min(kT,max_sigma,max_shift)<=0 or particles<2:raise ValueError('Positive thermal parameters and at least2 particles required')
    anchor=anchor.double();n=len(anchor);ea,fa=oracle.evaluate_chunked(torch.cat([anchor,-anchor]),max_request=32)
    energy=(ea[:n]+ea[n:])/2;force=(fa[:n]-fa[n:])/2
    source,proposal,sigma,shift=escorted_candidates(anchor,force,kT=kT,particles=particles,max_sigma=max_sigma,max_shift=max_shift,seed=seed)
    keys=[graph_key(infer_chemical_graph(x,condition['numbers'],condition['charge'])) for x in anchor]
    valid=torch.zeros((n,particles),dtype=torch.bool)
    for i in range(n):
        for j in range(particles):
            try:valid[i,j]=graph_key(infer_chemical_graph(proposal[i,j],condition['numbers'],condition['charge']))==keys[i]
            except (ValueError,RuntimeError):pass
    y=proposal[valid];proposal_energy=torch.zeros((n,particles),dtype=anchor.dtype)
    if len(y):
        ey,fy=oracle.evaluate_chunked(torch.cat([y,-y]),max_request=32);count=len(y)
        proposal_energy[valid]=(ey[:count]+ey[count:])/2
    else:ey=torch.empty(0,dtype=anchor.dtype);fy=torch.empty((0,*anchor.shape[1:]),dtype=anchor.dtype)
    logw=local_work_log_weights(anchor,source,proposal,energy,proposal_energy,sigma,kT,valid)
    eligible=valid.any(1);weights=torch.zeros_like(logw)
    weights[eligible]=logw[eligible].softmax(-1)
    ess=torch.zeros(n,dtype=anchor.dtype);ess[eligible]=1/weights[eligible].square().sum(-1)
    uniform=valid.to(anchor.dtype)/valid.sum(-1).clamp_min(1)[:,None]
    return dict(anchor=anchor,source=source,proposal=proposal,sigma=sigma,shift=shift,valid=valid,eligible=eligible,weights=weights,uniform_weights=uniform,local_particle_ess=ess,log_work_weights=logw,work_eV=-kT*logw,
        anchor_raw_energy_eV=ea,anchor_raw_force_eV_A=fa,anchor_energy_eV=energy,anchor_force_eV_A=force,proposal_raw_energy_eV=ey,proposal_raw_force_eV_A=fy,proposal_energy_eV=proposal_energy,graph_keys=keys,
        configuration=dict(kT=kT,particles=particles,max_sigma=max_sigma,max_shift=max_shift,seed=seed),
        scope='Normalized finite-particle work weights approximate the local anchored energy tilt. ESS is local particle ESS only. Whole-generator density, equilibrium isomer weights and global Boltzmann sampling are not established.')
