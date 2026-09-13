"""Bounded topology-preserving optimization diagnostics, NOT a sampler."""
import math
import torch
from cfm_mol.nonequilibrium import centered_orthonormal_basis


def mobility_basis(n,roots,mode):
    if mode=='collective':return centered_orthonormal_basis(n)
    if mode!='roots' or len(set(roots))!=len(roots) or any(i<0 or i>=n for i in roots):
        raise ValueError('Valid distinct movable roots required')
    m=len(roots)
    if m==n:return centered_orthonormal_basis(n)
    if m<1:raise ValueError('Nonempty mobile set required')
    v=torch.eye(n,dtype=torch.float64)[:,list(roots)]-1/n
    # Symmetric inverse square root of I - 11^T/n, avoiding eigenvector choices.
    correction=((1-m/n)**-.5-1)/m
    return v@(torch.eye(m,dtype=torch.float64)+correction*torch.ones(m,m,dtype=torch.float64))


def gradient(state,basis,restraint):
    force=state['force_eV_A']-restraint*state['positions']
    g=-(basis.T@force).flatten()
    projected=-(basis@g.reshape(basis.shape[1],3))
    return g,float(projected.norm(dim=1).max())


def lbfgs_direction(g,history,initial_scale):
    q=g.clone();alpha=[]
    for s,y in reversed(history):
        rho=1/torch.dot(s,y);a=rho*torch.dot(s,q);q=q-a*y;alpha.append(a)
    scale=initial_scale
    if history:
        s,y=history[-1];scale=float((torch.dot(s,y)/torch.dot(y,y)).clamp(1e-5,.1))
    result=scale*q
    for (s,y),a in zip(history,reversed(alpha)):
        beta=torch.dot(y,result)/torch.dot(s,y);result=result+s*(a-beta)
    return -result


@torch.no_grad()
def relax_arms(target,pairs,options):
    """Batch Armijo trials across independent source/destination mobility arms.

    target.states must contain fresh initial energy/force states. All queried
    candidates remain in target.states/query_trace, including rejected trials.
    Geometry probes retain their positions/reasons and incur no oracle calls.
    """
    arms=[]
    for pair in pairs:
        for endpoint in ['source','destination']:
            initial_id=pair[endpoint+'_state_id'];initial=target.states[initial_id]
            for mode in ['roots','collective']:
                basis=mobility_basis(len(initial['positions']),pair['roots'],mode)
                g,residual=gradient(initial,basis,target.restraint)
                arms.append(dict(pair_id=pair['pair_id'],parent=pair['parent'],endpoint=endpoint,mobility=mode,
                    initial_state_id=initial_id,current_state_id=initial_id,best_state_id=initial_id,
                    basis=basis,origin=initial['positions'].clone(),frozen_bonds=initial['graph']['bond_orders'].clone(),
                    history=[],search=None,evaluations=0,geometry_probes=0,accepted_steps=0,
                    status='converged_initial' if residual<=options['force_tolerance_eV_A'] else 'active',events=[],checkpoints=[]))
    round_index=0
    while any(a['status']=='active' for a in arms):
        pending=[]
        for arm in arms:
            if arm['status']!='active':continue
            current=target.states[arm['current_state_id']];basis=arm['basis']
            if arm['search'] is None:
                g,residual=gradient(current,basis,target.restraint)
                if residual<=options['force_tolerance_eV_A']:
                    arm['status']='converged';continue
                direction=lbfgs_direction(g,arm['history'],options['initial_inverse_hessian_A2_eV'])
                if not torch.isfinite(direction).all():raise ValueError('Nonfinite L-BFGS direction')
                reset=False
                if float(torch.dot(g,direction))>=0:
                    arm['history']=[];direction=-options['initial_inverse_hessian_A2_eV']*g;reset=True
                cart=basis@direction.reshape(basis.shape[1],3);maximum=float(cart.norm(dim=1).max())
                if maximum>options['max_atom_step_A']:direction*=options['max_atom_step_A']/maximum
                arm['search']=dict(origin_state_id=arm['current_state_id'],z=(basis.T@(current['positions']-arm['origin'])).flatten(),
                    gradient=g,direction=direction,alpha=1.,backtracks=0,history_reset=reset)
            search=arm['search']
            while True:
                if search['backtracks']>=options['max_backtracks']:
                    arm['status']='line_search_blocked';break
                change=basis@(search['alpha']*search['direction']).reshape(basis.shape[1],3)
                if float(change.norm(dim=1).max())<options['minimum_atom_step_A']:
                    arm['status']='step_too_small';break
                trial=arm['origin']+basis@(search['z']+search['alpha']*search['direction']).reshape(basis.shape[1],3)
                trial-=trial.mean(0)
                event=dict(round=round_index,origin_state_id=search['origin_state_id'],positions=trial.clone(),alpha=search['alpha'],
                    backtracks=search['backtracks'],geometry_supported=False,queried=False,accepted=False)
                arm['geometry_probes']+=1
                try:
                    candidate=target.coordinate_state(trial)
                    if not torch.equal(candidate['graph']['bond_orders'],arm['frozen_bonds']):
                        raise ValueError('Perceived graph differs from frozen endpoint graph')
                except ValueError as exc:
                    event.update(failure_reason=str(exc),exception_type=type(exc).__name__);arm['events'].append(event)
                    search['alpha']*=.5;search['backtracks']+=1;continue
                event['geometry_supported']=True;pending.append((arm,candidate,event));break
        if not pending:continue
        before=target.oracle.evaluated
        target.evaluate([c for arm,c,event in pending],phase=f'mobility_trial_{round_index}')
        assert target.oracle.evaluated-before==2*len(pending)
        for arm,candidate,event in pending:
            arm['evaluations']+=1;search=arm['search'];origin=target.states[search['origin_state_id']]
            threshold=float(origin['potential_eV'])+options['armijo_c1']*search['alpha']*float(torch.dot(search['gradient'],search['direction']))+options['energy_noise_tolerance_eV']
            accepted=float(candidate['potential_eV'])<=threshold
            event.update(queried=True,state_id=candidate['state_id'],armijo_threshold_eV=threshold,accepted=accepted,
                         evaluation=arm['evaluations']);arm['events'].append(event)
            if float(candidate['potential_eV'])<float(target.states[arm['best_state_id']]['potential_eV']):arm['best_state_id']=candidate['state_id']
            if accepted:
                basis=arm['basis'];new_g,residual=gradient(candidate,basis,target.restraint)
                step=(basis.T@(candidate['positions']-origin['positions'])).flatten();change=new_g-search['gradient']
                curvature=float(torch.dot(step,change));scale=float(step.norm()*change.norm())
                if curvature>max(1e-14,options['curvature_relative_floor']*scale):
                    arm['history'].append((step,change));arm['history']=arm['history'][-options['history_size']:]
                arm['current_state_id']=candidate['state_id'];arm['search']=None;arm['accepted_steps']+=1
                if residual<=options['force_tolerance_eV_A']:arm['status']='converged'
            else:search['alpha']*=.5;search['backtracks']+=1
            if arm['evaluations'] in options['readouts']:
                arm['checkpoints'].append(dict(evaluations=arm['evaluations'],best_state_id=arm['best_state_id'],current_state_id=arm['current_state_id']))
            if arm['evaluations']>=options['max_evaluations'] and arm['status']=='active':arm['status']='budget_exhausted'
        round_index+=1
    for arm in arms:
        best=target.states[arm['best_state_id']];current=target.states[arm['current_state_id']]
        arm['best_projected_force_max_eV_A']=gradient(best,arm['basis'],target.restraint)[1]
        arm['final_projected_force_max_eV_A']=gradient(current,arm['basis'],target.restraint)[1]
        arm['best_potential_eV']=float(best['potential_eV']);arm['final_potential_eV']=float(current['potential_eV'])
        arm['initial_potential_eV']=float(target.states[arm['initial_state_id']]['potential_eV'])
    return arms
