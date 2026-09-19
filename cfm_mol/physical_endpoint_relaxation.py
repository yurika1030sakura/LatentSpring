"""Bounded graph-preserving Cartesian relaxation for TRAINING endpoints only."""
import torch
from .mobility_relaxation import lbfgs_direction


def relax_endpoint(positions,evaluate,allowed,*,max_steps=96,max_evaluations=160,
        force_tolerance_eV_A=.1,max_atom_step_A=.05,initial_inverse_hessian_A2_eV=.02,
        history_size=10,backtracking_steps=10,armijo=1e-4,energy_tolerance_eV=2e-5):
    """Return the last accepted state, retaining all queries and rejection records.

    The map is defined even when its budget ends before stationarity: return the
    last finite, graph-preserving accepted point (possibly the original point).
    Such endpoints are NOT silently called converged minima. No basin weights
    are inferred here. The caller owns the empirical distribution of references.
    """
    if min(max_steps,max_evaluations,history_size,backtracking_steps)<1:
        raise ValueError('Positive optimization budgets required')
    x=torch.as_tensor(positions,dtype=torch.float64).clone();x-=x.mean(0)
    if x.ndim!=2 or x.shape[1]!=3 or not torch.isfinite(x).all() or not allowed(x):
        raise ValueError('Invalid initial endpoint')
    queries=[];events=[];history=[];accepted=[];probes=0
    def query(y):
        energy,force=evaluate(y)
        force=torch.as_tensor(force,dtype=torch.float64)
        energy=float(energy)
        if force.shape!=y.shape or not torch.isfinite(force).all() or not torch.isfinite(torch.tensor(energy)):
            raise FloatingPointError('Nonfinite energy/force; no endpoint is substituted')
        row=dict(positions=y.clone(),energy_eV=energy,force_eV_A=force.clone())
        queries.append(row);return row
    current=query(x);current_index=0
    status='step_limit'
    for step in range(max_steps):
        if float(current['force_eV_A'].norm(dim=-1).max())<=force_tolerance_eV_A:
            status='converged';break
        if len(queries)>=max_evaluations:status='query_limit';break
        force=current['force_eV_A'];g=-(force-force.mean(0)).flatten()
        direction=lbfgs_direction(g,history,initial_inverse_hessian_A2_eV)
        if not torch.isfinite(direction).all():raise FloatingPointError('Nonfinite optimization direction')
        if float(g@direction)>=0:history=[];direction=-initial_inverse_hessian_A2_eV*g
        cart=direction.reshape_as(x);cart-=cart.mean(0)
        maximum=float(cart.norm(dim=-1).max())
        if maximum>max_atom_step_A:cart*=max_atom_step_A/maximum
        directional=float(g@cart.flatten());moved=False
        for trial in range(backtracking_steps):
            if len(queries)>=max_evaluations:status='query_limit';break
            scale=2.**(-trial);y=x+scale*cart;y-=y.mean(0);probes+=1
            if not allowed(y):
                events.append(dict(step=step,trial=trial,accepted=False,reason='graph_or_geometry',positions=y.clone()));continue
            candidate=query(y);index=len(queries)-1
            ok=candidate['energy_eV']<=current['energy_eV']+armijo*scale*directional+energy_tolerance_eV
            events.append(dict(step=step,trial=trial,query_index=index,accepted=bool(ok),reason='armijo'))
            if not ok:continue
            displacement=(y-x).flatten();new_force=candidate['force_eV_A']
            new_g=-(new_force-new_force.mean(0)).flatten();difference=new_g-g
            if float(displacement@difference)>1e-10*float(displacement.norm()*difference.norm()):
                history.append((displacement.clone(),difference.clone()));history=history[-history_size:]
            x=y;current=candidate;current_index=index;accepted.append(index);moved=True;break
        if not moved:
            if status!='query_limit':status='line_search_stalled'
            break
    converged=float(current['force_eV_A'].norm(dim=-1).max())<=force_tolerance_eV_A
    if converged:status='converged'
    return dict(initial=queries[0],final=current,final_query_index=current_index,
        accepted_query_indices=accepted,accepted_steps=len(accepted),queries=queries,
        events=events,geometry_probes=probes,evaluations=len(queries),status=status,
        converged=converged,all_coordinates_movable=True)
