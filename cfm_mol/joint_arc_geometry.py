"""Autoregressive two-root graph regrowth with normalized feasible-circle laws."""
import math
import torch

from cfm_mol.chemical_path_guide import exchanged_bond_graph
from cfm_mol.geodesic_arc_proposal import root_distance_constraints,draw_direction,direction_log_prob
from cfm_mol.joint_chemical_geometry import distinct_anchor_actions,radial_cartesian_log_density
from cfm_mol.local_site_guide import confinement_parameter
from cfm_mol.masked_angular_guide import masked_angular_context
from cfm_mol.normalized_site_guide import physical_site_parameter


def passive_geometry_supported(x,bonds,radii,leaf,*,margin=1e-8):
    """Qualify the fixed passive context before treating its angular law as physical."""
    indices=[i for i in range(len(x)) if i!=leaf]
    xx=x[indices];rr=radii[indices]
    distance=(xx[:,None]-xx[None,:]).norm(dim=2)
    sizes=rr[:,None]+rr[None,:];off=~torch.eye(len(indices),dtype=torch.bool,device=x.device)
    bonded=bonds[indices][:,indices]>0
    okay=torch.where(bonded,(distance>.6*sizes+margin)&(distance<1.25*sizes-margin),distance>1.25*sizes+margin)
    return bool(okay[off].all())


def arc_score_parameter(context,desired,numbers,electronic,root,kind,model,site_concentration,restraint):
    if kind=='arc_uniform':return context.new_zeros(3)
    roots=torch.tensor([root],dtype=torch.long,device=context.device)
    if kind=='arc_model':
        if model is None:raise ValueError('An explicit frozen score model is required')
        parameters,weights=model(context[None],desired[None],numbers,electronic,roots)
        if parameters.shape!=(1,1,3) or weights.shape!=(1,1):
            raise ValueError('This arc prototype accepts one vector score, not a mixture reinterpretation')
        return parameters[0,0]
    masked,radius,roles=masked_angular_context(context[None],roots)
    site=physical_site_parameter(masked,roles,desired[None],roots,site_concentration)[0]
    if kind=='arc_site':return site
    if kind=='arc_site_confinement':
        return site+confinement_parameter(masked,radius,electronic[2:3],restraint)[0]
    raise ValueError('Unknown arc score family')


@torch.no_grad()
def joint_arc_proposal(x,bonds,numbers,electronic,radii,action,*,kind,order,
                       generator=None,observed=None,model=None,radial_width=.05,
                       site_concentration=64.,restraint=.1,max_segment_width=math.pi/64,
                       margin=1e-8,chart_tolerance=1e-10):
    """Draw/evaluate a Cartesian-root conditional subdensity, including failures.

    Both radii precede the two angles. Each first context contains the other
    root's source-transported template. An invalid passive context, empty circle,
    or numerical chart produces one failed attempt, never conditional resampling.
    The constant COM chart volume cancels between source/target assignments.
    """
    if kind not in {'arc_uniform','arc_site','arc_site_confinement','arc_model'}:
        raise ValueError('Unknown arc proposal')
    if order not in (0,1) or action not in distinct_anchor_actions(numbers,bonds):
        raise ValueError('Eligible action and binary order required')
    i,j,k,l=action;desired=exchanged_bond_graph(bonds,action);roots=[(i,l),(j,k)]
    mean=torch.stack([(radii[leaf]+radii[anchor]).log() for leaf,anchor in roots])
    trace=dict(order=order,kind=kind,desired_bonds=desired,steps=[],failed=False)
    def failed(reason):
        trace.update(failed=True,failure=reason,log_coordinate_density=x.new_tensor(-torch.inf))
        return None,x.new_tensor(-torch.inf),trace
    if observed is None:
        noise=torch.randn(2,dtype=x.dtype,device=x.device,generator=generator)
        log_r=mean+radial_width*noise
    else:
        passive=[a for a in range(len(x)) if a not in (i,j)]
        if not torch.allclose(observed[passive]-observed[k],x[passive]-x[k],atol=1e-9,rtol=1e-9):
            return failed('Observed endpoint changes passive relative coordinates')
        log_r=torch.stack([(observed[leaf]-observed[anchor]).norm().log() for leaf,anchor in roots]);noise=None
    if not torch.isfinite(log_r).all():return failed('Nonfinite endpoint radii')
    radius=log_r.exp();template=x.clone()
    for n,(leaf,anchor) in enumerate(roots):
        partner=j if leaf==i else i;vector=x[partner]-x[anchor]
        template[leaf]=x[anchor]+radius[n]*vector/vector.norm()
    template-=template.mean(0)
    radial_log=radial_cartesian_log_density(log_r,mean,radial_width)
    trace.update(log_radii=log_r,radial_means=mean,radial_width=radial_width,radial_noise=noise,
        radial_log_densities=radial_log,template=template,max_segment_width=max_segment_width,
        distance_margin_A=margin,chart_tolerance=chart_tolerance)
    y=template.clone();logq=radial_log.sum()
    for n in ([0,1] if order==0 else [1,0]):
        leaf,anchor=roots[n];context=y.clone()
        if not passive_geometry_supported(context,desired,radii,leaf,margin=margin):
            return failed('Passive context fails the desired contact/overlap graph')
        base=context[leaf]-context[anchor];base/=base.norm()
        normals,limits=root_distance_constraints(context,roots[n],radius[n],radii,margin=margin)
        eta=arc_score_parameter(context,desired,numbers,electronic,roots[n],kind,model,site_concentration,restraint)
        if observed is None:
            direction,angular_log,random=draw_direction(base,normals,limits,eta,generator=generator,
                max_segment_width=max_segment_width,chart_tolerance=chart_tolerance)
        else:
            direction=(observed[leaf]-observed[anchor])/radius[n]
            angular_log,random=direction_log_prob(direction,base,normals,limits,eta,
                max_segment_width=max_segment_width,chart_tolerance=chart_tolerance)
        trace['steps'].append(dict(root=roots[n],context=context,base_direction=base,eta=eta,
            normals=normals,limits=limits,direction=direction,angular_log_density=angular_log,random=random))
        if direction is None or not torch.isfinite(angular_log):return failed('No supported angular draw or observed density')
        logq+=angular_log;y[leaf]=y[anchor]+radius[n]*direction;y-=y.mean(0)
    if observed is not None:torch.testing.assert_close(y,observed-observed.mean(0),atol=1e-9,rtol=1e-9)
    trace['log_coordinate_density']=logq
    return y,logq,trace


@torch.no_grad()
def marginal_joint_arc_proposal(x,bonds,numbers,electronic,radii,action,*,kind,order,
                                generator=None,observed=None,**kwargs):
    """Marginalize the fair decoder-order coin in both proposal directions.

    The selected order is sampled once. An unsuccessful draw stays a failure;
    evaluating the other order never turns that failure into resampling.
    """
    if order not in (0,1):raise ValueError('Binary decoder order required')
    inputs=(x,bonds,numbers,electronic,radii,action)
    logs=[None,None];components=[None,None]
    if observed is None:
        y,logs[order],components[order]=joint_arc_proposal(*inputs,kind=kind,order=order,
            generator=generator,**kwargs)
        if y is None:
            return None,x.new_tensor(-torch.inf),dict(order=order,order_marginalized=True,failed=True,
                failure=components[order]['failure'],components=components,log_coordinate_density=x.new_tensor(-torch.inf))
    else:y=observed
    for index in [0,1]:
        if logs[index] is None:
            _,logs[index],components[index]=joint_arc_proposal(*inputs,kind=kind,order=index,observed=y,**kwargs)
    logq=torch.logsumexp(torch.stack(logs),0)-math.log(2)
    trace=dict(order=order,order_marginalized=True,failed=not bool(torch.isfinite(logq)),
        order_log_densities=torch.stack(logs),components=components,log_coordinate_density=logq)
    if trace['failed']:
        trace['failure']='No density under either decoder order';return None,logq,trace
    return y,logq,trace
