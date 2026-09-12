"""Normalized graph/radius/direction proposals on labelled COM coordinates.

Two terminal leaves exchange anchors. Their radii and directions are sampled,
not deterministically transported. Densities use Cartesian conditional volume;
the constant passive-coordinate COM chart factor cancels in MH. Graph failures
are single-draw self transitions, never support-conditioned resampling.
"""
import math
import torch
from cfm_mol.angular_envelope import envelope_parameters, envelope_draw, logcosh
from cfm_mol.chemical_moves import terminal_exchange_actions, exchange_terminal_sites
from cfm_mol.chemical_path_guide import exchanged_bond_graph
from cfm_mol.normalized_site_guide import mixture_draw, mixture_log_prob


def distinct_anchor_actions(numbers, bonds):
    return [a for a in terminal_exchange_actions(numbers, bonds) if a[2] != a[3]]


def radial_cartesian_log_density(log_radius, mean, width):
    """Normal density in log r divided by r^3, per root and unit solid angle."""
    if width <= 0 or not math.isfinite(width):
        raise ValueError('Positive finite log-radius width required')
    return -.5*((log_radius-mean)/width)**2-math.log(width*math.sqrt(2*math.pi))-3*log_radius


def normalized_direction_log_prob(direction, eta, matrix):
    """Density of the normalized two-vMF mixture, NOT Fisher--Bingham density."""
    p = envelope_parameters(eta, matrix)
    return (eta*direction).sum(-1)+logcosh(p['gap']*(p['axis']*direction).sum(-1))-p['log_base_partition']


def direction_parameters(x, bonds, numbers, electronic, root, *, kind, model, site_concentration):
    if kind in ('vector', 'tensor'):
        if model is None:
            raise ValueError('Frozen guide required')
        eta, matrix, _ = model(x[None], bonds[None], numbers, electronic,
                              torch.tensor([root], dtype=torch.long, device=x.device))
        return eta, matrix
    eta = x.new_zeros(1, 3)
    if kind == 'site':
        leaf, anchor = root
        mask = bonds[anchor] > 0
        mask[leaf] = False
        vectors = x[mask]-x[anchor]
        away = -(vectors/vectors.norm(dim=1, keepdim=True).clamp_min(1e-12)).sum(0)
        eta[0] = site_concentration*away/away.norm().clamp_min(1e-12)
    elif kind != 'uniform':
        raise ValueError('Unknown normalized directional decoder')
    return eta, x.new_zeros(1, 3, 3)


@torch.no_grad()
def joint_geometry_proposal(x, bonds, numbers, electronic, radii, action, *, kind,
                            order, generator=None, observed=None, model=None,
                            radial_width=.05, site_concentration=10.):
    """Draw, or evaluate an observed endpoint using the same sequential decoder.

    order is an augmented fair coin, retained under reversal. The first decoder
    sees the second leaf's SOURCE-TRANSPORTED template, not its eventual output.
    Both radii are sampled before either conditional angle. All passives retain
    their relative coordinates; arbitrary translations are removed by centering.
    """
    if order not in (0, 1) or action not in distinct_anchor_actions(numbers, bonds):
        raise ValueError('Eligible different-anchor action and binary order required')
    i, j, k, l = action
    desired = exchanged_bond_graph(bonds, action)
    roots = [(i, l), (j, k)]
    mean = torch.stack([(radii[leaf]+radii[anchor]).log() for leaf, anchor in roots])
    if observed is None:
        noise = torch.randn(2, dtype=x.dtype, device=x.device, generator=generator)
        log_r = mean+radial_width*noise
    else:
        passive = [a for a in range(len(x)) if a not in (i, j)]
        if not torch.allclose(observed[passive]-observed[k], x[passive]-x[k], atol=1e-9, rtol=1e-9):
            raise ValueError('Endpoint changed the passive relative coordinates')
        log_r = torch.stack([(observed[leaf]-observed[anchor]).norm().log() for leaf, anchor in roots])
        noise = None
    if not torch.isfinite(log_r).all():
        raise ValueError('Finite positive endpoint radii required')
    radius = log_r.exp()
    # Source directions are transported from the partner at the NEW anchor.
    template = x.clone()
    for n, (leaf, anchor) in enumerate(roots):
        partner = j if leaf == i else i
        vector = x[partner]-x[anchor]
        template[leaf] = x[anchor]+radius[n]*vector/vector.norm()
    template -= template.mean(0)
    y = template.clone()
    radial_log = radial_cartesian_log_density(log_r, mean, radial_width)
    log_q = radial_log.sum()
    steps = []
    for n in ([0, 1] if order == 0 else [1, 0]):
        leaf, anchor = roots[n]
        context = y.clone()
        if kind == 'normalized_site':
            if model is None:
                raise ValueError('Normalized directional model required')
            parameters, log_weights = model(context[None], desired[None], numbers, electronic,
                torch.tensor([roots[n]], dtype=torch.long, device=x.device))
        else:
            eta, matrix = direction_parameters(context, desired, numbers, electronic, roots[n],
                kind=kind, model=model, site_concentration=site_concentration)
        if observed is None:
            if kind == 'normalized_site':
                direction, random = mixture_draw(parameters, log_weights, generator=generator)
            else:
                direction, _, random = envelope_draw(eta, matrix, generator=generator)
            # NO rejection-to-score step: the full-sphere mixture is our q.
            direction = direction[0]
        else:
            direction = (observed[leaf]-observed[anchor])/radius[n]
            random = None
        angular_log = (mixture_log_prob(direction[None], parameters, log_weights)[0] if kind == 'normalized_site'
            else normalized_direction_log_prob(direction[None], eta, matrix)[0])
        log_q += angular_log
        y[leaf] = y[anchor]+radius[n]*direction
        y -= y.mean(0)
        record = dict(root=roots[n], context=context, direction=direction, angular_log_density=angular_log, random=random)
        if kind == 'normalized_site':
            record.update(parameters=parameters[0], log_weights=log_weights[0])
        else:
            record.update(eta=eta[0], matrix=matrix[0])
        steps.append(record)
    if observed is not None:
        torch.testing.assert_close(y, observed-observed.mean(0), atol=1e-9, rtol=1e-9)
    return y, log_q, dict(order=order, log_radii=log_r, radial_means=mean,
        radial_width=radial_width, radial_noise=noise, radial_log_densities=radial_log,
        template=template, steps=steps, desired_bonds=desired, log_coordinate_density=log_q)


@torch.no_grad()
def defensive_joint_proposal(x, bonds, numbers, electronic, radii, action, *, kind='defensive_site',
                            order, generator=None, observed=None, model=None,
                            radial_width=.05, site_concentration=10., physical_weight=.5):
    """Marginal mixture of WHOLE joint proposals; evaluate both components.

    For fixed weight w, the corrected probability flow dominates w times that
    of the physical-site proposal. This is a standard mixture property, not a
    finite-time convergence certificate or novelty claim.
    """
    if kind != 'defensive_site' or not 0 < physical_weight < 1:
        raise ValueError('Defensive joint mixture requires both components')
    kwargs = dict(order=order, model=model, radial_width=radial_width, site_concentration=site_concentration)
    inputs = (x, bonds, numbers, electronic, radii, action)
    component = None
    choice_uniform = None
    densities, traces = [None, None], [None, None]
    kinds = ['site', 'normalized_site']
    if observed is None:
        choice_uniform = torch.rand((), dtype=x.dtype, device=x.device, generator=generator)
        component = int(float(choice_uniform) >= physical_weight)
        y, densities[component], traces[component] = joint_geometry_proposal(*inputs,
            kind=kinds[component], generator=generator, **kwargs)
    else:
        y = observed
    for index in range(2):
        if index != component:
            _, densities[index], traces[index] = joint_geometry_proposal(*inputs,
                kind=kinds[index], observed=y, **kwargs)
    component_logs = torch.stack(densities)
    log_weights = x.new_tensor([physical_weight, 1-physical_weight]).log()
    log_q = torch.logsumexp(component_logs+log_weights, 0)
    return y, log_q, dict(component=component, choice_uniform=choice_uniform,
        physical_weight=physical_weight, component_log_densities=component_logs, components=traces,
        log_coordinate_density=log_q)


@torch.no_grad()
def joint_chemical_transition(target, states, *, kind, generator, phase,
                              model=None, radial_width=.05, site_concentration=10.):
    """Matched restricted deterministic or normalized joint-geometry MH move."""
    numbers = torch.tensor(target.numbers, dtype=torch.long)
    electronic = torch.tensor([target.condition['charge'], target.condition['spin_multiplicity'], target.kT], dtype=torch.float64)
    rows, candidates = [], []
    proposal = defensive_joint_proposal if kind == 'defensive_site' else joint_geometry_proposal
    for old in states:
        actions = distinct_anchor_actions(numbers, old['graph']['bond_orders'])
        row = dict(kind='joint_exchange', decoder=kind, phase=phase,
            old_state_id=old['state_id'], new_state_id=-1, forward_count=len(actions),
            valid=False, accepted=False)
        candidate = None
        if actions:
            index = int(torch.randint(len(actions), (1,), generator=generator))
            action = actions[index]
            i, j, k, l = action
            inverse = (i, j, l, k)
            desired = exchanged_bond_graph(old['graph']['bond_orders'], action)
            row.update(choice_index=index, action=action, inverse_action=inverse)
            if kind == 'deterministic':
                y, log_volume, _ = exchange_terminal_sites(old['positions'], target.radii, action)
                row.update(log_volume=log_volume, coordinate_log_ratio=log_volume)
            else:
                order = int(torch.randint(2, (1,), generator=generator))
                y, log_q, forward = proposal(old['positions'], old['graph']['bond_orders'],
                    numbers, electronic, target.radii, action, kind=kind, order=order, generator=generator,
                    model=model, radial_width=radial_width, site_concentration=site_concentration)
                row.update(order=order, forward=forward, log_forward_coordinate=log_q)
            row['proposal_positions'] = y
            try:
                candidate = target.coordinate_state(y)
                if not torch.equal(candidate['graph']['bond_orders'], desired):
                    raise ValueError('Endpoint differs from desired exchanged bond graph')
                reverse_actions = distinct_anchor_actions(numbers, candidate['graph']['bond_orders'])
                if inverse not in reverse_actions:
                    raise ValueError('Inverse graph action ineligible')
                if kind != 'deterministic':
                    _, reverse_q, reverse = proposal(y, desired, numbers, electronic,
                        target.radii, inverse, kind=kind, order=order, observed=old['positions'],
                        model=model, radial_width=radial_width, site_concentration=site_concentration)
                    row.update(reverse=reverse, log_reverse_coordinate=reverse_q,
                        coordinate_log_ratio=reverse_q-log_q)
                row.update(valid=True, reverse_count=len(reverse_actions),
                    action_log_ratio=math.log(len(actions)/len(reverse_actions)))
            except ValueError as exc:
                candidate = None
                row['rejection_reason'] = str(exc)
        else:
            row['rejection_reason'] = 'No different-anchor terminal action'
        rows.append(row)
        candidates.append(candidate)
    target.evaluate([s for s in candidates if s is not None], phase=phase)
    log_u = torch.rand(len(states), dtype=torch.float64, generator=generator).log()
    updated = list(states)
    for index, (old, new, row) in enumerate(zip(states, candidates, rows)):
        row['log_uniform'] = float(log_u[index])
        if new is None:
            continue
        target_ratio = -float(new['potential_eV']-old['potential_eV'])/target.kT
        ratio = target_ratio+float(row['coordinate_log_ratio'])+row['action_log_ratio']
        take = float(log_u[index]) < min(0., ratio)
        row.update(new_state_id=new['state_id'], target_log_ratio=target_ratio,
                   log_acceptance_ratio=ratio, accepted=take)
        if take:
            updated[index] = new
    return updated, rows
