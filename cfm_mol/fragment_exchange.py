"""Reversible pendant-fragment exchanges with explicit Cartesian auxiliaries.

This is a physical proposal primitive, not learned sampling novelty or a reaction
trajectory. The target and graph-perception domain are supplied by the caller.
"""
import math
import torch
from cfm_mol.joint_chemical_geometry import radial_cartesian_log_density
from cfm_mol.spherical_proposal import vmf_sample, vmf_log_prob


def _neighbors(bonds):
    if (bonds.ndim != 2 or bonds.shape[0] != bonds.shape[1]
            or not torch.isfinite(bonds).all() or not torch.equal(bonds, bonds.T)
            or bonds.diagonal().any() or (bonds < 0).any()):
        raise ValueError('Finite symmetric bond matrix with zero diagonal required')
    return [(row > 0).nonzero().flatten().tolist() for row in bonds]


def _component(neighbors, root, anchor):
    visited, pending = {root}, [root]
    while pending:
        node = pending.pop()
        for other in neighbors[node]:
            if (node == root and other == anchor) or (node == anchor and other == root):
                continue
            if other not in visited:
                visited.add(other)
                pending.append(other)
    return visited


def fragment_exchange_actions(bonds, *, max_fragment_atoms=None):
    neighbors = _neighbors(bonds)
    n = len(neighbors)
    cap = n-3 if max_fragment_atoms is None else min(n-3, max_fragment_atoms)
    if max_fragment_atoms is not None and max_fragment_atoms < 1:
        raise ValueError('Positive fragment cap required')
    if n and len(_component(neighbors, 0, 0)) != n:
        raise ValueError('Connected source bond graph required')
    fragments = []
    for root in range(n):
        for anchor in neighbors[root]:
            if float(bonds[root, anchor]) != 1.:
                continue
            atoms = _component(neighbors, root, anchor)
            if anchor not in atoms and len(atoms) <= cap:
                fragments.append((root, anchor, atoms))
    result = []
    for index, (i, k, first) in enumerate(fragments):
        for j, l, second in fragments[index+1:]:
            if i >= j or k == l or first & second or k in second or l in first:
                continue
            result.append(dict(roots=(i, j, k, l), fragments=(tuple(sorted(first)), tuple(sorted(second)))))
    return sorted(result, key=lambda a: a['roots'])


def inverse_fragment_action(action):
    i, j, k, l = action['roots']
    return dict(roots=(i, j, l, k), fragments=action['fragments'])


def fragment_bond_graph(bonds, action):
    neighbors = _neighbors(bonds)
    i, j, k, l = action['roots']
    first, second = map(set, action['fragments'])
    if (i == j or k == l or first & second or {k, l} & (first | second)
            or _component(neighbors, i, k) != first or _component(neighbors, j, l) != second
            or float(bonds[i, k]) != 1 or float(bonds[j, l]) != 1
            or float(bonds[i, l]) != 0 or float(bonds[j, k]) != 0):
        raise ValueError('Disjoint pendant fragments and passive anchors required')
    result = bonds.clone()
    result[i, k] = result[k, i] = result[j, l] = result[l, j] = 0
    result[i, l] = result[l, i] = result[j, k] = result[k, j] = 1
    return result


def skew(vector):
    a, b, c = vector.unbind(-1)
    z = torch.zeros_like(a)
    return torch.stack([z, -c, b, c, z, -a, -b, a, z], -1).reshape(*vector.shape[:-1], 3, 3)


def alignment_rotation(old_unit, new_unit, *, antipodal_tolerance=1e-8):
    cosine = (old_unit*new_unit).sum(-1)
    if (cosine <= -1+antipodal_tolerance).any():
        raise ValueError('Symmetric near-antipodal alignment exclusion')
    k = skew(torch.linalg.cross(old_unit, new_unit, dim=-1))
    eye = torch.eye(3, dtype=k.dtype, device=k.device)
    return eye+k+(k@k)/(1+cosine)[..., None, None]


def axis_rotation(unit, angle):
    k = skew(unit)
    eye = torch.eye(3, dtype=k.dtype, device=k.device)
    return eye+angle.sin()[..., None, None]*k+(1-angle.cos())[..., None, None]*(k@k)


def fragment_exchange_map(x, action, new_vectors, torsions):
    """Augmented involution: (x,new_vectors,torsions) -> (y,old_vectors,-torsions).

    Both root Cartesian vectors are swapped with auxiliaries; internal offsets
    rotate properly. The full COM-plus-auxiliary absolute Jacobian is one.
    """
    unbatched = x.ndim == 2
    if unbatched:
        x, new_vectors, torsions = x[None], new_vectors[None], torsions[None]
    if (x.ndim != 3 or x.shape[-1] != 3 or new_vectors.shape != (len(x), 2, 3)
            or torsions.shape != (len(x), 2) or not 4 <= x.shape[1] <= 200):
        raise ValueError('Matched COM coordinate and auxiliary batches required')
    if not all(torch.isfinite(v).all() for v in [x, new_vectors, torsions]):
        raise ValueError('Finite coordinates and auxiliaries required')
    if float(x.mean(1).abs().max()) > 1e-8:
        raise ValueError('Noncentered source')
    i, j, k, l = action['roots']
    first, second = action['fragments']
    if (i not in first or j not in second or set(first) & set(second)
            or {k, l} & (set(first) | set(second)) or k == l):
        raise ValueError('Disjoint fragments and passive distinct anchors required')
    old_vectors = torch.stack([x[:, i]-x[:, k], x[:, j]-x[:, l]], 1)
    old_radius = old_vectors.norm(dim=2, keepdim=True)
    new_radius = new_vectors.norm(dim=2, keepdim=True)
    if (old_radius <= 1e-10).any() or (new_radius <= 1e-10).any():
        raise ValueError('Nonzero attachment vectors required')
    old_unit, new_unit = old_vectors/old_radius, new_vectors/new_radius
    rotations = axis_rotation(new_unit, torsions)@alignment_rotation(old_unit, new_unit)
    y = x.clone()
    for index, (root, anchor, atoms) in enumerate([(i, l, first), (j, k, second)]):
        offsets = x[:, list(atoms)]-x[:, root, None]
        rotated = torch.einsum('bij,bnj->bni', rotations[:, index], offsets)
        y[:, list(atoms)] = x[:, anchor, None]+new_vectors[:, index, None]+rotated
    y = y-y.mean(1, keepdim=True)
    if unbatched:
        return y[0], old_vectors[0], -torsions[0], rotations[0]
    return y, old_vectors, -torsions, rotations


def fragment_force_responses(x, forces, root, anchor, atoms, kT):
    """Candidate direction/torsion/log-radius physical derivatives at the source.

    The log-radius response here excludes the +3 density-coordinate term; it
    differentiates only -U/kT. Forces must include the physical restraint.
    """
    if not math.isfinite(kT) or kT <= 0 or x.shape != forces.shape:
        raise ValueError('Matched forces and positive finite temperature required')
    force = forces-forces.mean(0)
    w = x[root]-x[anchor]
    if float(w.norm()) <= 1e-10:
        raise ValueError('Nonzero attachment vector required')
    radius, unit = w.norm(), w/w.norm()
    total = force[list(atoms)].sum(0)
    torque = torch.linalg.cross(x[list(atoms)]-x[root], force[list(atoms)], dim=1).sum(0)
    raw = radius*total+torch.linalg.cross(torque, unit)
    return dict(direction=(raw-torch.dot(raw, unit)*unit)/kT,
                torsion=torch.dot(torque, unit)/kT, log_radius=radius*torch.dot(total, unit)/kT)


def root_proposal_parameters(x, bonds, radii, action, concentration):
    desired = fragment_bond_graph(bonds, action)
    i, j, k, l = action['roots']
    etas, means = [], []
    for root, anchor in [(i, l), (j, k)]:
        mask = desired[anchor] > 0
        mask[root] = False
        vectors = x[mask]-x[anchor]
        away = -(vectors/vectors.norm(dim=1, keepdim=True).clamp_min(1e-12)).sum(0)
        etas.append(concentration*away/away.norm().clamp_min(1e-12))
        means.append((radii[root]+radii[anchor]).log())
    return torch.stack(etas), torch.stack(means), desired


def root_auxiliary_log_prob(vectors, eta, mean, width):
    radius = vectors.norm(dim=1)
    return (radial_cartesian_log_density(radius.log(), mean, width)
            +vmf_log_prob(vectors/radius[:, None], eta)).sum()-2*math.log(2*math.pi)


@torch.no_grad()
def physical_fragment_proposal(x, bonds, radii, action, *, generator,
                               radial_width=.05, concentration=10.):
    eta, mean, desired = root_proposal_parameters(x, bonds, radii, action, concentration)
    noise = torch.randn(2, dtype=x.dtype, device=x.device, generator=generator)
    radius = (mean+radial_width*noise).exp()
    directions, random = vmf_sample(eta, generator=generator)
    vectors = radius[:, None]*directions
    uniform = torch.rand(2, dtype=x.dtype, device=x.device, generator=generator)
    torsions = 2*math.pi*uniform-math.pi
    trace = dict(radial_noise=noise, radial_means=mean, radial_width=radial_width,
        direction_parameters=eta, direction_random=random, torsion_uniform=uniform,
        new_vectors=vectors, torsions=torsions, desired_bonds=desired,
        log_forward=root_auxiliary_log_prob(vectors, eta, mean, radial_width))
    try:
        y, old_vectors, reverse_torsions, rotations = fragment_exchange_map(x, action, vectors, torsions)
    except ValueError as exc:
        trace.update(map_valid=False, reason=str(exc))
        return None, trace
    inverse = inverse_fragment_action(action)
    eta_back, mean_back, recovered_bonds = root_proposal_parameters(y, desired, radii, inverse, concentration)
    recovered, _, _, _ = fragment_exchange_map(y, inverse, old_vectors, reverse_torsions)
    torch.testing.assert_close(recovered, x, atol=1e-8, rtol=1e-9)
    assert torch.equal(recovered_bonds, bonds)
    trace.update(map_valid=True, old_vectors=old_vectors, reverse_torsions=reverse_torsions,
        rotations=rotations, reverse_parameters=eta_back, reverse_means=mean_back,
        log_reverse=root_auxiliary_log_prob(old_vectors, eta_back, mean_back, radial_width),
        augmented_log_jacobian=0.)
    return y, trace
