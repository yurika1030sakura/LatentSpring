"""SO(3)-orbit pairings for FM with a restored Gaussian source marginal.

Alignment selects a proper rotation of the Gaussian source. An independent,
shared Haar rotation of both endpoints restores the source's Gaussian orbit
law and rotates the data marginal uniformly without changing molecular shape.
These are training couplings, not inference maps or physical energy models.
"""
import math
import torch


def haar_rotation(like, generator):
    q = torch.randn((4,), dtype=like.dtype, device=like.device, generator=generator)
    q = q / q.norm()
    w, x, y, z = q.unbind()
    return torch.stack([
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]).reshape(3, 3)


def proper_alignment(source, target):
    u, _, vh = torch.linalg.svd(source.T @ target)
    sign = torch.ones(3, dtype=source.dtype, device=source.device)
    sign[-1] = torch.where(torch.linalg.det(u @ vh) >= 0, 1., -1.)
    return (u * sign[None]) @ vh


def path_cost(source, target, radii, steric_weight=4.):
    """Endpoint displacement plus bounded overlap cost along three path slices.

    Covalent radii define only a geometric surrogate. No bonds, physical energy
    values, target forces or validation outcomes enter this cost.
    """
    if source.ndim == 2:
        source = source[None]
    n = target.shape[0]
    displacement = (source-target).square().sum(-1).mean(-1)
    if n < 2:
        return displacement, torch.zeros_like(displacement)
    ii, jj = torch.triu_indices(n, n, offset=1, device=source.device)
    cutoff2 = (.6 * (radii[ii] + radii[jj])).square()
    t = source.new_tensor([.25, .5, .75])
    path = (1-t[None, :, None, None])*source[:, None] + t[None, :, None, None]*target[None, None]
    distance2 = (path[:, :, ii] - path[:, :, jj]).square().sum(-1)
    penalty = (1-distance2/cutoff2).clamp_min(0).square().sum(-1).mean(-1)/n
    return displacement + steric_weight * penalty, penalty


@torch.no_grad()
def orbit_pair(source, target, radii, *, mode, generator, steric_weight=4.):
    if mode not in ['independent', 'rotation', 'steric']:
        raise ValueError('Unknown orbit-pairing mode')
    if source.shape != target.shape or source.ndim != 2 or source.shape[-1] != 3:
        raise ValueError('One matching pair of atom coordinate matrices required')
    if radii.shape != (len(source),) or not torch.isfinite(radii).all() or (radii <= 0).any():
        raise ValueError('Positive covalent radii required')
    if not all(torch.isfinite(x).all() for x in [source, target]) or not math.isfinite(steric_weight) or steric_weight < 0:
        raise ValueError('Finite coordinates and nonnegative cost weight required')
    if max(float(source.mean(0).abs().max()), float(target.mean(0).abs().max())) > 1e-5:
        raise ValueError('Center both endpoints before coupling')
    # Draw all auxiliary randomness in the same order in every arm.
    axes = torch.randn((6, 3), dtype=source.dtype, device=source.device, generator=generator)
    axes = axes / axes.norm(dim=-1, keepdim=True)
    augmentation = haar_rotation(source, generator)
    identity = torch.eye(3, dtype=source.dtype, device=source.device)
    rotation = proper_alignment(source, target)
    rotations = [rotation]
    for i, axis in enumerate(axes):
        x, y, z = axis.unbind()
        zero = x*0
        skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero]).reshape(3, 3)
        angle = .4 if i < 3 else .8
        for sign in [-1, 1]:
            delta = identity + math.sin(sign*angle)*skew + (1-math.cos(angle))*(skew@skew)
            rotations.append(rotation @ delta)
    candidates = torch.einsum('ni,kij->knj', source, torch.stack(rotations))
    costs, penalties = path_cost(candidates, target, radii, steric_weight)
    chosen = int(costs.argmin()) if mode == 'steric' else 0
    aligned = source if mode == 'independent' else candidates[chosen]
    selected_cost, selected_penalty = path_cost(aligned, target, radii, steric_weight)
    return aligned @ augmentation, target @ augmentation, dict(
        mode=mode, selected_candidate=chosen if mode != 'independent' else -1,
        standard_cost=float(costs[0]), selected_cost=float(selected_cost[0]),
        standard_overlap=float(penalties[0]), selected_overlap=float(selected_penalty[0]),
        displacement_per_atom=float((aligned-target).square().sum(-1).mean()))


@torch.no_grad()
def typed_orbit_pair(source, target, radii, groups, *, generator):
    """Alternating type-preserving assignment/Kabsch, followed by group Haar.

    Groups must preserve every clamped node feature, and edge conditioning must
    be permutation invariant (checked by the caller). This is a standard
    symmetry-matching baseline, not the proposed collision-aware pairing.
    """
    from scipy.optimize import linear_sum_assignment
    if groups.shape!=(len(source),):raise ValueError('One conditioning group per atom required')
    # Use the same rotation/augmentation stream as the existing controls.
    axes=torch.randn((6,3),dtype=source.dtype,device=source.device,generator=generator)
    augmentation=haar_rotation(source,generator)
    rotation=proper_alignment(source,target)
    permutation=torch.arange(len(source),device=source.device)
    identity=permutation.clone()
    class_indices=[torch.where(groups==g)[0] for g in groups.unique()]
    for _ in range(3):
        moved=source@rotation
        for selected in class_indices:
            cost=(moved[selected,None]-target[None,selected]).square().sum(-1).cpu().numpy()
            left,right=linear_sum_assignment(cost)
            permutation[selected[torch.as_tensor(right,device=source.device)]]=selected[torch.as_tensor(left,device=source.device)]
        rotation=proper_alignment(source[permutation],target)
    # Independently randomize the finite label-stabilizer group in BOTH endpoints.
    random_permutation=identity.clone()
    for selected in class_indices:
        order=torch.randperm(len(selected),device=source.device,generator=generator)
        random_permutation[selected]=selected[order]
    aligned=source[permutation]@rotation
    standard=source@proper_alignment(source,target)
    cost,overlap=path_cost(aligned,target,radii)
    standard_cost,standard_overlap=path_cost(standard,target,radii)
    return aligned[random_permutation]@augmentation,target[random_permutation]@augmentation,dict(
        mode='typed_rotation',selected_candidate=-2,standard_cost=float(standard_cost[0]),selected_cost=float(cost[0]),
        standard_overlap=float(standard_overlap[0]),selected_overlap=float(overlap[0]),
        displacement_per_atom=float((aligned-target).square().sum(-1).mean()),
        atoms_reassigned=int((permutation!=identity).sum()))
