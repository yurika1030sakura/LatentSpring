"""Fixed-context angular probes and identifiable local vMF surrogate fits.

The fitted full-sphere density is a local proposal approximation. It is not the
normalizer or density of a hard-support molecular conditional distribution.
"""
import math
import torch


def angular_probes(x, root, *, generator, fit_angle=.15, check_angle=.1):
    leaf, anchor = root
    vector = x[leaf] - x[anchor]
    radius = vector.norm()
    if not torch.isfinite(x).all() or radius <= 1e-10 or not 0 < check_angle < fit_angle < math.pi/2:
        raise ValueError('Finite geometry, positive radius and ordered local probe angles required')
    u = vector/radius
    noise = torch.randn(3, dtype=x.dtype, device=x.device, generator=generator)
    tangent = noise-(noise*u).sum()*u
    if tangent.norm() < 1e-12:
        raise ValueError('Degenerate random tangent frame')
    a = tangent/tangent.norm()
    b = torch.cross(u, a, dim=0)
    specs = [('fit', a, 1), ('fit', a, -1), ('fit', b, 1), ('fit', b, -1),
             ('check', (a+b)/math.sqrt(2), 1), ('check', (a+b)/math.sqrt(2), -1)]
    result = []
    for number, (role, axis, sign) in enumerate(specs):
        angle = fit_angle if role == 'fit' else check_angle
        direction = math.cos(angle)*u+sign*math.sin(angle)*axis
        y = x.clone()
        y[leaf] = x[anchor]+radius*direction
        y -= y.mean(0)
        result.append(dict(number=number, role=role, axis=axis, sign=sign,
                           angle_rad=angle, positions=y))
    return result, noise


def angular_force(state, root, kT, restraint):
    leaf, anchor = root
    vector = state['positions'][leaf]-state['positions'][anchor]
    radius = vector.norm()
    u = vector/radius
    force = state['force_eV_A']-restraint*state['positions']
    force = force-force.mean(0)
    return u, radius*(force[leaf]-(force[leaf]*u).sum()*u)/kT


def identify_parameter(directions, scores, *, rank_threshold=1e-5):
    if directions.shape != scores.shape or directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError('Matched angular direction and score arrays required')
    if not torch.isfinite(directions).all() or not torch.isfinite(scores).all():
        raise ValueError('Finite angular observations required')
    torch.testing.assert_close(directions.norm(dim=1), torch.ones_like(directions[:, 0]), atol=1e-9, rtol=0)
    projectors = torch.eye(3, dtype=directions.dtype, device=directions.device)[None]-directions[:, :, None]*directions[:, None, :]
    design = projectors.reshape(-1, 3)
    eigenvalues = torch.linalg.eigvalsh(design.T@design)
    result = dict(design_eigenvalues=eigenvalues.tolist(), full_rank=bool(eigenvalues[0] > rank_threshold))
    if result['full_rank']:
        eta = torch.linalg.lstsq(design, scores.flatten()).solution
        result.update(fitted_parameter=eta.tolist(), fit_score_mse=float((design@eta-scores.flatten()).square().mean()))
    return result
