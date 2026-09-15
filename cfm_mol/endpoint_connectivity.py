"""Coordinate-only endpoint support penalty; no bond labels or inference repair.

The minimum-spanning-tree objective is a standard piecewise smooth construction,
not a new tree theorem. A zero tree hinge ensures contact connectivity at the
declared radius, but neither chemical validity nor thermal sampling follows.
"""
import math
import numpy as np
import torch


def spanning_edges(distance):
    """Prim on detached costs; differentiate the selected minimum branch."""
    values = distance.detach().double().cpu().numpy()
    n = len(values)
    reached = np.zeros(n, dtype=bool)
    reached[0] = True
    edges = []
    for _ in range(n - 1):
        candidate = np.where(reached[:, None] & ~reached[None, :], values, np.inf)
        i, j = np.unravel_index(np.argmin(candidate), candidate.shape)
        edges.append((int(i), int(j)))
        reached[j] = True
    return torch.tensor(edges, dtype=torch.long, device=distance.device)


def endpoint_support_loss(x, radii, mode='tree', contact=1.15, exclusion=0.65):
    if mode not in ('tree', 'local'):
        raise ValueError('Unknown endpoint support mode')
    if x.ndim != 2 or x.shape[-1] != 3 or radii.shape != x.shape[:1] or len(x) < 2:
        raise ValueError('Expected at least two coordinates and one radius per atom')
    if not torch.isfinite(x).all() or not torch.isfinite(radii).all() or (radii <= 0).any():
        raise ValueError('Coordinates and positive radii must be finite')
    if not (math.isfinite(contact) and math.isfinite(exclusion) and 0 < exclusion < contact):
        raise ValueError('Invalid geometric margins')
    relative = torch.cdist(x[None], x[None])[0] / (radii[:, None] + radii[None, :])
    if mode == 'tree':
        edges = spanning_edges(relative)
        bridge = (relative[edges[:, 0], edges[:, 1]] - contact).relu().square().mean()
    else:
        diagonal = torch.eye(len(x), dtype=torch.bool, device=x.device)
        nearest = relative.masked_fill(diagonal, torch.inf).min(-1).values
        bridge = (nearest - contact).relu().square().mean()
    i, j = torch.triu_indices(len(x), len(x), 1, device=x.device)
    overlap = (exclusion - relative[i, j]).relu().square().sum() / len(x)
    return bridge + overlap
