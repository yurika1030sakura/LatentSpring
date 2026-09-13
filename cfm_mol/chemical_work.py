"""Paired finite-work surrogates; these are not physical energy or density models.

Both active atoms are absent from the passive encoder. A shared decoder sees
their two placements, and its difference has exact pair-reversal antisymmetry.
This property does not imply cycle consistency across different active sets.
"""
import math
import torch
from torch import nn
from cfm_mol.chemical_moves import covalent_radii


def harmonic_change(x, y, restraint=.1):
    x = x - x.mean(-2, keepdim=True)
    y = y - y.mean(-2, keepdim=True)
    return restraint / 2 * (y.square().sum((-2, -1)) - x.square().sum((-2, -1)))


def informed_log_prob(work, log_volume, kT, uniform_fraction=.1):
    """Normalized heuristic informed policy with positive uniform support.

    The exact MH correction still needs the independently normalized reverse
    catalogue; the uniform mixture is not itself a locally balanced kernel.
    """
    if work.ndim != 1 or not len(work) or not 0 < uniform_fraction <= 1 or kT <= 0:
        raise ValueError('Nonempty catalogue, positive temperature and mixture required')
    if not torch.isfinite(work).all() or not torch.isfinite(log_volume).all():
        raise ValueError('Nonfinite finite-work prediction or map volume')
    logits = torch.nn.functional.logsigmoid(-work / kT + log_volume)
    normalized = logits - torch.logsumexp(logits, 0)
    if uniform_fraction == 1:
        return torch.full_like(work, -math.log(len(work)))
    return torch.logaddexp(normalized + math.log1p(-uniform_fraction),
                           torch.full_like(work, math.log(uniform_fraction / len(work))))


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


class PairedChemicalWork(nn.Module):
    def __init__(self, hidden=24, radial=16, geometry=True, restraint=.1):
        super().__init__()
        self.configuration = dict(hidden=hidden, radial=radial, geometry=geometry, restraint=restraint)
        self.geometry, self.restraint = geometry, restraint
        z = torch.arange(119, dtype=torch.float64)
        radius = covalent_radii(range(119))
        self.register_buffer('radii', radius)
        self.register_buffer('descriptors', torch.stack([z/118, torch.log1p(z)/math.log(119), radius/2], -1))
        self.register_buffer('centers', torch.linspace(0., 6., radial))
        self.width = 6. / (radial - 1)
        self.elements = mlp(3, hidden, hidden)
        self.electronic = mlp(3, hidden, hidden)
        self.messages = nn.ModuleList([mlp(2*hidden+radial+1, hidden, hidden) for _ in range(2)])
        self.updates = nn.ModuleList([mlp(2*hidden, hidden, hidden) for _ in range(2)])
        self.query = mlp(2*hidden+radial+1, hidden, hidden)
        self.active_update = mlp(2*hidden, hidden, hidden)
        self.node_energy = mlp(hidden, hidden, 1)
        self.pair_energy = mlp(2*hidden+radial+1, hidden, 1)
        for head in (self.node_energy[-1], self.pair_energy[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def radial_features(self, delta, radius):
        if not self.geometry:
            return delta.new_zeros((*delta.shape[:-1], len(self.centers)))
        distance = delta.square().sum(-1).clamp_min(1e-20).sqrt() / radius
        return torch.exp(-.5*((distance[..., None]-self.centers)/self.width)**2)

    def encode(self, x, bonds, numbers, electronic, active):
        """Batch has one composition, possibly different two-atom masks."""
        batch, n, _ = x.shape
        if not 4 <= n <= 200 or active.shape != (batch, 2):
            raise ValueError('Two distinct active atoms and 4..200 total atoms required')
        if (active < 0).any() or (active >= n).any() or (active[:, 0] == active[:, 1]).any():
            raise ValueError('Invalid active indices')
        mask = torch.ones((batch, n), dtype=torch.bool, device=x.device)
        mask.scatter_(1, active, False)
        passive = torch.arange(n, device=x.device).expand(batch, n)[mask].reshape(batch, n-2)
        b = torch.arange(batch, device=x.device)[:, None]
        positions = x[b, passive]
        positions = positions - positions.mean(1, keepdim=True)
        graph = bonds[b[:, :, None], passive[:, :, None], passive[:, None, :]]
        z = numbers[passive]
        state = self.electronic(electronic.to(x))[:, None]
        nodes = self.elements(self.descriptors[z].to(x)) + state
        radial = self.radial_features(positions[:, :, None]-positions[:, None, :],
                                     self.radii[z][:, :, None]+self.radii[z][:, None, :])
        gate = (~torch.eye(n-2, dtype=torch.bool, device=x.device))[None, :, :, None]
        for message, update in zip(self.messages, self.updates):
            left = nodes[:, :, None].expand(-1, -1, n-2, -1)
            right = nodes[:, None, :].expand(-1, n-2, -1, -1)
            features = torch.cat([left+right, (left-right).square(), radial, graph[..., None]/3], -1)
            aggregate = (message(features)*gate).sum(2) / math.sqrt(n-3)
            nodes = nodes + update(torch.cat([nodes, aggregate], -1))
        return dict(passive=passive, active=active, nodes=nodes, positions=positions,
                    numbers=numbers, state=state)

    def energy(self, x, bonds, context):
        passive, active = context['passive'], context['active']
        batch = torch.arange(len(x), device=x.device)[:, None]
        # Recenter each endpoint by its own passive centroid. Under the bare
        # edit this is the same passive frame despite the total-COM translation.
        location = x[batch, active] - x[batch, passive].mean(1, keepdim=True)
        numbers = context['numbers']
        root = self.elements(self.descriptors[numbers[active]].to(x)) + context['state']
        nodes = context['nodes']
        left = root[:, :, None].expand(-1, -1, nodes.shape[1], -1)
        right = nodes[:, None].expand(-1, 2, -1, -1)
        radial = self.radial_features(location[:, :, None]-context['positions'][:, None],
            self.radii[numbers[active]][:, :, None]+self.radii[numbers[passive]][:, None])
        graph = bonds[batch[:, :, None], active[:, :, None], passive[:, None, :]]
        query = self.query(torch.cat([left, right, radial, graph[..., None]/3], -1))
        root = root + self.active_update(torch.cat([root, query.sum(2)/math.sqrt(nodes.shape[1])], -1))
        joint_radial = self.radial_features(location[:, 0]-location[:, 1], self.radii[numbers[active]].sum(1))
        joint_bond = bonds[batch[:, 0], active[:, 0], active[:, 1]][:, None]/3
        joint = torch.cat([root.sum(1), (root[:, 0]-root[:, 1]).square(), joint_radial, joint_bond], -1)
        return self.node_energy(root).sum((1, 2)) + self.pair_energy(joint)[:, 0]

    def forward(self, x, y, bonds, new_bonds, numbers, electronic, active):
        context = self.encode(x, bonds, numbers, electronic, active)
        return (self.energy(y, new_bonds, context)-self.energy(x, bonds, context)
                + harmonic_change(x, y, self.restraint))


class LinearBondWork(nn.Module):
    """Additive typed bond energies; unobserved element pairs contribute zero."""
    def __init__(self, elements, restraint=.1):
        super().__init__()
        self.configuration = dict(elements=list(elements), restraint=restraint)
        self.restraint = restraint
        lookup = torch.full((119, 119), -1, dtype=torch.long)
        index = 0
        for i, z in enumerate(elements):
            for w in elements[i:]:
                lookup[z, w] = lookup[w, z] = index
                index += 1
        self.register_buffer('lookup', lookup)
        self.register_buffer('orders', torch.tensor([1., 1.5, 2., 3.]))
        self.coefficients = nn.Parameter(torch.zeros(index, 4))

    def energy(self, bonds, numbers):
        index = self.lookup[numbers[:, None], numbers[None, :]]
        allowed = (index >= 0) & torch.triu(torch.ones_like(index, dtype=torch.bool), diagonal=1)
        order = (bonds[..., None]-self.orders).abs() < 1e-6
        values = self.coefficients[index.clamp_min(0)]
        return (values[None]*order*allowed[None, ..., None]).sum((1, 2, 3))

    def forward(self, x, y, bonds, new_bonds, numbers, electronic, active):
        return self.energy(new_bonds, numbers)-self.energy(bonds, numbers)+harmonic_change(x, y, self.restraint)


class ResidualChemicalWork(nn.Module):
    """A frozen additive work model plus a paired non-additive correction.

    Residual learning is a control-preserving architectural change, not itself
    a novel sampling principle. The correction contains no second restraint.
    """
    def __init__(self, elements, hidden=24, radial=16, geometry=True, restraint=.1):
        super().__init__()
        self.configuration = dict(elements=list(elements), hidden=hidden, radial=radial,
                                  geometry=geometry, restraint=restraint)
        self.backbone = LinearBondWork(elements, restraint=restraint)
        self.backbone.requires_grad_(False)
        self.residual = PairedChemicalWork(hidden=hidden, radial=radial, geometry=geometry, restraint=0.)

    def forward(self, x, y, bonds, new_bonds, numbers, electronic, active):
        args = (x, y, bonds, new_bonds, numbers, electronic, active)
        return self.backbone(*args)+self.residual(*args)
