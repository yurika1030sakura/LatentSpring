"""Molecular coupling adapter with exact constrained volume and neural contexts.

The base flow-matching generator is frozen. These layers operate on its sampled
coordinates; they require neither its score nor density for energy refinement.
Equivariant/convex coupling principles are prior art. Molecular performance and
novelty require separate experiments.
"""
import math

import torch
from torch import nn

from cfm_mol.centered_convex_flow import (centered_convex_forward, centered_convex_inverse,
    convex_point_map, convex_point_inverse)


class InvariantContext(nn.Module):
    def __init__(self, hidden=32, radial=24):
        super().__init__()
        self.elements = nn.Embedding(119, hidden)
        self.roles = nn.Embedding(3, hidden)
        self.state = nn.Sequential(nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.register_buffer('centers', torch.linspace(0, 8, radial))
        self.messages = nn.ModuleList([nn.Sequential(nn.Linear(2*hidden+radial, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU()) for _ in range(2)])
        self.updates = nn.ModuleList([nn.Sequential(nn.Linear(2*hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)) for _ in range(2)])
        self.node_head = nn.Linear(hidden, 3)
        self.group_head = nn.Linear(hidden, 2)
        for head in [self.node_head, self.group_head]:
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)

    def forward(self, context, origin, numbers, roles, electronic, mode):
        batch, atoms, _ = context.shape
        distance = ((context[:, :, None]-context[:, None, :]).square().sum(-1)+1e-8).sqrt()
        radial = torch.exp(-.5*((distance[..., None]-self.centers)/.4)**2)
        state = torch.cat([electronic, electronic.new_tensor([mode])])
        nodes = (self.elements(numbers)+self.roles(roles)+self.state(state)[None]).expand(batch, atoms, -1)
        mask = (~torch.eye(atoms, dtype=torch.bool, device=context.device))[None, :, :, None]
        for message, update in zip(self.messages, self.updates):
            left = nodes[:, :, None].expand(-1, -1, atoms, -1)
            right = nodes[:, None, :].expand(-1, atoms, -1, -1)
            pair = message(torch.cat([left+right, (left-right).square(), radial], -1))
            nodes = nodes+update(torch.cat([nodes, (pair*mask).sum(2)/atoms], -1))
        node_output = self.node_head(nodes); group_output = self.group_head(nodes.mean(1))
        directions = context-origin
        unit = directions/(directions.square().sum(-1, keepdim=True)+.01).sqrt()
        shift = .1*(torch.tanh(node_output[..., 2])[..., None]*unit).mean(1, keepdim=True)
        parameters = (node_output[..., 0], directions, node_output[..., 1], group_output[:, 0], group_output[:, 1])
        return parameters, shift


class SpeciesCouplingAdapter(nn.Module):
    """Whole-element equivariant flow, with an optional labelled-block diagnostic.

    ``split_groups=True`` uses fixed atom-index halves. It preserves O(3), COM
    and the exact conditional determinant, but is NOT generally equivariant to
    same-element permutations. It also fixes each half's centroid. Do not use
    that diagnostic to support the whole-element architecture's symmetry claim.
    """
    #: smallest active block the centered map accepts (it needs at least two points)
    MINIMUM_ACTIVE = 2

    def __init__(self, numbers, *, charge, spin_multiplicity, kT, sweeps=1, hidden=32, radial=24,
                 split_groups=False):
        super().__init__()
        numbers = torch.as_tensor(numbers, dtype=torch.long)
        if numbers.ndim != 1 or not 2 <= len(numbers) <= 200 or ((numbers < 1) | (numbers > 118)).any():
            raise ValueError('Require2--200 declared atoms')
        if not math.isfinite(kT) or kT <= 0 or spin_multiplicity < 1 or not isinstance(sweeps, int) or sweeps < 1:
            raise ValueError('Invalid condition or sweep count')
        self.register_buffer('numbers', numbers)
        self.register_buffer('electronic', torch.tensor([charge/5., (spin_multiplicity-1)/5., math.log(kT)], dtype=torch.float64))
        self.conditioner = InvariantContext(hidden, radial)
        types = sorted(set(numbers.tolist()))
        one = []
        for kind in types:
            index = (numbers == kind).nonzero().flatten().tolist()
            if len(index) < 2:
                continue
            if split_groups and len(index) >= 2*self.MINIMUM_ACTIVE:
                # Split the group. Off by default so that checkpoints trained before
                # this repair replay bit-for-bit; new runs opt in.
                # A single whole-group layer replaces every active
                # atom by the group centroid, so on a homogeneous system every atom
                # is active, every direction vector context-origin is exactly zero,
                # every sigmoid feature vanishes and the per-atom conditioner is
                # structurally dead -- the map degenerates to a fixed two-parameter
                # isotropic radial rescaling. Splitting keeps the complementary half
                # at its real positions, so the conditioner sees real neighbours.
                half = len(index)//2
                one.append(('internal', kind, tuple(index[:half])))
                one.append(('internal', kind, tuple(index[half:])))
            else:
                one.append(('internal', kind, None))
        one += [('centroid', first, second) for first, second in zip(types, types[1:])]
        self.layers = []
        for sweep in range(sweeps):self.layers.extend(one if sweep % 2 == 0 else list(reversed(one)))
        self.permutation_equivariant = not any(mode == 'internal' and block is not None
            for mode, _, block in one)
        self.configuration = {'sweeps': sweeps, 'hidden': hidden, 'radial': radial,
            'charge': charge, 'spin_multiplicity': spin_multiplicity, 'kT': kT,
            'minimum_active': self.MINIMUM_ACTIVE, 'split_groups': bool(split_groups),
            'permutation_equivariant': self.permutation_equivariant,
            'internal_blocks': [list(layer[2]) if layer[2] is not None else None
                                for layer in one if layer[0] == 'internal']}

    def split_context(self, x, layer):
        mode, first, second = layer
        if mode == 'internal' and second is not None:
            a = torch.zeros_like(self.numbers, dtype=torch.bool)
            a[torch.as_tensor(second, dtype=torch.long, device=self.numbers.device)] = True
        else:
            a = self.numbers == first
        ma = x[:, a].mean(1, keepdim=True)
        context = x.clone(); roles = torch.zeros_like(self.numbers); roles[a] = 1
        if mode == 'internal':
            # Fixed labelled complement; exchanging atoms across these blocks
            # changes the map. This role embedding does not restore permutation symmetry.
            roles[(self.numbers == first) & ~a] = 2
            context[:, a] = ma
            return context, ma, x[:, a]-ma, roles, a, None
        b = self.numbers == second; roles[b] = 2
        mb = x[:, b].mean(1, keepdim=True)
        na, nb = int(a.sum()), int(b.sum())
        center = (na*ma+nb*mb)/(na+nb)
        context[:, a] = center+x[:, a]-ma
        context[:, b] = center+x[:, b]-mb
        return context, center, ma-mb, roles, a, b

    def apply_layer(self, x, layer, *, inverse=False, tolerance=1e-11):
        context, origin, active, roles, a, b = self.split_context(x, layer)
        parameters, shift = self.conditioner(context, origin, self.numbers, roles, self.electronic,
            0. if layer[0] == 'internal' else 1.)
        out = x.clone()
        if layer[0] == 'internal':
            if inverse:
                changed, diagnostic = centered_convex_inverse(active, *parameters, tolerance=tolerance)
                _, volume = centered_convex_forward(changed, *parameters)
            else:
                changed, volume = centered_convex_forward(active, *parameters); diagnostic = None
            out[:, a] = origin+changed
        else:
            if inverse:
                changed, diagnostic = convex_point_inverse(active-shift, *parameters, tolerance=tolerance)
                _, derivative, _ = convex_point_map(changed, *parameters)
            else:
                changed, derivative, _ = convex_point_map(active, *parameters)
                changed = changed+shift; diagnostic = None
            volume = 2*torch.linalg.cholesky(derivative[:, 0]).diagonal(dim1=-2, dim2=-1).log().sum(-1)
            difference = changed-active
            na, nb = int(a.sum()), int(b.sum())
            out[:, a] = x[:, a]+nb/(na+nb)*difference
            out[:, b] = x[:, b]-na/(na+nb)*difference
        return out, -volume if inverse else volume, diagnostic

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (len(self.numbers), 3) or not torch.isfinite(x).all():
            raise ValueError('Invalid molecular coordinates')
        if float(x.mean(1).abs().max()) > 1e-8:raise ValueError('Input must be in the declared COM-free subspace')
        total = x.new_zeros(len(x))
        for layer in self.layers:
            x, volume, _ = self.apply_layer(x, layer)
            total = total+volume
        return x, total

    @torch.no_grad()
    def inverse(self, y, *, tolerance=1e-11):
        """Reconstruction only; this does not implement inverse likelihood gradients."""
        x = y; total = y.new_zeros(len(y)); diagnostics = []
        for layer in reversed(self.layers):
            x, volume, diagnostic = self.apply_layer(x, layer, inverse=True, tolerance=tolerance)
            total = total+volume; diagnostics.append(diagnostic)
        reconstructed, forward_volume = self(x)
        residual = float((reconstructed-y).abs().max())
        if residual > 1e-8:raise RuntimeError('Full coupling inverse fails reconstruction tolerance')
        return x, total, {'maximum_residual': residual, 'volume_cancellation_error': float((total+forward_volume).abs().max()),
            'layers': diagnostics}
