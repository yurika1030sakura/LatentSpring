"""Learned local geometry in the existing self-conditioning edge channel.

Directional first and second moments expose neighborhood angles and departure
from a plane using O(N^2) work. Zero initialization exactly preserves the old
distance context. Only coordinates and elements are inputs; no bonds are used.
This candidate is separate from the frozen recovery-training pilot.
"""
import torch
from torch import nn
from .chemical_moves import covalent_radii


def local_moments(coordinates, radii):
    relative = coordinates[:, :, None] - coordinates[:, None, :]
    distance2 = relative.square().sum(-1)
    distance = (distance2 + 1e-12).sqrt()
    lengths = radii[:, :, None] + radii[:, None, :]
    offdiag = 1 - torch.eye(coordinates.shape[1], dtype=coordinates.dtype,
                             device=coordinates.device)[None]
    contacts = torch.sigmoid((1.30 - distance / lengths) / .20) * offdiag
    degree = contacts.sum(-1)
    normalized = contacts / degree.clamp_min(1.)[..., None]
    unit = relative / distance[..., None]
    first = (normalized[..., None] * unit).sum(2)
    second = torch.einsum('bij,bijd,bije->bide', normalized, unit, unit)
    return distance2, lengths, unit, contacts, degree, first, second, offdiag


class GeometryMomentContext(nn.Module):
    def __init__(self, atomic_numbers, embedding_dim=16, hidden_dim=64, amplitude=.25, mode='moments'):
        super().__init__()
        if amplitude <= 0 or len(set(atomic_numbers)) != len(atomic_numbers):
            raise ValueError('Positive amplitude and unique element vocabulary required')
        if mode not in ('moments', 'radial'):
            raise ValueError('Unknown geometry context feature mode')
        self.configuration = dict(atomic_numbers=list(atomic_numbers),
            embedding_dim=embedding_dim, hidden_dim=hidden_dim, amplitude=amplitude, mode=mode)
        lookup = torch.full((84,), -1, dtype=torch.long)
        lookup[torch.tensor(atomic_numbers)] = torch.arange(len(atomic_numbers))
        self.register_buffer('lookup', lookup)
        self.register_buffer('radii', covalent_radii(atomic_numbers).float())
        self.embedding = nn.Embedding(len(atomic_numbers), embedding_dim)
        self.neighbor_embedding = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.network = nn.Sequential(nn.Linear(2 * embedding_dim + 10, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.amplitude = float(amplitude)
        self.mode = mode

    def pair_features(self, coordinates, numbers):
        # Match detached provisional-geometry self-conditioning while allowing
        # gradients through the learned context parameters.
        x = coordinates.detach()
        types = self.lookup[numbers]
        if (types < 0).any():
            raise ValueError('Unsupported element')
        radii = self.radii[types]
        d2, lengths, unit, contacts, degree, first, second, offdiag = local_moments(x, radii)
        embeddings = self.embedding(types)
        neighborhood = torch.einsum('bij,bje->bie', contacts, embeddings) / degree.clamp_min(1.)[..., None]
        nodes = embeddings + self.neighbor_embedding(neighborhood)
        log_degree = torch.log1p(degree)
        first_norm = first.square().sum(-1)
        tensor_norm = second.square().sum((-1, -2))
        determinant = torch.linalg.det(second)
        tensors = second[:, :, None] + second[:, None, :]
        directional = torch.einsum('bijd,bijde,bije->bij', unit, tensors, unit)
        scalars = torch.stack([
            torch.log1p(d2 / lengths.square()),
            log_degree[:, :, None] + log_degree[:, None, :],
            (log_degree[:, :, None] - log_degree[:, None, :]).square(),
            ((first[:, :, None] - first[:, None, :]) * unit).sum(-1),
            first_norm[:, :, None] + first_norm[:, None, :],
            (first[:, :, None] * first[:, None, :]).sum(-1),
            tensor_norm[:, :, None] + tensor_norm[:, None, :],
            determinant[:, :, None] + determinant[:, None, :],
            directional, lengths / 2], -1)
        if self.mode == 'radial':
            centers = x.new_tensor([.8, 1., 1.2, 1.5, 2., 3.])
            scaled_distance = (d2 + 1e-12).sqrt() / lengths
            radial = torch.exp(-.5 * ((scaled_distance[..., None] - centers) / .25).square())
            scalars = torch.cat([scalars[..., :3], radial, scalars[..., -1:]], -1)
        features = torch.cat([nodes[:, :, None] + nodes[:, None, :],
            nodes[:, :, None] * nodes[:, None, :], scalars], -1)
        return d2, lengths, unit, contacts, degree, first, second, offdiag, features

    def forward(self, coordinates, numbers):
        d2, lengths, _, _, _, _, _, offdiag, features = self.pair_features(coordinates, numbers)
        residual = torch.tanh(self.network(features)[..., 0])
        window = torch.sigmoid((2.0 - (d2 + 1e-12).sqrt() / lengths) / .30)
        value = d2 + self.amplitude * lengths.square() * window * offdiag * residual
        if not torch.isfinite(value).all():
            raise FloatingPointError('Nonfinite geometry context')
        return value.reshape(-1, 1)
