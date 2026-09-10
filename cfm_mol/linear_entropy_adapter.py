"""Bounded atom-type-equivariant linear flow with exact COM-free log volume.

This is a simple exact-entropy refinement baseline, not a new normalizing-flow
identity. Its objective does not require the base generator density or score.
"""
import math

import torch
from torch import nn


class LinearEntropyAdapter(nn.Module):
    def __init__(self, numbers, *, kind='typed', maximum_weight=.25):
        super().__init__()
        numbers = torch.as_tensor(numbers, dtype=torch.long)
        if numbers.ndim != 1 or not 2 <= len(numbers) <= 200 or kind not in ['typed', 'scalar']:
            raise ValueError('Require2--200 atoms and a declared adapter kind')
        if not 0 < maximum_weight <= 1:
            raise ValueError('Invalid spectral bound')
        self.kind = kind; self.maximum_weight = float(maximum_weight)
        pairs = torch.triu_indices(len(numbers), len(numbers), 1)
        pair_types = torch.stack([numbers[pairs[0]], numbers[pairs[1]]], -1).sort(-1).values
        unique = sorted(set(map(tuple, pair_types.tolist())))
        ids = torch.tensor([unique.index(tuple(pair)) for pair in pair_types.tolist()]) if kind == 'typed' else torch.zeros(len(pair_types), dtype=torch.long)
        self.register_buffer('numbers', numbers)
        self.register_buffer('pairs', pairs)
        self.register_buffer('pair_ids', ids)
        self.raw_weights = nn.Parameter(torch.zeros(len(unique) if kind == 'typed' else 1, dtype=torch.float64))

    def matrices(self):
        n = len(self.numbers)
        weights = self.maximum_weight*torch.tanh(self.raw_weights)
        adjacency = weights.new_zeros(n, n)
        adjacency[self.pairs[0], self.pairs[1]] = weights[self.pair_ids]
        adjacency = adjacency+adjacency.T
        generator = (adjacency-torch.diag(adjacency.sum(1)))/n
        transform = torch.matrix_exp(generator)
        # The translation eigenvalue of generator is exactly zero. Each atom
        # eigenmode occurs in all three Cartesian coordinates.
        return transform, 3*torch.trace(generator), generator

    def forward(self, positions):
        if positions.ndim != 3 or positions.shape[1:] != (len(self.numbers), 3) or not torch.isfinite(positions).all():
            raise ValueError('Invalid molecular positions')
        transform, logdet, _ = self.matrices()
        return torch.einsum('nm,bmd->bnd', transform, positions), logdet


def endpoint_kl_change(old_energy, new_energy, old_positions, new_positions, *, kT, restraint, log_volume):
    """Paired per-sample change whose expectation equals endpoint reverse-KL change.

    The target and source generator are fixed, and the adapter is an invertible
    deterministic map. The unknown base entropy and target normalizer cancel.
    Sampling uncertainty and integrability assumptions still apply.
    """
    if not math.isfinite(kT) or kT <= 0 or not math.isfinite(restraint) or restraint < 0:
        raise ValueError('Invalid physical target')
    if old_positions.ndim != 3 or new_positions.shape != old_positions.shape or old_energy.shape != (len(old_positions),) or new_energy.shape != old_energy.shape:
        raise ValueError('Mismatched paired energy and coordinate shapes')
    if not all(torch.isfinite(v).all() for v in [old_energy, new_energy, old_positions, new_positions, torch.as_tensor(log_volume)]):
        raise ValueError('Non-finite paired KL inputs')
    return (new_energy-old_energy+restraint/2*(new_positions.square().sum((1, 2))-old_positions.square().sum((1, 2))))/kT-log_volume
