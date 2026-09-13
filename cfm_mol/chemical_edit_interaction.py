"""Commuting terminal edits and their four-state interaction work.

Composition of local moves is established prior art. These primitives isolate
non-additive physical work; they are not a learned sampler or reaction pathway.
"""
import torch
from cfm_mol.chemical_moves import exchange_terminal_sites
from cfm_mol.chemical_path_guide import exchanged_bond_graph


def independent_actions(a, b):
    """Moved atoms are disjoint and cannot be passive anchors of the other edit."""
    return not (set(a[:2]) & set(b) or set(b[:2]) & set(a))


def four_positions(x, radii, a, b):
    if not independent_actions(a, b):
        raise ValueError('Four-state commuting square requires independent moved atoms')
    xa, ja, ia = exchange_terminal_sites(x, radii, a)
    xb, jb, ib = exchange_terminal_sites(x, radii, b)
    xab, jab, _ = exchange_terminal_sites(xa, radii, b)
    xba, jba, _ = exchange_terminal_sites(xb, radii, a)
    torch.testing.assert_close(xab, xba, atol=1e-10, rtol=0)
    torch.testing.assert_close(ja+jb, jab+jba, atol=1e-12, rtol=0)
    return torch.stack([x, xa, xb, xab]), ja+jb, (ia, ib)


def four_graphs(bonds, a, b):
    if not independent_actions(a, b):
        raise ValueError('Overlapping chemical edits')
    ga = exchanged_bond_graph(bonds, a); gb = exchanged_bond_graph(bonds, b)
    gab = exchanged_bond_graph(ga, b)
    assert torch.equal(gab, exchanged_bond_graph(gb, a))
    return torch.stack([bonds, ga, gb, gab])


def mixed_difference(values):
    """Four-state order is original, A, B, AB; last axis enumerates states."""
    if values.shape[-1] != 4:
        raise ValueError('Exactly four corner values required')
    return values[..., 3]-values[..., 1]-values[..., 2]+values[..., 0]


def restraint_interaction(positions, restraint=.1):
    """Known COM-restraint contribution, separate from electronic interaction."""
    centered = positions-positions.mean(-2, keepdim=True)
    return mixed_difference(restraint/2*centered.square().sum((-2, -1)))
