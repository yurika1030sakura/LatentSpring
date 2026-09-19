"""DGL-free diagnostic harness for existing tensor-only repository components.

TensorGraph is NOT a replacement DGL implementation or a FlowMol validation.
It implements only the immutable complete-graph interface used by clamped_fm,
clamped_density and the repository's RadialPairReference diagnostic backbone.
"""
from __future__ import annotations
from contextlib import contextmanager
from types import SimpleNamespace
import torch
from torch import nn
from .radial_reference import RadialPairReference


class TensorGraph:
    def __init__(self, draws, atom_numbers=(1, 6, 7, 8), dtype=torch.float32):
        self.batch_size = len(draws)
        self._counts = torch.tensor([len(r.positions) for r in draws])
        self.nbi = torch.repeat_interleave(torch.arange(len(draws)), self._counts)
        xyz = torch.cat([torch.tensor(r.positions, dtype=dtype) for r in draws])
        z = [n for r in draws for n in r.condition['numbers']]
        atom = nn.functional.one_hot(torch.tensor([atom_numbers.index(n) for n in z]), len(atom_numbers)).to(dtype)
        charge = nn.functional.one_hot(torch.full((len(z),), 2, dtype=torch.long), 6).to(dtype)
        src, dst, offset = [], [], 0
        for n in self._counts.tolist():
            upper = torch.triu_indices(n, n, 1)
            both = torch.cat([upper, upper.flip(0)], dim=1) + offset
            src.append(both[0]); dst.append(both[1]); offset += n
        self._edges = (torch.cat(src), torch.cat(dst))
        edge = nn.functional.one_hot(torch.zeros(len(self._edges[0]), dtype=torch.long), 4).to(dtype)
        self.ndata = dict(x_1_true=xyz, x_t=xyz.clone(), a_1_true=atom, a_t=atom.clone(),
            c_1_true=charge, c_t=charge.clone(), has_reference_geometry=torch.ones((len(z), 1), dtype=torch.bool))
        self.edata = dict(e_1_true=edge, e_t=edge.clone())
    def edges(self): return self._edges
    def num_nodes(self): return int(self._counts.sum())
    def num_edges(self): return len(self._edges[0])
    def batch_num_nodes(self): return self._counts
    @contextmanager
    def local_scope(self):
        n, e = self.ndata.copy(), self.edata.copy()
        try: yield self
        finally: self.ndata, self.edata = n, e


class LinearSchedule:
    feats = ['x', 'a', 'c', 'e']
    def alpha_t(self, t): return t[:, None].expand(-1, 4)
    def alpha_t_prime(self, t): return torch.ones_like(t)[:, None].expand(-1, 4)


class NativeRadialHead(nn.Module):
    """Thin displacement-head wrapper, not the 5.9M FlowMol backbone."""
    def __init__(self):
        super().__init__()
        self.reference = RadialPairReference(4, 6)
        self.interpolant_scheduler = LinearSchedule()
    def forward(self, graph, t, node_batch_idx, **kwargs):
        x = graph.ndata['x_t']
        velocity, _ = self.reference.velocity_and_divergence(graph, x, t, node_batch_idx)
        return {'x': x + velocity}


def radial_model():
    return SimpleNamespace(vector_field=NativeRadialHead(), _research_prior_kind='harmonic_tree')
