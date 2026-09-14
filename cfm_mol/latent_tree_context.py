"""Invariant latent-tree information for a conditional molecular flow field.

These auxiliary edges are not chemical-bond labels. A tree-conditioned field
defines conditional, not marginal, transport; its final density is unqualified.
"""
from types import MethodType

import numpy as np
import torch
from torch import nn

from .tree_mixture_prior import weighted_tree


def tree_features(n, edges):
    """Symmetric [N,N,3]: adjacency, log path distance, mean log degree."""
    if not 1 <= n <= 200 or len(edges) != n-1:
        raise ValueError('A tree on1--200 atoms is required')
    adjacency = np.zeros((n, n), dtype=np.float64)
    for i, j in edges:
        if not 0 <= i < n or not 0 <= j < n or i == j or adjacency[i, j]:
            raise ValueError('Invalid or duplicate latent edge')
        adjacency[i, j] = adjacency[j, i] = 1.
    distance = np.full((n, n), np.inf)
    for start in range(n):
        distance[start, start] = 0.
        stack = [start]
        for parent in stack:
            for child in np.flatnonzero(adjacency[parent]):
                if not np.isfinite(distance[start, child]):
                    distance[start, child] = distance[start, parent] + 1
                    stack.append(child)
    if not np.isfinite(distance).all():
        raise ValueError('Latent edges are disconnected')
    degree = np.log1p(adjacency.sum(1))
    features = np.stack([adjacency, np.log1p(distance), .5*(degree[:, None]+degree[None, :])], -1)
    return torch.from_numpy(features)


def context_tree(prior, numbers, charge, spin, source_tree, seed, mode):
    if mode == 'actual':
        return list(source_tree)
    if mode != 'sham':
        raise ValueError('Context must be actual or sham')
    loga, _ = prior.parameters_for(numbers, charge, spin)
    return weighted_tree(loga.detach().cpu().numpy(), np.random.default_rng(seed))


def validate_context(context, graph, node_batch_idx):
    if len(context) != graph.batch_size:
        raise ValueError('One prior context matrix per graph is required')
    matrices = []
    for i, matrix in enumerate(context):
        n = int((node_batch_idx == i).sum())
        if matrix.shape != (n, n, 3) or not torch.isfinite(matrix).all():
            raise ValueError('Finite [N,N,3] prior context required')
        if not torch.allclose(matrix, matrix.transpose(0, 1), atol=1e-7, rtol=1e-7):
            raise ValueError('Undirected context must be symmetric')
        matrices.append(matrix.detach().clone())
    return matrices


def attach_context(graph, matrices, node_batch_idx):
    matrices = validate_context(matrices, graph, node_batch_idx)
    source, destination = graph.edges()
    edge_batch = node_batch_idx[source]
    if not torch.equal(edge_batch, node_batch_idx[destination]):
        raise ValueError('Context cannot cross molecular graphs')
    positions = graph.ndata['x_t'] if 'x_t' in graph.ndata else graph.ndata['x_1_true']
    features = torch.empty((len(source), 3), dtype=positions.dtype, device=graph.device)
    for i, matrix in enumerate(matrices):
        nodes = torch.where(node_batch_idx == i)[0]
        local = torch.full((graph.num_nodes(),), -1, dtype=torch.long, device=graph.device)
        local[nodes] = torch.arange(len(nodes), device=graph.device)
        chosen = edge_batch == i
        features[chosen] = matrix.to(features)[local[source[chosen]], local[destination[chosen]]]
    graph.edata['latent_tree_context'] = features


def patch_latent_tree_context(model, hidden=32):
    """Add a zero-initialized residual edge adapter without editing FlowMol."""
    field = model.vector_field
    configuration = dict(hidden=int(hidden))
    if hidden < 1:
        raise ValueError('Positive hidden width required')
    if hasattr(field, 'latent_tree_adapter'):
        if field._latent_tree_configuration != configuration:
            raise ValueError('Cannot change an installed tree adapter configuration')
        return model
    edge_width = next(layer.out_features for layer in reversed(field.edge_embedding) if isinstance(layer, nn.Linear))
    anchor = next(field.parameters())
    field.latent_tree_adapter = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, edge_width)).to(anchor)
    nn.init.zeros_(field.latent_tree_adapter[-1].weight)
    nn.init.zeros_(field.latent_tree_adapter[-1].bias)
    field._latent_tree_configuration = configuration
    original = field.denoise_graph

    def denoise(self, g, node_scalar_features, node_vec_features, node_positions,
                edge_features, node_batch_idx, upper_edge_mask, apply_softmax=False, remove_com=False):
        if 'latent_tree_context' not in g.edata:
            raise ValueError('Tree-conditioned field requires latent_tree_context')
        context = g.edata['latent_tree_context']
        if context.shape != (g.num_edges(), 3) or not torch.isfinite(context).all():
            raise ValueError('Malformed latent_tree_context')
        return original(g, node_scalar_features, node_vec_features, node_positions,
                        edge_features + self.latent_tree_adapter(context.to(edge_features)),
                        node_batch_idx, upper_edge_mask, apply_softmax, remove_com)

    field.denoise_graph = MethodType(denoise, field)
    return model
