"""Thin adapter from weighted endpoints to the existing LatentSpring FM loss.

No new loss, hidden graph input, per-minibatch weight normalization, replay mix,
or physical-minus-replay parameter arithmetic. DGL is imported only at the
production graph boundary; diagnostic tests can exercise the loss on tensors.
"""
from __future__ import annotations
from typing import Any
import numpy as np
import torch
from .clamped_fm import clamped_fm_loss
from .chemical_moves import covalent_radii
from .weighted_endpoints import EndpointDraw


def checkpoint_source(prior: Any, draws: list[EndpointDraw], seed: int) -> torch.Tensor:
    """Sample the EXACT saved prior object (including Gaussian controls)."""
    result = []
    for j, draw in enumerate(draws):
        c = draw.condition
        g = torch.Generator().manual_seed(seed + j)
        if prior is None:
            x = torch.randn((c['n_atoms'], 3), dtype=torch.float64, generator=g)
            x -= x.mean(0)
        else:
            x, _ = prior.sample(c['numbers'], c['charge'], c['spin_multiplicity'],
                rng=np.random.default_rng(seed + j + 700000001), generator=g)
        result.append(x.detach().cpu())
    return torch.cat(result)


def dgl_batch(draws: list[EndpointDraw], config: dict, device: torch.device,
              *, model_kT_eV: float = 1.0):
    """Construct native, no-bond graphs. A fixed feature avoids a T confound.

    First-stage adaptation has one teacher temperature. ``model_kT_eV=1``
    deliberately preserves the historical frozen temperature FEATURE; it is
    not the teacher's physical kBT. Both values are recorded by the driver.
    Multi-temperature generation is not claimed by this integration.
    """
    import dgl
    from .condition_systems import graph_from_condition
    from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask
    graphs = []
    for row in draws:
        condition = {**row.condition, 'requested_kT_eV': model_kT_eV}
        graph = graph_from_condition(condition, config['dataset']['atom_map'], device=device,
            n_bond_classes=5 if config['mol_fm'].get('explicit_aromaticity', False) else 4)
        graph.ndata['x_1_true'] = torch.tensor(row.positions, dtype=torch.float32, device=device)
        graph.ndata['has_reference_geometry'] = torch.ones((len(row.positions), 1),
                                                         dtype=torch.bool, device=device)
        graphs.append(graph)
    graph = dgl.batch(graphs)
    return graph, get_batch_idxs(graph)[0], get_upper_edge_mask(graph)


def endpoint_fm_loss(model: Any, graph: Any, nbi: torch.Tensor, uem: torch.Tensor,
                     draws: list[EndpointDraw], prior: Any, seed: int) -> torch.Tensor:
    """Endpoints were already drawn with their weights: DO NOT weight twice."""
    reference = graph.ndata['x_1_true']
    numbers = [z for draw in draws for z in draw.condition['numbers']]
    if len(numbers) != len(reference):
        raise ValueError('Endpoint batch and graph disagree')
    x0 = checkpoint_source(prior, draws, seed)
    generator = torch.Generator(device=reference.device).manual_seed(seed + 100000003)
    pairing_generator = torch.Generator(device=reference.device).manual_seed(seed + 200000003)
    return clamped_fm_loss(model, graph, nbi, uem, terminal_time=1.,
        parameterization='displacement', prior_positions=x0,
        pairing='typed_rotation', pairing_radii=covalent_radii(numbers,
            device=reference.device, dtype=reference.dtype),
        generator=generator, pairing_generator=pairing_generator)
