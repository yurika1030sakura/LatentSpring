"""Calibrated scalar energy head for BGFM.

The energy head is attached to the flow backbone and produces a single
scalar per molecular configuration:

    E_psi(r, c) : R^{3N} x C -> R

It is trained jointly with the flow against an external universal
neural potential (OMol25) via energy regression and force matching:

    L_head = E |E_psi(r, c) - E_NP(r, c)|
           + lambda_F * E ||-grad_r E_psi(r, c) - F_NP(r, c)||^2

At inference time the energy head provides:
  (i)  a calibrated scalar landscape for the Langevin corrector
       (see cfm_mol.refinement), and
  (ii) a cross-model scorer for ranking generated samples.

Architecture
------------

The head is an invariant readout that consumes the final scalar
features produced by the FlowMol3 GVP-Transformer backbone, scatter-
sums to a per-graph representation, and applies a small MLP to a
scalar. Gradients with respect to coordinates are obtained via
torch.autograd so that the force term in L_head is well-defined
without a separate force head.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class EnergyHead(nn.Module):
    """Scalar energy head over molecular configurations."""

    def __init__(
        self,
        n_scalar_features: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        act = {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}[activation]
        layers: list[nn.Module] = []
        last = n_scalar_features
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(last, hidden_dim))
            layers.append(act())
            last = hidden_dim
        layers.append(nn.Linear(last, 1))
        self.mlp = nn.Sequential(*layers)
        # Zero-init the final projection so the head adds no signal
        # at the start of training; the calibration loss drives it
        # toward the neural-potential energy.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        node_scalar_features: torch.Tensor,
        node_batch_idx: torch.Tensor,
        n_graphs: int,
    ) -> torch.Tensor:
        """Produce one scalar energy per graph.

        Args:
            node_scalar_features: (N_total, F) per-atom scalar features
                produced by the backbone after its final update layer.
            node_batch_idx: (N_total,) graph index for each atom.
            n_graphs: number of graphs in the batch.

        Returns:
            (n_graphs,) tensor of predicted energies.
        """
        per_atom = self.mlp(node_scalar_features).squeeze(-1)  # (N_total,)
        out = torch.zeros(n_graphs, device=per_atom.device, dtype=per_atom.dtype)
        out.scatter_add_(0, node_batch_idx, per_atom)
        return out


def energy_and_force(
    head: EnergyHead,
    backbone_forward_fn,
    positions: torch.Tensor,
    node_batch_idx: torch.Tensor,
    n_graphs: int,
    create_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (E_psi(r, c), -grad_r E_psi(r, c)) jointly.

    The backbone_forward_fn(positions) returns per-atom scalar
    features for the current discrete state c (which is supplied by
    the caller via closure). Gradients are taken with respect to
    `positions`.

    Args:
        head: the EnergyHead module.
        backbone_forward_fn: callable that takes positions and returns
            per-atom scalar features (N_total, F).
        positions: (N_total, 3) coordinates; must be a leaf with
            requires_grad=True.
        node_batch_idx: (N_total,) graph index per atom.
        n_graphs: number of graphs.
        create_graph: if True, build a graph for higher-order grads
            (needed during training for the L_head force term).

    Returns:
        E_pred: (n_graphs,) scalar energies.
        F_pred: (N_total, 3) negative-gradient forces.
    """
    if not positions.requires_grad:
        positions = positions.detach().requires_grad_(True)
    feats = backbone_forward_fn(positions)
    E_pred = head(feats, node_batch_idx, n_graphs)
    grad, = torch.autograd.grad(
        outputs=E_pred.sum(),
        inputs=positions,
        create_graph=create_graph,
        retain_graph=create_graph,
    )
    F_pred = -grad
    return E_pred, F_pred
