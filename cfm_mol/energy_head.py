"""Calibrated scalar energy head for BGFM.

The energy head produces a single scalar per molecular configuration:

    E_psi(r, c) : R^{3N} x C -> R

It is trained jointly with the flow against an external universal
neural potential (OMol25) via energy regression and force matching:

    L_head = E |E_psi(r, c) - E_NP(r, c)|
           + lambda_F * E ||-grad_r E_psi(r, c) - F_NP(r, c)||^2.

At inference time the energy head provides:
  (i)  a calibrated scalar landscape for the Langevin corrector
       (see cfm_mol.refinement); and
  (ii) a cross-model scorer for ranking generated samples.

Design
------

We implement the head as a self-contained small invariant network
that consumes the molecular configuration directly --- it does not
share weights with the flow backbone. This keeps the head's gradient
well-defined for the Langevin corrector without forcing the backbone
to be re-evaluated at arbitrary perturbed coordinates. The head uses:

  - per-atom token embeddings of atom type and formal charge;
  - pairwise distance features between every pair of atoms within a
    cutoff;
  - a small invariant aggregator (per-atom MLP -> sum per graph);
  - a final scalar MLP to a single energy.

The architecture is ~200k--500k parameters at the recommended sizes,
compared with the ~6M-parameter FlowMol3 backbone. It is therefore
cheap to evaluate inside the Langevin corrector loop.

Two helpers are exposed:

  EnergyHead.forward(positions, atom_types, charges, node_batch_idx)
      -> (n_graphs,) energy per graph

  energy_and_force(head, positions, atom_types, charges,
                   node_batch_idx, n_graphs, create_graph=False)
      -> (E, F) where F = -grad_r E
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def _expand_atomwise_pairs(node_batch_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (src, dst) atom-index pairs for all intra-graph pairs.

    For each graph, all ordered pairs (i, j) with i != j inside that
    graph are returned. The number of pairs scales as O(sum N_g^2),
    which is fine for the molecule sizes used here (< 200 atoms).
    """
    n_total = node_batch_idx.shape[0]
    src = torch.arange(n_total, device=node_batch_idx.device).repeat_interleave(n_total)
    dst = torch.arange(n_total, device=node_batch_idx.device).repeat(n_total)
    same_graph = (node_batch_idx[src] == node_batch_idx[dst]) & (src != dst)
    return src[same_graph], dst[same_graph]


class EnergyHead(nn.Module):
    """Invariant scalar energy head over molecular configurations."""

    def __init__(
        self,
        n_atom_types: int = 83,
        n_charge_classes: int = 6,
        hidden_dim: int = 128,
        n_rbf: int = 32,
        cutoff: float = 5.0,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        self.n_rbf = int(n_rbf)
        self.atom_emb = nn.Embedding(n_atom_types + 4, hidden_dim)
        self.charge_emb = nn.Embedding(n_charge_classes + 4, hidden_dim)
        # RBF parameters for pairwise distances.
        rbf_centers = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("rbf_centers", rbf_centers)
        self.rbf_gamma = 1.0 / max((cutoff / n_rbf) ** 2, 1e-6)
        # Pair-feature MLP: (RBF_dim + src_h + dst_h) -> hidden
        self.pair_mlp = nn.Sequential(
            nn.Linear(n_rbf + 2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Per-atom update layers (scatter sum of pair messages -> MLP).
        update_layers: list[nn.Module] = []
        for _ in range(n_layers - 1):
            update_layers.append(nn.Linear(hidden_dim, hidden_dim))
            update_layers.append(nn.SiLU())
        update_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.update_mlp = nn.Sequential(*update_layers)
        # Per-atom readout to scalar.
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Zero-init readout so the head adds no signal at init; the
        # calibration loss drives it toward the neural-potential
        # energy during training.
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def _rbf(self, dist: torch.Tensor) -> torch.Tensor:
        # dist: (n_pairs,); returns (n_pairs, n_rbf)
        diff = dist.unsqueeze(-1) - self.rbf_centers.unsqueeze(0)
        return torch.exp(-self.rbf_gamma * diff * diff)

    def forward(
        self,
        positions: torch.Tensor,
        atom_types: torch.Tensor,
        charges: torch.Tensor,
        node_batch_idx: torch.Tensor,
        n_graphs: int,
    ) -> torch.Tensor:
        """Return (n_graphs,) energy per graph.

        Args:
            positions: (N_total, 3) coordinates.
            atom_types: (N_total,) integer type indices.
            charges: (N_total,) integer charge indices (e.g. 0..5 with
                offset).
            node_batch_idx: (N_total,) graph index per atom.
            n_graphs: number of graphs in the batch.
        """
        h = self.atom_emb(atom_types.long()) + self.charge_emb(charges.long())
        src, dst = _expand_atomwise_pairs(node_batch_idx)
        if src.numel() == 0:
            return torch.zeros(n_graphs, device=positions.device, dtype=positions.dtype)
        diff = positions[dst] - positions[src]
        dist = diff.norm(dim=-1)
        mask = dist < self.cutoff
        src, dst, dist = src[mask], dst[mask], dist[mask]
        if src.numel() == 0:
            return torch.zeros(n_graphs, device=positions.device, dtype=positions.dtype)
        rbf = self._rbf(dist)
        pair_in = torch.cat([rbf, h[src], h[dst]], dim=-1)
        pair_msg = self.pair_mlp(pair_in)
        # Scatter-sum pair messages onto destination atoms.
        msg_sum = torch.zeros_like(h)
        msg_sum.index_add_(0, dst, pair_msg)
        h2 = self.update_mlp(h + msg_sum)
        per_atom_E = self.readout(h2).squeeze(-1)
        out = torch.zeros(n_graphs, device=positions.device, dtype=positions.dtype)
        out.index_add_(0, node_batch_idx, per_atom_E)
        return out


def energy_and_force(
    head: EnergyHead,
    positions: torch.Tensor,
    atom_types: torch.Tensor,
    charges: torch.Tensor,
    node_batch_idx: torch.Tensor,
    n_graphs: int,
    create_graph: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute (E_psi(r, c), -grad_r E_psi(r, c)) jointly.

    Args:
        head: EnergyHead.
        positions: (N_total, 3) coordinates; will be detached and
            re-marked requires_grad=True so that autograd can
            compute the force.
        atom_types: (N_total,) integer type indices.
        charges: (N_total,) integer charge indices.
        node_batch_idx: (N_total,) graph index per atom.
        n_graphs: number of graphs in the batch.
        create_graph: pass True during training so the force-matching
            term can backpropagate; False at inference inside the
            Langevin corrector.

    Returns:
        E_pred: (n_graphs,) scalar energies.
        F_pred: (N_total, 3) forces (-grad_r E_pred).
    """
    r = positions.detach().requires_grad_(True)
    E_pred = head(r, atom_types, charges, node_batch_idx, n_graphs)
    (grad,) = torch.autograd.grad(
        outputs=E_pred.sum(),
        inputs=r,
        create_graph=create_graph,
        retain_graph=create_graph,
    )
    F_pred = -grad
    return E_pred, F_pred
