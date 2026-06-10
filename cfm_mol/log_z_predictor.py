"""Per-molecule log-Z auxiliary head for the BGFM anchor loss.

Predicts the per-molecule normalization constant log Z_m from INVARIANT
features only (atom-type counts, total charge, n_atoms). Deliberately
under-parameterized so it cannot trivially absorb arbitrary per-molecule
offsets -- this is what breaks the trivial-constant failure mode of the
variance-only L_energy.

Architecture (~few k params, vs ~6M in the main flow model):
  per_atom_contrib  = per-element scalar embedding (n_atom_types -> 1)
  graph_features    = [sum per-atom contribs, n_atoms, total_charge, |total_charge|]
  log_Z_pred        = small_mlp(graph_features), scalar per graph

The atomic-energy form is motivated by atomization-energy decompositions
in NN potentials (Behler-Parrinello): total energy ~ sum of per-atom
contributions + small corrections. log Z is dominated by the bulk
"chemistry" of which elements are present and how many.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LogZPredictor(nn.Module):
    """Permutation-invariant per-molecule scalar predictor.

    Args:
        n_atom_types: number of distinct atom-type indices (e.g. 83 for OMol25
            without the ctmc mask channel)
        hidden_dim: MLP hidden width (default 16; intentionally small)
    """

    def __init__(self, n_atom_types: int, hidden_dim: int = 16):
        super().__init__()
        self.n_atom_types = n_atom_types
        # Per-element scalar contribution (like Behler-Parrinello atomic energy)
        self.elem_e = nn.Embedding(n_atom_types, 1)
        nn.init.zeros_(self.elem_e.weight)  # start at 0 so log_Z_pred ~ 0 at init
        # Small correction from (n_atoms, total_charge, |total_charge|)
        self.corr = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize correction to zero output so log_Z_pred starts at 0
        nn.init.zeros_(self.corr[-1].weight)
        nn.init.zeros_(self.corr[-1].bias)

    def forward(
        self,
        atom_type_idx: torch.Tensor,
        node_batch_idx: torch.Tensor,
        atom_charges_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return (B,) log Z prediction per graph.

        Args:
            atom_type_idx: (N_total,) int atom-type indices (0..n_atom_types-1)
            node_batch_idx: (N_total,) which graph each atom belongs to
            atom_charges_raw: (N_total,) int per-atom charge. The preprocess
                puts total molecular charge on atom 0, so summing recovers it.
                If None, treated as zero charge per atom.
        """
        device = atom_type_idx.device
        # Clamp out-of-range (e.g. ctmc mask class) — those won't appear at
        # data time but are defensive.
        valid_mask = (atom_type_idx >= 0) & (atom_type_idx < self.n_atom_types)
        safe_idx = torch.where(valid_mask, atom_type_idx,
                               torch.zeros_like(atom_type_idx))
        per_atom = self.elem_e(safe_idx).squeeze(-1) * valid_mask.to(self.elem_e.weight.dtype)

        B = int(node_batch_idx.max().item()) + 1 if node_batch_idx.numel() else 0
        per_graph_sum = torch.zeros(B, device=device, dtype=per_atom.dtype)
        per_graph_sum.scatter_add_(0, node_batch_idx, per_atom)

        n_atoms_per_graph = torch.zeros(B, device=device, dtype=per_atom.dtype)
        n_atoms_per_graph.scatter_add_(0, node_batch_idx,
                                       torch.ones_like(per_atom))

        if atom_charges_raw is None:
            total_charge = torch.zeros(B, device=device, dtype=per_atom.dtype)
        else:
            total_charge = torch.zeros(B, device=device, dtype=per_atom.dtype)
            total_charge.scatter_add_(0, node_batch_idx,
                                      atom_charges_raw.to(per_atom.dtype))

        feats = torch.stack(
            [n_atoms_per_graph, total_charge, total_charge.abs()], dim=-1
        )
        correction = self.corr(feats).squeeze(-1)
        return per_graph_sum + correction
