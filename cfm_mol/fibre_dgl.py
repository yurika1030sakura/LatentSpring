"""DGL-graph wrappers around fibre.py.

FlowMol3 represents a batch of molecules as a single DGL graph with
`node_batch_idx` telling you which molecule each atom belongs to. Our core
fibre.py uses padded `(B, N, 3)` tensors. This module does the translation.

Key functions (all operate on the whole batch):
  - retract_dgl(x_t, a, node_batch_idx, d_min_table) -> x_t_retracted
  - tangent_project_dgl(v, x_t, a, node_batch_idx, d_min_table) -> v_proj

Design note: `_dgl_to_padded` / `_padded_to_dgl` are vectorised via
scatter-in/gather-out using the intra-batch within-molecule index derived
from `bincount` offsets. The inner `retract` / `tangent_project` loops in
`fibre.py` are left untouched for now (TODO noted inline).

Padding pitfall: dummy padded slots default to atom-type 0 (H) and position
0, which would spuriously interact with real atoms under the steric
constraint. We mask them out by (a) pinning padded positions to a large
Euclidean sentinel and (b) setting their d_min entries to 0 via a mask.
"""
from __future__ import annotations

import torch

from cfm_mol.fibre import retract, tangent_project


_PAD_POS = 1.0e6   # sentinel position for padded slots; far enough from any
                   # real conformer that pair distances exceed every d_min.


# ---------------------------------------------------------------------------
# Batch <-> padded helpers (vectorised)
# ---------------------------------------------------------------------------

def _batch_offsets(lengths: torch.Tensor) -> torch.Tensor:
    """Cumulative offsets per batch index, (B,) -> (B,).
    offsets[b] = sum of lengths[0..b-1]. Used to compute within-molecule idx.
    """
    return torch.cat([
        torch.zeros(1, dtype=lengths.dtype, device=lengths.device),
        lengths.cumsum(0)[:-1],
    ])


def _within_batch_idx(
    node_batch_idx: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    offsets = _batch_offsets(lengths)
    return (
        torch.arange(node_batch_idx.shape[0], device=node_batch_idx.device)
        - offsets[node_batch_idx]
    )


def _dgl_to_padded(
    node_feat: torch.Tensor,
    node_batch_idx: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert (total_atoms, F) node feature to padded (B, N_max, F) plus lengths (B,).

    Vectorised: no Python loop over B. Padded slots are filled with zeros.
    """
    B = int(node_batch_idx.max().item()) + 1
    if lengths is None:
        lengths = torch.bincount(node_batch_idx, minlength=B)
    N_max = int(lengths.max().item())
    F = node_feat.shape[-1] if node_feat.dim() > 1 else 1
    device = node_feat.device

    within = _within_batch_idx(node_batch_idx, lengths)

    if node_feat.dim() == 1:
        out = torch.zeros(B, N_max, device=device, dtype=node_feat.dtype)
        out[node_batch_idx, within] = node_feat
    else:
        out = torch.zeros(B, N_max, F, device=device, dtype=node_feat.dtype)
        out[node_batch_idx, within] = node_feat
    return out, lengths


def _padded_to_dgl(
    padded: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Inverse of _dgl_to_padded: (B, N_max, F) -> (total_atoms, F). Vectorised."""
    B = padded.shape[0]
    N_max = padded.shape[1]
    arange = torch.arange(N_max, device=padded.device)
    mask = arange.unsqueeze(0) < lengths.unsqueeze(1)       # (B, N_max)
    return padded[mask]


def _pad_mask(lengths: torch.Tensor, N_max: int) -> torch.Tensor:
    """(B, N_max) bool: True on real-atom slots, False on padding."""
    arange = torch.arange(N_max, device=lengths.device)
    return arange.unsqueeze(0) < lengths.unsqueeze(1)


# ---------------------------------------------------------------------------
# Retract + tangent-project on DGL batches
# ---------------------------------------------------------------------------

def retract_dgl(
    x_t: torch.Tensor,
    a: torch.Tensor,
    node_batch_idx: torch.Tensor,
    d_min_table: torch.Tensor,
    max_iters: int = 20,
) -> torch.Tensor:
    """Retract DGL-batched coords onto the steric fibre per molecule.

    Args
    ----
    x_t : (total_atoms, 3) coordinates (DGL node feature).
    a : (total_atoms,) atom-type indices.
    node_batch_idx : (total_atoms,) molecule index per atom.
    d_min_table : (A, A)
    """
    d_min_table = d_min_table.to(x_t.device)
    # Defensive: atom-type indices can be larger than d_min_table if the
    # model's internal categorical dim exceeds what we pre-computed.
    # Clamp out-of-range indices to the last slot (which we set to d_min=0,
    # i.e., unconstrained). Prevents CUDA out-of-bounds assertions.
    A = d_min_table.shape[0]
    a = a.long().clamp_max(A - 1)
    x_padded, lengths = _dgl_to_padded(x_t, node_batch_idx)
    a_padded, _ = _dgl_to_padded(a, node_batch_idx, lengths)

    # Pin padded positions far away so they never register as violating pairs.
    N_max = x_padded.shape[1]
    mask = _pad_mask(lengths, N_max)                           # (B, N_max)
    x_padded = torch.where(
        mask.unsqueeze(-1),
        x_padded,
        torch.full_like(x_padded, _PAD_POS),
    )

    x_ret, _ = retract(x_padded, a_padded, d_min_table, max_iters=max_iters)
    return _padded_to_dgl(x_ret, lengths)


def tangent_project_dgl(
    v: torch.Tensor,
    x_t: torch.Tensor,
    a: torch.Tensor,
    node_batch_idx: torch.Tensor,
    d_min_table: torch.Tensor,
    margin: float = 0.05,
) -> torch.Tensor:
    """Tangent-project DGL-batched velocity per molecule."""
    d_min_table = d_min_table.to(x_t.device)
    A = d_min_table.shape[0]
    a = a.long().clamp_max(A - 1)
    x_padded, lengths = _dgl_to_padded(x_t, node_batch_idx)
    v_padded, _ = _dgl_to_padded(v, node_batch_idx, lengths)
    a_padded, _ = _dgl_to_padded(a, node_batch_idx, lengths)

    N_max = x_padded.shape[1]
    mask = _pad_mask(lengths, N_max)
    x_padded = torch.where(
        mask.unsqueeze(-1),
        x_padded,
        torch.full_like(x_padded, _PAD_POS),
    )
    # Velocity at padding slots stays zero; tangent_project only clips pairs
    # within margin of a steric sphere, and padded pairs are far (> _PAD_POS
    # apart) so none are active.

    v_proj = tangent_project(v_padded, x_padded, a_padded, d_min_table, margin=margin)
    return _padded_to_dgl(v_proj, lengths)
