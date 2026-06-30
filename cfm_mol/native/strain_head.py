"""Composition-conditioned residual strain energy head for BGFM-Native.

The central design choice is to avoid regressing absolute molecular
energies directly.  Absolute OMol25 energies are strongly dominated by
atom count, element composition, and total charge.  For generation and
refinement we need the coordinate-dependent *strain* landscape.  The
head therefore decomposes

    E_hat(r, c) = b_phi(c) + DeltaU_psi(r, c),

where b_phi(c) is a permutation-invariant composition baseline and
DeltaU_psi(r, c) is an invariant coordinate-dependent residual.  Forces
for the drift/corrector are computed from DeltaU only:

    F_psi(r, c) = - grad_r DeltaU_psi(r, c).

This module is intentionally self-contained: it consumes positions,
atom-type indices, charge classes/raw charges, and graph membership; it
can be evaluated at arbitrary perturbed coordinates during Langevin
correction without requiring the FlowMol backbone to be re-run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pair_indices(node_batch_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n = int(node_batch_idx.numel())
    if n == 0:
        z = torch.empty(0, dtype=torch.long, device=node_batch_idx.device)
        return z, z
    src = torch.arange(n, device=node_batch_idx.device).repeat_interleave(n)
    dst = torch.arange(n, device=node_batch_idx.device).repeat(n)
    keep = (node_batch_idx[src] == node_batch_idx[dst]) & (src != dst)
    return src[keep], dst[keep]


def _safe_charge_class(charges: torch.Tensor, n_charge_classes: int) -> torch.Tensor:
    """Map raw or one-hot charge-like tensors to class indices.

    Existing FlowMol batches usually store charge classes as one-hot c_t or
    c_1_true with 6 classes corresponding to raw charge + 2.  Perturbation
    shards often store raw integer charges.  This helper accepts either.
    """
    if charges.dim() > 1:
        charges = charges.argmax(dim=-1)
    charges = charges.long()
    # If values look raw, shift by +2 into the 0..5 convention.  If they
    # already look like class indices, the clamp leaves them alone.
    if charges.numel() and int(charges.min().item()) < 0:
        charges = charges + 2
    return charges.clamp(0, n_charge_classes - 1)


class CompositionBaseline(nn.Module):
    """Invariant baseline b_phi(c) from atom histogram + total charge."""

    def __init__(self, n_atom_types: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.n_atom_types = int(n_atom_types)
        self.net = nn.Sequential(
            nn.Linear(n_atom_types + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        atom_types: torch.Tensor,
        charges: torch.Tensor,
        node_batch_idx: torch.Tensor,
        n_graphs: int,
    ) -> torch.Tensor:
        h = torch.zeros(
            n_graphs, self.n_atom_types,
            dtype=torch.float32, device=atom_types.device,
        )
        at = atom_types.long().clamp(0, self.n_atom_types - 1)
        h.index_add_(0, node_batch_idx, F.one_hot(at, self.n_atom_types).float())
        # total charge and atom count are useful scale controls.
        q = charges.float()
        if q.dim() > 1:
            # one-hot class -> raw-like charge centered at zero
            q = q.argmax(dim=-1).float() - 2.0
        q_total = torch.zeros(n_graphs, 1, device=atom_types.device)
        q_total.index_add_(0, node_batch_idx, q.unsqueeze(-1))
        n_atoms = torch.zeros(n_graphs, 1, device=atom_types.device)
        n_atoms.index_add_(0, node_batch_idx, torch.ones_like(q_total[node_batch_idx]))
        feat = torch.cat([h, q_total, n_atoms], dim=-1)
        return self.net(feat).squeeze(-1)


class ResidualStrainEnergyHead(nn.Module):
    """Invariant residual strain scorer DeltaU_psi plus baseline b_phi."""

    def __init__(
        self,
        n_atom_types: int = 83,
        n_charge_classes: int = 6,
        hidden_dim: int = 192,
        n_rbf: int = 32,
        cutoff: float = 5.0,
        n_layers: int = 3,
        baseline_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.n_atom_types = int(n_atom_types)
        self.n_charge_classes = int(n_charge_classes)
        self.hidden_dim = int(hidden_dim)
        self.cutoff = float(cutoff)
        self.n_rbf = int(n_rbf)

        self.atom_emb = nn.Embedding(n_atom_types + 4, hidden_dim)
        self.charge_emb = nn.Embedding(n_charge_classes + 4, hidden_dim)
        centers = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("rbf_centers", centers)
        self.rbf_gamma = 1.0 / max((cutoff / max(n_rbf - 1, 1)) ** 2, 1e-6)

        self.pair_mlp = nn.Sequential(
            nn.Linear(n_rbf + 2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        update = []
        for _ in range(max(n_layers - 1, 1)):
            update += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        update.append(nn.Linear(hidden_dim, hidden_dim))
        self.update_mlp = nn.Sequential(*update)
        self.strain_readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.strain_readout[-1].weight)
        nn.init.zeros_(self.strain_readout[-1].bias)
        self.baseline = CompositionBaseline(n_atom_types, baseline_hidden_dim)

    def _rbf(self, dist: torch.Tensor) -> torch.Tensor:
        diff = dist.unsqueeze(-1) - self.rbf_centers.unsqueeze(0)
        return torch.exp(-self.rbf_gamma * diff * diff)

    def forward_parts(
        self,
        positions: torch.Tensor,
        atom_types: torch.Tensor,
        charges: torch.Tensor,
        node_batch_idx: torch.Tensor,
        n_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (E_total, DeltaU, baseline), each shaped (n_graphs,)."""
        # Force fp32 throughout: autocast (bf16-mixed in our trainer config)
        # routes nn.Linear ops to bf16, which then mismatches the fp32
        # accumulators allocated by torch.zeros_like(...) on embedding
        # outputs. The strain head is ~500k params so fp32 is trivial,
        # and autograd through it for force computation is more accurate
        # in fp32 anyway. Re-cast back to positions.dtype on return.
        # NOTE: autocast is harmless on CPU runs (no-op).
        from contextlib import nullcontext
        if torch.cuda.is_available() and positions.is_cuda:
            ctx = torch.cuda.amp.autocast(enabled=False)
        else:
            ctx = nullcontext()
        with ctx:
            at = atom_types.long().clamp(0, self.n_atom_types - 1)
            ch = _safe_charge_class(charges, self.n_charge_classes)
            h = (self.atom_emb(at) + self.charge_emb(ch)).float()
            src, dst = _pair_indices(node_batch_idx)
            if src.numel() > 0:
                d = (positions.float()[dst] - positions.float()[src]).norm(dim=-1)
                keep = d < self.cutoff
                src, dst, d = src[keep], dst[keep], d[keep]
                if src.numel() > 0:
                    pair = torch.cat([self._rbf(d), h[src], h[dst]], dim=-1)
                    msg = self.pair_mlp(pair).to(h.dtype)
                    acc = torch.zeros_like(h)
                    acc.index_add_(0, dst, msg)
                    h = self.update_mlp(h + acc)
            per_atom = self.strain_readout(h).squeeze(-1).to(positions.dtype)
            strain = torch.zeros(n_graphs, dtype=positions.dtype, device=positions.device)
            strain.index_add_(0, node_batch_idx, per_atom)
            base = self.baseline(at, charges, node_batch_idx, n_graphs).to(dtype=positions.dtype)
        return base + strain, strain, base

    def forward(
        self,
        positions: torch.Tensor,
        atom_types: torch.Tensor,
        charges: torch.Tensor,
        node_batch_idx: torch.Tensor,
        n_graphs: int,
    ) -> torch.Tensor:
        total, _, _ = self.forward_parts(positions, atom_types, charges, node_batch_idx, n_graphs)
        return total


def residual_energy_and_force(
    head: ResidualStrainEnergyHead,
    positions: torch.Tensor,
    atom_types: torch.Tensor,
    charges: torch.Tensor,
    node_batch_idx: torch.Tensor,
    n_graphs: int,
    create_graph: bool = False,
    detach_positions: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (E_total, DeltaU, baseline, force_from_DeltaU).

    The returned force is -grad_r DeltaU, not -grad_r E_total.  Since
    the baseline is composition-only, both are equivalent in exact
    arithmetic, but using DeltaU documents the intended strain-based
    correction.
    """
    # Force computation needs autograd through positions even when the
    # outer caller is inside torch.no_grad() (validation / inference). A
    # bare .requires_grad_(True) inside no_grad mode does NOT enable
    # tracking; we have to also enter torch.enable_grad(). The drift in
    # forward_with_energy_drift adds this term to the velocity output at
    # both train and val time, so we always need the force value
    # regardless of the outer grad state.
    with torch.enable_grad():
        r = positions.detach().requires_grad_(True) if detach_positions else positions.requires_grad_(True)
        total, strain, base = head.forward_parts(r, atom_types, charges, node_batch_idx, n_graphs)
        (grad,) = torch.autograd.grad(
            outputs=strain.sum(),
            inputs=r,
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=False,
        )
    force = -grad
    return total, strain, base, force


@dataclass
class HeadLossDiagnostics:
    energy_mae_total: float
    energy_mae_residual: float
    force_mse: float
    force_cosine: float


def residual_head_loss(
    head: ResidualStrainEnergyHead,
    positions: torch.Tensor,
    atom_types: torch.Tensor,
    charges: torch.Tensor,
    node_batch_idx: torch.Tensor,
    n_graphs: int,
    energy_target: torch.Tensor,
    force_target: torch.Tensor,
    lambda_force: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Calibration loss for the residual strain head.

    The energy residual target is E_NP - stopgrad(b_phi(c)).  This avoids
    forcing DeltaU to explain composition/size offsets.  The total energy
    error is logged but not the only supervision signal.
    """
    total, strain, base, force = residual_energy_and_force(
        head, positions, atom_types, charges, node_batch_idx, n_graphs,
        create_graph=True, detach_positions=False,
    )
    energy_target = energy_target.to(total.device, dtype=total.dtype).view_as(total)
    residual_target = energy_target - base.detach()
    e_res = (strain - residual_target).abs().mean()
    e_total = (total - energy_target).abs().mean()
    f_mse = (force - force_target).pow(2).sum(dim=-1).mean()
    cos = F.cosine_similarity(force, force_target, dim=-1, eps=1e-8).mean()
    loss = e_res + lambda_force * f_mse
    return loss, {
        "native_head_energy_mae_total": float(e_total.detach().item()),
        "native_head_energy_mae_residual": float(e_res.detach().item()),
        "native_head_force_mse": float(f_mse.detach().item()),
        "native_head_force_cosine": float(cos.detach().item()),
    }
