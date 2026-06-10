"""Reflected-SDE coordinate-channel baseline for E2 (convergence rate).

Reference: Lou, A.; Ermon, S. Reflected Diffusion Models. ICML 2023.

Purpose: demonstrate empirically that Euler-Maruyama discretisation of a
reflected SDE on the steric fibre converges at rate $O(\\sqrt{\\Delta t})$,
while our flow-matching Euler scheme converges at $O(\\Delta t)$ (Theorem
4.1 + Lemma 5.2 of the methods derivation). This is the single-figure
comparison that validates our core theoretical advantage.

Scope: coordinate channel only. We fix the discrete state $(a, b)$ to the
training data and compare SDE vs ODE integrators head-to-head on the same
steric fibre $\\mathcal{M}_{\\rm ster}(a)$. Not a full molecule generator.

Architecture: simple 3-layer MLP on pairwise distances + atom one-hots.
Per-molecule scores are E(3)-equivariant because we output scalar scores
on distances and lift to 3D by projecting along pair axes. Not as strong
as GVP but suffices for E2 (which measures DISCRETISATION RATE, not
sample quality).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cfm_mol.fibre import retract


# ---------------------------------------------------------------------------
# Simple equivariant score network (coord channel only)
# ---------------------------------------------------------------------------

class CoordScoreNet(nn.Module):
    """E(3)-equivariant score network: distance-conditioned edge MLP,
    score = sum of pair unit-vectors weighted by a learned scalar.

    Inputs:
      r : (B, N, 3) positions
      a : (B, N) atom-type indices  (one-hot'd internally)
      t : (B,) time in [0, 1]
    Output:
      score : (B, N, 3) E(3)-equivariant
    """

    def __init__(self, n_atom_types: int = 5, hidden: int = 128,
                 time_dim: int = 32):
        super().__init__()
        self.n_atom_types = n_atom_types
        # (pair distance + atom-i one-hot + atom-j one-hot + time) -> scalar
        in_dim = 1 + 2 * n_atom_types + time_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.time_dim = time_dim

    def _time_embed(self, t: torch.Tensor) -> torch.Tensor:
        freqs = torch.exp(-math.log(10_000.0) *
                          torch.arange(0, self.time_dim, 2, device=t.device) /
                          self.time_dim)
        ang = t.unsqueeze(-1) * freqs
        return torch.cat([ang.sin(), ang.cos()], dim=-1)  # (B, time_dim)

    def forward(self, r: torch.Tensor, a: torch.LongTensor,
                t: torch.Tensor) -> torch.Tensor:
        B, N, _ = r.shape
        # Pairwise directions + distances.
        diffs = r.unsqueeze(-2) - r.unsqueeze(-3)          # (B, N, N, 3)
        dists = diffs.norm(dim=-1).clamp_min(1e-6)        # (B, N, N)
        units = diffs / dists.unsqueeze(-1)                # (B, N, N, 3)

        # One-hot atoms.
        a_oh = F.one_hot(a, num_classes=self.n_atom_types).float()   # (B, N, A)
        a_i = a_oh.unsqueeze(-2).expand(B, N, N, -1)
        a_j = a_oh.unsqueeze(-3).expand(B, N, N, -1)

        # Time features broadcast.
        t_emb = self._time_embed(t)                        # (B, time_dim)
        t_e = t_emb.view(B, 1, 1, -1).expand(B, N, N, -1)

        feats = torch.cat([dists.unsqueeze(-1), a_i, a_j, t_e], dim=-1)
        scalars = self.edge_mlp(feats).squeeze(-1)         # (B, N, N)

        # Mask diagonal.
        eye = torch.eye(N, dtype=torch.bool, device=r.device)
        scalars = scalars.masked_fill(eye.unsqueeze(0), 0.0)

        # Score on atom i: sum over j of scalar_{ij} * unit_{ij}.
        score = (scalars.unsqueeze(-1) * units).sum(dim=-2)  # (B, N, 3)
        return score


# ---------------------------------------------------------------------------
# Reflected-SDE model wrapper
# ---------------------------------------------------------------------------

class ReflectedSDEModel(nn.Module):
    """Lou-Ermon reflected-SDE on the steric fibre $\\mathcal{M}_{\\rm ster}(a)$.

    VE noise schedule $\\sigma(t) = \\sigma_{\\min}
    (\\sigma_{\\max}/\\sigma_{\\min})^t$.
    """

    def __init__(self, score_net: nn.Module, d_min_table: torch.Tensor,
                 sigma_min: float = 0.01, sigma_max: float = 1.0):
        super().__init__()
        self.score_net = score_net
        self.register_buffer("d_min_table", d_min_table)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def sigma_deriv(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma(t) * math.log(self.sigma_max / self.sigma_min)

    def training_loss(self, r1: torch.Tensor, a: torch.LongTensor,
                      t: torch.Tensor) -> torch.Tensor:
        """Denoising score matching. Eq. 7 of Lou-Ermon (simplified,
        without explicit boundary Jacobian term — standard in practice)."""
        sigma_t = self.sigma(t).view(-1, 1, 1)
        noise = torch.randn_like(r1)
        r_noisy = r1 + sigma_t * noise
        r_noisy, _ = retract(r_noisy, a, self.d_min_table)

        score_target = -noise / sigma_t
        score_pred = self.score_net(r_noisy, a, t)

        weight = sigma_t ** 2
        loss = ((score_pred - score_target) ** 2 * weight).mean()
        return loss

    @torch.no_grad()
    def sample_euler_maruyama(
        self, a: torch.LongTensor, n_steps: int,
    ) -> torch.Tensor:
        B, N = a.shape
        device = a.device
        r = torch.randn(B, N, 3, device=device) * self.sigma_max
        r, _ = retract(r, a, self.d_min_table)
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t = 1.0 - step * dt
            t_tensor = torch.full((B,), t, device=device)
            sigma_t = self.sigma(t_tensor).view(-1, 1, 1)
            sigma_dot = self.sigma_deriv(t_tensor).view(-1, 1, 1)
            score = self.score_net(r, a, t_tensor)
            g_sq = 2 * sigma_t * sigma_dot
            drift = -g_sq * score
            diffusion = torch.sqrt(g_sq * dt) * torch.randn_like(r)
            r = r + drift * (-dt) + diffusion
            r, _ = retract(r, a, self.d_min_table)
        return r

    @torch.no_grad()
    def sample_probability_flow_ode(
        self, a: torch.LongTensor, n_steps: int,
    ) -> torch.Tensor:
        """Deterministic probability-flow ODE companion. Useful as a
        *same-network* sanity check: probability-flow ODE should converge
        at $O(\\Delta t)$ (matching FM), while SDE converges at
        $O(\\sqrt{\\Delta t})$. Both use the same score network."""
        B, N = a.shape
        device = a.device
        r = torch.randn(B, N, 3, device=device) * self.sigma_max
        r, _ = retract(r, a, self.d_min_table)
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t = 1.0 - step * dt
            t_tensor = torch.full((B,), t, device=device)
            sigma_t = self.sigma(t_tensor).view(-1, 1, 1)
            sigma_dot = self.sigma_deriv(t_tensor).view(-1, 1, 1)
            score = self.score_net(r, a, t_tensor)
            # Probability-flow ODE: dr = [f - 0.5 * g^2 * score] dt
            drift = -0.5 * (2 * sigma_t * sigma_dot) * score
            r = r + drift * (-dt)
            r, _ = retract(r, a, self.d_min_table)
        return r
