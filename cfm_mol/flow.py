"""Conditional flow-matching loss on the continuous fibre.

This module implements the training objective
    L_CFM(theta) = E_{t, r1, r0} || v_theta(r_t, t, a, b) - (r1 - r0) ||^2
from methods_derivation.tex eq. (1), with r_t sampled from the straight-line
interpolant retracted onto the steric fibre M_ster(a).

Discrete-channel loss (atom types, bond orders) uses discrete flow matching
(Gat et al. 2024) and is handled in cfm_mol.flow_model together with the
network architecture.
"""
from __future__ import annotations

import torch

from cfm_mol.fibre import sample_interpolant, sample_prior


# ---------------------------------------------------------------------------
# Time sampling
# ---------------------------------------------------------------------------

def sample_time(batch_size: int, device: str | torch.device = "cpu") -> torch.Tensor:
    """Uniform t ~ U(0, 1). Shape (batch_size,).

    TODO[Week 3]: try logit-normal time sampling (Esser et al. 2024) which is
    SOTA for flow matching; logit-normal emphasises mid-trajectory where the
    velocity is hardest to learn.
    """
    return torch.rand(batch_size, device=device)


# ---------------------------------------------------------------------------
# CFM regression loss (continuous channel)
# ---------------------------------------------------------------------------

def cfm_loss(
    v_theta: torch.nn.Module,
    r1: torch.Tensor,
    a: torch.LongTensor,
    b: torch.LongTensor,
    d_min_table: torch.Tensor,
    prior_scale: float = 3.0,
) -> torch.Tensor:
    """Compute the CFM regression loss for the continuous (coordinate) channel.

    For each molecule in the batch, sample r0 from the prior, t ~ U(0,1), form
    the retracted interpolant r_t, and regress v_theta(r_t, t, a, b) against
    the conditional velocity r1 - r0.

    Args
    ----
    v_theta : callable mapping (r_t, t, a, b) -> (B, N, 3) velocity.
              Must be a torch.nn.Module-like object; passed arguments are
              batched.
    r1 : (B, N, 3) data sample (on fibre).
    a : (B, N) atom types.
    b : (B, N, N) bond-order indices.
    d_min_table : (A, A).
    prior_scale : Gaussian std for prior samples.

    Returns
    -------
    loss : scalar tensor (mean squared error).
    """
    B, N, _ = r1.shape
    device = r1.device

    r0 = sample_prior(
        n_atoms=N,
        batch_size=B,
        a=a,
        d_min_table=d_min_table,
        scale=prior_scale,
        device=device,
    )
    t = sample_time(B, device=device)
    r_t, u_target = sample_interpolant(r0, r1, t, a, d_min_table)

    v_pred = v_theta(r_t, t, a, b)
    return ((v_pred - u_target) ** 2).mean()


# ---------------------------------------------------------------------------
# Toy linear velocity field — for testing the loss pipeline before wiring
# up the real (equivariant) network.
# ---------------------------------------------------------------------------

class LinearVelocityNet(torch.nn.Module):
    """Trivial velocity net for sanity tests.

    Maps (r_t, t, a, b) -> learned per-atom velocity that depends only on r_t
    and t through a tiny MLP. Not equivariant, not expressive — purely here
    so cfm_loss() can be called end-to-end without needing FlowMol3's
    architecture wired up.
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(4, hidden),    # 3 coord + 1 time
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, 3),
        )

    def forward(
        self,
        r_t: torch.Tensor,
        t: torch.Tensor,
        a: torch.LongTensor,
        b: torch.LongTensor,
    ) -> torch.Tensor:
        B, N, _ = r_t.shape
        t_ = t.view(B, 1, 1).expand(B, N, 1)
        x = torch.cat([r_t, t_], dim=-1)               # (B, N, 4)
        return self.mlp(x)
