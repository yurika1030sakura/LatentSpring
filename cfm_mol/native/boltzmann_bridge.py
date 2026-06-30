"""Local Boltzmann bridge loss for BGFM-Native.

For each parent molecule m we have K perturbations, their external
neural-potential energies E_m^k, and the model coordinate log-densities
ell_m^k = log p_theta^pos(r_m^k | c_m).  The cloud-restricted target is

    w_m^k = softmax_k(- beta E_m^k),

and the model distribution on the same finite cloud is

    q_m^k = softmax_k(ell_m^k).

The bridge minimizes KL(w || q).  The unknown partition term Z_c cancels
inside the parent-wise softmax, so this is a direct finite-sample
Boltzmann-ratio objective rather than an arbitrary energy regularizer.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _unique_groups(parent_id: torch.Tensor) -> torch.Tensor:
    return torch.unique(parent_id.detach(), sorted=True)


def local_boltzmann_bridge_loss(
    logp: torch.Tensor,
    energies: torch.Tensor,
    parent_id: torch.Tensor,
    kT: float | torch.Tensor = 1.0,
    min_group_size: int = 2,
    energy_clip: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute mean parent-wise KL(softmax(-E/kT) || softmax(logp))."""
    if logp.shape != energies.shape or logp.shape != parent_id.shape:
        raise ValueError(
            f"expected logp, energies, parent_id to have same shape; got "
            f"{tuple(logp.shape)}, {tuple(energies.shape)}, {tuple(parent_id.shape)}"
        )
    valid = torch.isfinite(logp) & torch.isfinite(energies)
    if int(valid.sum().item()) < min_group_size:
        return logp.sum() * 0.0, {"bridge_groups_used": 0.0, "bridge_kl": 0.0}

    losses = []
    entropies = []
    for pid in _unique_groups(parent_id[valid]):
        mask = valid & (parent_id == pid)
        if int(mask.sum().item()) < min_group_size:
            continue
        l = logp[mask]
        e = energies[mask]
        if isinstance(kT, torch.Tensor):
            kt = kT[mask].to(e.dtype)
        else:
            kt = torch.full_like(e, float(kT))
        beta_e = e / kt.clamp_min(1e-8)
        # stabilize by subtracting min; optional clip protects early training.
        beta_e = beta_e - beta_e.min()
        if energy_clip is not None:
            beta_e = beta_e.clamp(max=float(energy_clip))
        log_w = F.log_softmax(-beta_e, dim=0)
        log_q = F.log_softmax(l, dim=0)
        w = log_w.exp()
        losses.append((w * (log_w - log_q)).sum())
        entropies.append(-(w * log_w).sum())

    if not losses:
        return logp.sum() * 0.0, {"bridge_groups_used": 0.0, "bridge_kl": 0.0}
    loss = torch.stack(losses).mean()
    ent = torch.stack(entropies).mean()
    return loss, {
        "bridge_groups_used": float(len(losses)),
        "bridge_kl": float(loss.detach().item()),
        "bridge_target_entropy": float(ent.detach().item()),
    }
