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
    *,
    kT_schedule: dict | None = None,
    step_frac: float | None = None,
    symmetric: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute mean parent-wise KL(softmax(-E/kT) || softmax(logp)).

    kT_schedule (optional, SPEC item A): dict with keys
        {mode, start_eV, final_eV}. If provided AND step_frac is given,
        the effective kT is annealed log-linearly from start_eV to final_eV
        and OVERRIDES the static `kT` argument. The static path
        (kT_schedule=None) preserves legacy behavior exactly.

    symmetric (paper Eq. eq:bridge-sym): if True, return the symmetric
        average 0.5 * (KL(w || q) + KL(q || w)). Adds a mode-seeking term
        that penalizes the model for placing mass on cloud entries OMol25
        considers high-energy. Forward KL (default, False) is mode-
        covering and is the headline objective; the symmetric variant
        is used only in the bridge ablation column.
    """
    if logp.shape != energies.shape or logp.shape != parent_id.shape:
        raise ValueError(
            f"expected logp, energies, parent_id to have same shape; got "
            f"{tuple(logp.shape)}, {tuple(energies.shape)}, {tuple(parent_id.shape)}"
        )
    valid = torch.isfinite(logp) & torch.isfinite(energies)
    if int(valid.sum().item()) < min_group_size:
        return logp.sum() * 0.0, {"bridge_groups_used": 0.0, "bridge_kl": 0.0}

    # SPEC item A: optional bridge-kT anneal. When in effect, this is the
    # "tempered Boltzmann bridge" -- DO NOT call this "room temperature"
    # unless kT_eff == 0.02569 eV. The schedule overrides the static kT arg.
    if kT_schedule is not None and step_frac is not None:
        mode = str(kT_schedule.get("mode", "annealed"))
        start = float(kT_schedule.get("start_eV", 0.25))
        final = float(kT_schedule.get("final_eV", 0.02569))
        if mode == "fixed_tempered":
            kT = start
        else:
            f = max(min(float(step_frac), 1.0), 0.0)
            import math
            kT = math.exp(math.log(start) + f * (math.log(final) - math.log(start)))

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
        forward_kl = (w * (log_w - log_q)).sum()
        if symmetric:
            q = log_q.exp()
            reverse_kl = (q * (log_q - log_w)).sum()
            losses.append(0.5 * (forward_kl + reverse_kl))
        else:
            losses.append(forward_kl)
        entropies.append(-(w * log_w).sum())

    if not losses:
        return logp.sum() * 0.0, {"bridge_groups_used": 0.0, "bridge_kl": 0.0}
    loss = torch.stack(losses).mean()
    ent = torch.stack(entropies).mean()
    return loss, {
        "bridge_groups_used": float(len(losses)),
        "bridge_kl": float(loss.detach().item()),
        "bridge_target_entropy": float(ent.detach().item()),
        "bridge_kT_eV_effective": (
            float(kT) if not isinstance(kT, torch.Tensor)
            else float(kT.mean().item())
        ),
        "bridge_symmetric": float(symmetric),
    }
