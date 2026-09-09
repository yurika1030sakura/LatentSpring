"""Grouped residual objectives with explicit stochastic-density semantics.

For independent conditionally unbiased log-density replicates, their centered
cross-product estimates the squared exact-trace residual without the extra
trace-noise penalty. This is a standard independent-product identity, not an
unbiased continuous-time likelihood claim. Sample losses can be negative.
"""
from __future__ import annotations

import math
import torch


def grouped_replica_residual(log_q, energies, parent_id, *, kT=1.0,
                             estimator="squared"):
    """Equal-parent loss, excluding invalid geometries jointly across replicas.

    ``squared`` squares the mean of all replicas; ``replica_product`` requires
    exactly two independent replicas, with the same total trace budget as a
    squared two-replica control. Energies are centered in float64 BEFORE adding
    log q to avoid cancellation against large composition-dependent offsets.
    No positivity clipping is applied to the replica product.
    """
    if estimator not in {"squared", "replica_product"}:
        raise ValueError("Unknown residual estimator")
    if log_q.ndim == 1:
        log_q = log_q[None, :]
    if log_q.ndim != 2 or log_q.shape[0] == 0:
        raise ValueError("Expected log q shape (replicas, geometries)")
    if estimator == "replica_product" and log_q.shape[0] != 2:
        raise ValueError("Replica product requires exactly two independent traces")
    if energies.shape != parent_id.shape or energies.shape != log_q.shape[1:]:
        raise ValueError("Energy, group and geometry counts must match")
    temperature = torch.as_tensor(kT, device=energies.device, dtype=torch.float64)
    if not torch.isfinite(temperature).all() or (temperature <= 0).any():
        raise ValueError("kT must be positive and finite")
    scaled_energy = energies.double()/temperature
    valid = torch.isfinite(log_q).all(0) & torch.isfinite(scaled_energy)
    losses, plug_ins, penalties = [], [], []
    for group in torch.unique(parent_id):
        mask = valid & (parent_id == group)
        if int(mask.sum()) < 2:
            continue
        e = scaled_energy[mask]
        e = e-e.mean()
        r = log_q[:, mask].double()+e
        r = r-r.mean(-1, keepdim=True)
        plug_in = r.mean(0).square().mean()
        if estimator == "replica_product":
            value = (r[0]*r[1]).mean()
            penalty = (r[0]-r[1]).square().mean()/4
        else:
            value = plug_in
            penalty = plug_in.new_zeros(())
        losses.append(value)
        plug_ins.append(plug_in.detach())
        penalties.append(penalty.detach())
    if not losses:
        zero = log_q[:, valid].sum()*0
        return zero, {"n_valid": int(valid.sum()), "n_groups_used": 0,
            "residual_within_std": 0., "squared_replica_mean": 0.,
            "trace_noise_penalty": 0.}
    loss = torch.stack(losses).mean()
    plug_in = torch.stack(plug_ins).mean()
    return loss, {"n_valid": int(valid.sum()), "n_groups_used": len(losses),
        "residual_within_std": math.sqrt(float(plug_in)),
        "squared_replica_mean": float(plug_in),
        "trace_noise_penalty": float(torch.stack(penalties).mean())}
