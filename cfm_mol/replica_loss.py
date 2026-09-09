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
        penalty = (r[0]-r[1]).square().mean()/4 if len(r) == 2 else plug_in.new_zeros(())
        if estimator == "replica_product":
            value = (r[0]*r[1]).mean()
        else:
            value = plug_in
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


def make_grouped_probe_sampler(node_batch_idx, parent_id, generator=None, distribution='rademacher'):
    """Common random probes across corresponding atoms of sibling geometries.

    DGL batches store nodes contiguously and perturbation shards preserve atom
    order. Replicas remain independent: every callback draws fresh base noise.
    Common random numbers reduce contrast variance but do not generally remove
    the squared-loss bias when Jacobians depend on geometry.
    """
    if distribution not in {'rademacher','gaussian'}:raise ValueError('Unknown trace probe distribution')
    counts = torch.bincount(node_batch_idx,minlength=len(parent_id))
    if (counts == 0).any() or (node_batch_idx[1:] < node_batch_idx[:-1]).any():
        raise ValueError('Require nonempty, contiguous DGL graph nodes')
    parent_counts, parent_offsets, graph_offsets = {}, {}, []
    total = 0
    for parent, count in zip(parent_id.cpu().tolist(),counts.cpu().tolist()):
        if parent not in parent_counts:
            parent_counts[parent] = count
            parent_offsets[parent] = total
            total += count
        elif parent_counts[parent] != count:
            raise ValueError('Sibling geometries must have the same atom count and order')
        graph_offsets.append(parent_offsets[parent])
    starts = counts.cumsum(0)-counts
    local_index = torch.arange(len(node_batch_idx),device=node_batch_idx.device)-starts[node_batch_idx]
    offsets = torch.tensor(graph_offsets,device=node_batch_idx.device)
    mapping = offsets[node_batch_idx]+local_index
    def sample(step, replica, x):
        if distribution=='gaussian':
            base=torch.randn((total,x.shape[-1]),device=x.device,dtype=x.dtype,generator=generator)
        else:
            base = (2*torch.randint(0,2,(total,x.shape[-1]),device=x.device,generator=generator)-1).to(x)
        return base[mapping]
    return sample
