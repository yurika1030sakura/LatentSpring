"""Paired benchmark uncertainty with composition and source-parent clusters."""
import torch


def hierarchical_paired_mean(differences, *, generator, replicates=2000):
    x = torch.as_tensor(differences).detach().cpu().double()
    if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 2 or x.shape[2] < 2:
        raise ValueError('Require [algorithm replicas, at least two compositions, at least two parents]')
    if not torch.isfinite(x).all() or replicates < 100:
        raise ValueError('Finite complete paired endpoints and sufficient bootstrap draws required')
    # Algorithm/training replicas do not create new independent molecules.
    parents = x.mean(0)
    c, n = parents.shape
    composition_ids = torch.randint(c, (replicates,c), generator=generator)
    parent_ids = torch.randint(n, (replicates,c,n), generator=generator)
    draws = parents[composition_ids[:, :, None], parent_ids].mean((1,2))
    return dict(mean=float(parents.mean()),
        hierarchical_95_percent_interval=draws.quantile(torch.tensor([.025,.975],dtype=x.dtype)).tolist(),
        per_composition_mean=parents.mean(1).tolist(), per_algorithm_replica_mean=x.mean((1,2)).tolist(),
        independent_compositions=c, parents_per_composition=n, algorithm_replicas=x.shape[0],
        scope='Hierarchical percentile bootstrap conditional on the fixed trained checkpoints; algorithm replicas are averaged within source parent.')
