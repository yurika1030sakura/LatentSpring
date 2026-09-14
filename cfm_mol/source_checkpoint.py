"""Restore the declared coordinate source carried by a frozen research checkpoint."""
from .tree_mixture_prior import TreeMixturePrior
from .tree_prior_controls import TreePriorControl


def prior_from_checkpoint(checkpoint):
    kind = checkpoint['research_protocol'].get('source_prior_kind', 'gaussian')
    saved = checkpoint.get('source_prior')
    if kind == 'gaussian':
        if saved is not None:
            raise ValueError('Gaussian checkpoint has an unexpected source prior')
        return None
    if saved is None:
        raise ValueError('Non-Gaussian checkpoint is missing its source prior')
    config = saved['configuration']
    if kind in ['fixed', 'node', 'pair'] and config.get('mode') == kind:
        prior = TreeMixturePrior(**config).double()
    elif kind in ['harmonic_tree', 'covariance_gaussian'] and config.get('control') == kind:
        prior = TreePriorControl(**config).double()
    else:
        raise ValueError('Source declaration and saved configuration disagree')
    prior.load_state_dict(saved['state_dict'], strict=True)
    prior.requires_grad_(False)
    return prior.eval()
