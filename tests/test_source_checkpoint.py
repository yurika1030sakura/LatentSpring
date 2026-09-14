import numpy as np
import pytest
import torch

from cfm_mol.source_checkpoint import prior_from_checkpoint
from cfm_mol.tree_mixture_prior import TreeMixturePrior
from cfm_mol.tree_prior_controls import TreePriorControl


@pytest.mark.parametrize('kind', ['fixed', 'node', 'pair', 'harmonic_tree', 'covariance_gaussian'])
def test_saved_source_restores_sampling_and_density(kind):
    torch.manual_seed(31091)
    original = TreeMixturePrior(kind).double() if kind in ['fixed', 'node', 'pair'] else TreePriorControl(kind).double()
    if kind in ['node', 'pair']:
        with torch.no_grad():
            original.head[-1].weight.normal_(std=.05)
    checkpoint = dict(research_protocol=dict(source_prior_kind=kind), source_prior=dict(
        configuration=original.configuration, state_dict=original.state_dict()))
    restored = prior_from_checkpoint(checkpoint)
    numbers = [6, 6, 1, 1, 8, 1, 1]
    def sample(prior):
        return prior.sample(numbers, 0, 1, rng=np.random.default_rng(13), generator=torch.Generator().manual_seed(14))
    left, tree = sample(original)
    right, other = sample(restored)
    assert tree == other
    torch.testing.assert_close(left, right, atol=0, rtol=0)
    torch.testing.assert_close(original.log_prob(left, numbers, 0, 1), restored.log_prob(left, numbers, 0, 1), atol=0, rtol=0)
    assert not any(parameter.requires_grad for parameter in restored.parameters())


def test_wrong_or_missing_prior_is_rejected():
    assert prior_from_checkpoint(dict(research_protocol=dict(source_prior_kind='gaussian'), source_prior=None)) is None
    with pytest.raises(ValueError, match='missing'):
        prior_from_checkpoint(dict(research_protocol=dict(source_prior_kind='node'), source_prior=None))
    with pytest.raises(ValueError, match='disagree'):
        prior_from_checkpoint(dict(research_protocol=dict(source_prior_kind='node'), source_prior=dict(configuration=dict(mode='pair'), state_dict={})))
