import math
from collections import Counter

import numpy as np
import torch

from cfm_mol.tree_mixture_prior import TreeMixturePrior
from cfm_mol.tree_prior_controls import TreePriorControl, cayley_tree, edge_embedding, symmetrize_species_covariance


def test_pruefer_tree_law_and_edge_covariance():
    u = np.array([.2, 1.3, 2.])
    rng = np.random.default_rng(8)
    counts = Counter()
    for _ in range(10000):
        edges = cayley_tree(u, rng)
        counts[tuple(sorted(tuple(sorted(e)) for e in edges))] += 1
    trees = [((0, 1), (0, 2)), ((0, 1), (1, 2)), ((0, 2), (1, 2))]
    weights = np.array([np.prod([u[i]*u[j] for i, j in tree]) for tree in trees])
    weights /= weights.sum()
    assert np.max(abs(np.array([counts[t]/10000 for t in trees]) - weights)) < .02
    edges = [(0, 1), (2, 1)]
    embedding = edge_embedding(3, edges)
    incidence = np.array([[1., -1., 0.], [0., -1., 1.]])
    np.testing.assert_allclose(incidence @ embedding, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(embedding.mean(0), 0., atol=1e-14)
    variance = np.array([.3, 2.])
    covariance = (embedding * variance) @ embedding.T
    np.testing.assert_allclose(incidence @ covariance @ incidence.T, np.diag(variance), atol=1e-14)


def test_two_atom_density_and_second_moment_identity():
    numbers = [1, 9]
    harmonic = TreePriorControl('harmonic_tree')
    gaussian = TreePriorControl('covariance_gaussian')
    fixed = TreeMixturePrior('fixed').double()
    _, length = fixed.parameters_for(numbers, 0, 1)
    variance = float(length[0, 1]**2 * math.exp(.2**2) / 3)
    expected = torch.tensor([[variance/4, -variance/4], [-variance/4, variance/4]], dtype=torch.float64)
    torch.testing.assert_close(gaussian.covariance_for(numbers, 0, 1), expected, atol=1e-14, rtol=0)
    x = torch.tensor([[.5, .2, -.3], [-.5, -.2, .3]], dtype=torch.float64, requires_grad=True)
    reference = -.5*(x[0]-x[1]).square().sum()/variance - 1.5*math.log(2*math.pi*variance) + 1.5*math.log(2)
    for prior in [harmonic, gaussian]:
        actual = prior.log_prob(x, numbers, 0, 1)
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=0)
        torch.testing.assert_close(torch.autograd.grad(actual, x, retain_graph=True)[0],
                                   torch.autograd.grad(reference, x, retain_graph=True)[0], atol=1e-12, rtol=0)
    # The original shell distribution has this same edge second moment.
    samples = [fixed.sample(numbers, 0, 1, rng=np.random.default_rng(i),
                generator=torch.Generator().manual_seed(i))[0] for i in range(3000)]
    empirical = torch.stack(samples).square().mean((0, 2))
    torch.testing.assert_close(empirical, expected.diag(), atol=0., rtol=.04)


def test_single_gaussian_symmetry_density_and_sampling():
    numbers = [6, 1, 8, 1, 1]
    prior = TreePriorControl('covariance_gaussian', covariance_trees=128)
    covariance = prior.covariance_for(numbers, 0, 2)
    permutation = np.array([3, 2, 0, 4, 1])
    other = prior.covariance_for(np.array(numbers)[permutation].tolist(), 0, 2)
    torch.testing.assert_close(other, covariance[permutation][:, permutation], atol=1e-13, rtol=0)
    x, _ = prior.sample(numbers, 0, 2, rng=np.random.default_rng(3), generator=torch.Generator().manual_seed(4))
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(5)))
    torch.testing.assert_close(prior.log_prob(x, numbers, 0, 2),
        prior.log_prob((x @ rotation)[permutation], np.array(numbers)[permutation].tolist(), 0, 2), atol=1e-11, rtol=0)
    basis, factor = prior._factor(numbers, 0, 2)
    z = torch.randn(12000, len(numbers)-1, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(6))
    draws = torch.einsum('nk,bkd->bnd', basis @ factor, z)
    empirical = torch.einsum('bnd,bmd->nm', draws, draws)/(len(draws)*3)
    torch.testing.assert_close(empirical, covariance, atol=.03, rtol=.03)


def test_species_symmetrization_is_group_average():
    import itertools
    rng = np.random.default_rng(9)
    raw = rng.normal(size=(4, 3))
    covariance = raw @ raw.T
    expected = np.zeros((4, 4))
    for group in itertools.permutations([0, 1, 3]):
        order = [group[0], group[1], 2, group[2]]
        expected += covariance[np.ix_(order, order)] / 6
    p = np.eye(4) - np.ones((4, 4))/4
    np.testing.assert_allclose(symmetrize_species_covariance(covariance, [1, 1, 6, 1]), p @ expected @ p, atol=1e-13)
