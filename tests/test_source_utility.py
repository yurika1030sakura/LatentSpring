import itertools
import numpy as np
import pytest
import torch
from cfm_mol.tree_mixture_prior import TreeMixturePrior, log_tree_partition
from cfm_mol.source_utility import batched_log_partition, batched_log_prob, tree_kl, TrustMixturePrior, importance_utility


def enumerate_tree_scores(a):
    n = len(a)
    scores = []
    for edges in itertools.combinations(itertools.combinations(range(n), 2), n-1):
        reached = {0}
        for _ in range(n):
            for i, j in edges:
                if i in reached or j in reached:
                    reached.update([i, j])
        if len(reached) == n:
            scores.append(sum(a[i, j] for i, j in edges))
    return torch.stack(scores)


def test_partition_kl_and_gradients_against_enumeration():
    torch.manual_seed(31901)
    raw = torch.randn(4, 4, dtype=torch.double, requires_grad=True)
    a = (raw+raw.T)/2
    raw0 = torch.randn(4, 4, dtype=torch.double)
    a0 = (raw0+raw0.T)/2
    scores, scores0 = enumerate_tree_scores(a), enumerate_tree_scores(a0)
    exact = (scores.softmax(0)*(scores.log_softmax(0)-scores0.log_softmax(0))).sum()
    value = tree_kl(a, a0)
    torch.testing.assert_close(value, exact)
    torch.testing.assert_close(torch.autograd.grad(value, raw, retain_graph=True)[0], torch.autograd.grad(exact, raw)[0])
    torch.testing.assert_close(batched_log_partition(torch.stack([a, a0])), torch.stack([log_tree_partition(a), log_tree_partition(a0)]))


def test_batched_coordinate_density_and_finite_gradients():
    torch.manual_seed(31902)
    prior = TreeMixturePrior('pair').double()
    numbers = [6, 8, 1, 1]
    x = torch.stack([prior.sample(numbers, 0, 1, rng=np.random.default_rng(i), generator=torch.Generator().manual_seed(i))[0] for i in range(5)])
    a, length = prior.parameters_for(numbers, 0, 1)
    values = batched_log_prob(x, a, length, prior.width)
    torch.testing.assert_close(values, torch.stack([prior.log_prob(y, numbers, 0, 1) for y in x]))
    (-values.mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in prior.parameters() if p.grad is not None)


def test_mixture_bound_and_common_kernel_data_processing():
    torch.manual_seed(31903)
    base, candidate = TreeMixturePrior('fixed').double(), TreeMixturePrior('pair').double()
    torch.nn.init.normal_(candidate.head[-1].weight, std=2.)
    mix = TrustMixturePrior(candidate, base, delta=.001)
    a, a0, length, kl, epsilon = mix.components([6, 8, 1, 1], 0, 1)
    p, p0 = enumerate_tree_scores(a).softmax(0), enumerate_tree_scores(a0).softmax(0)
    pmix = (1-epsilon)*p0+epsilon*p
    kernel = torch.rand(len(p), 3, dtype=torch.double).softmax(-1)
    y, y0 = pmix@kernel, p0@kernel
    assert float((y*(y.log()-y0.log())).sum()) <= float(epsilon*kl)+1e-12
    assert float((pmix*(pmix.log()-p0.log())).sum()) <= float(epsilon*kl)+1e-12
    assert float(epsilon*kl) <= .001+1e-12
    x = torch.randn(3, 4, 3, dtype=torch.double); x -= x.mean(1, keepdim=True)
    logq, _ = mix.log_prob_batch(x, [6, 8, 1, 1], 0, 1)
    expected = ((1-epsilon)*batched_log_prob(x, a0, length, .2).exp()+epsilon*batched_log_prob(x, a, length, .2).exp()).log()
    torch.testing.assert_close(logq, expected)
    (-logq.mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in candidate.parameters() if p.grad is not None)


def test_leave_one_out_importance_exact_expectation():
    q0 = torch.tensor([.3, .7], dtype=torch.double)
    q = torch.tensor([.6, .4], dtype=torch.double)
    reward = torch.tensor([.2, 1.], dtype=torch.double)
    total = 0.
    for i, j in itertools.product(range(2), repeat=2):
        ids = [i, j]
        value, _ = importance_utility(q[ids].log(), q0[ids].log(), reward[ids])
        total += q0[i]*q0[j]*value
    torch.testing.assert_close(total, ((q-q0)*reward).sum())


def test_kernel_mismatch_rejected():
    with pytest.raises(ValueError, match='width'):
        TrustMixturePrior(TreeMixturePrior('pair', width=.3), TreeMixturePrior('fixed'))
