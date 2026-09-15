"""Checks for the independent benchmark's source law and diffusion adapter."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cfm_mol.matched_egnn import HarmonicSource, initialize, loss, sample, state_hash, vector
from cfm_mol.tree_prior_controls import edge_embedding


@pytest.fixture
def spec():
    root = Path(__file__).resolve().parents[1]
    value = json.loads((root/'research/evidence/conditional_edm_extended_s0_v1.json').read_text())
    if not Path(value['upstream']).exists():
        pytest.skip('Official EDM checkout is required for adapter checks')
    value.update(initialization_seed=40401, gaga_max_t=1000, data_variance_per_dof=1.)
    torch.set_num_threads(2)
    return value


def test_pruefer_source_matches_enumerated_three_atom_covariance():
    source = HarmonicSource(.2)
    numbers = [6, 1, 1]
    rng = np.random.default_rng(991)
    draws = torch.stack([source.sample(numbers, rng) for _ in range(5000)])
    u, scales = source.cache[tuple(numbers)]
    trees = [[(0, 1), (0, 2)], [(0, 1), (1, 2)], [(0, 2), (1, 2)]]
    weights = np.array([np.prod([u[i]*u[j] for i,j in tree]) for tree in trees])
    weights /= weights.sum()
    expected = sum(w*(edge_embedding(3, tree)*np.array([scales[i,j]**2 for i,j in tree])) @ edge_embedding(3, tree).T for w,tree in zip(weights, trees))
    measured = np.einsum('bik,bjk->ij', draws.numpy(), draws.numpy())/(len(draws)*3)
    np.testing.assert_allclose(measured, expected, atol=.015, rtol=.08)
    assert draws.mean(1).abs().max() < 1e-12


def test_initialization_and_trainable_objectives(spec):
    model = initialize(spec, 'cpu')
    other = initialize(spec, 'cpu')
    assert state_hash(model) == state_hash(other)
    source = HarmonicSource()
    numbers = torch.tensor([[6, 1, 1]])
    clean = torch.tensor([[[0., 0., 0.], [1., .2, 0.], [-1., .3, .1]]])
    for kind in ['edm', 'gaga', 'gaussian_fm', 'harmonic_fm']:
        model.zero_grad(set_to_none=True)
        value = loss(model, clean, numbers, kind, spec, source, 401)
        value.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert torch.isfinite(value) and gradients
        assert all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0


def test_diffusion_adapter_agrees_with_existing_sampler(spec):
    from scripts.research.run_conditional_edm import sample as reference_sample
    model = initialize(spec, 'cpu').eval()
    source = HarmonicSource()
    expected, initial = reference_sample(model, [6, 1, 1], spec, 704, 8, 2)
    actual, actual_initial = sample(model, [6, 1, 1], 'edm', spec, source, 704, 2, 8)
    torch.testing.assert_close(actual_initial, initial)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    # GAGA reduces to the same sampler when truncation is disabled and variance=1.
    gaga, _ = sample(model, [6, 1, 1], 'gaga', spec, source, 704, 2, 8)
    torch.testing.assert_close(gaga, actual, atol=2e-4, rtol=2e-4)


def test_velocity_equivariance(spec):
    model = initialize(spec, 'cpu').eval()
    x = torch.tensor([[[0., 0., 0.], [1., .2, 0.], [-1., .3, .1]]])
    x -= x.mean(1, keepdim=True)
    z = torch.tensor([[6, 1, 1]])
    rotation = torch.tensor([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    t = torch.tensor([[.4]])
    first = vector(model, x, t, z, spec)
    second = vector(model, x @ rotation, t, z, spec)
    torch.testing.assert_close(second, first @ rotation, atol=2e-6, rtol=2e-5)
