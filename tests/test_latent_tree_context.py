from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cfm_mol.latent_tree_context import tree_features, attach_context, patch_latent_tree_context, context_tree
from cfm_mol.clamped_density import deterministic_field, log_density_clamped_flow
from cfm_mol.clamped_fm import clamped_fm_path, clamped_fm_loss
from cfm_mol.orbit_pairing import typed_orbit_pair
from cfm_mol.radial_reference import prepare_research_backbone
from cfm_mol.tree_mixture_prior import TreeMixturePrior
from test_clamped_density import graph_batch, Schedule


def small_model():
    from flowmol.models.ctmc_vector_field import CTMCVectorField
    torch.manual_seed(4501)
    field = CTMCVectorField(n_atom_types=3, canonical_feat_order=['x', 'a', 'c', 'e'],
        interpolant_scheduler=Schedule(), n_bond_types=4, n_charges=6,
        n_hidden_scalars=8, n_vec_channels=4, n_hidden_edge_feats=8,
        n_recycles=1, n_molecule_updates=1, convs_per_update=2,
        n_message_gvps=1, n_update_gvps=1, n_expansion_gvps=1, rbf_dim=4,
        self_conditioning=True)
    return SimpleNamespace(vector_field=field)


def predict(model, graph, nbi, uem):
    with deterministic_field(model.vector_field):
        return model.vector_field(graph, torch.tensor([.4]), node_batch_idx=nbi, upper_edge_mask=uem)['x']


def prepare_graph():
    graph, nbi, uem = graph_batch((4,), dtype=torch.float32)
    for key in ['a', 'c']:
        graph.ndata[key+'_t'] = graph.ndata[key+'_1_true']
    graph.edata['e_t'] = graph.edata['e_1_true']
    return graph, nbi, uem


def test_tree_features_validation_and_relabeling():
    edges = [(0, 1), (1, 2), (1, 3)]
    features = tree_features(4, edges)
    assert features.shape == (4, 4, 3)
    torch.testing.assert_close(features, features.transpose(0, 1))
    assert features[0, 2, 1] == np.log1p(2)
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(permutation)
    relabeled = [(int(inverse[i]), int(inverse[j])) for i, j in edges]
    torch.testing.assert_close(tree_features(4, relabeled), features[permutation][:, permutation])
    for bad in [[(0, 1), (1, 0), (2, 3)], [(0, 1), (1, 2), (2, 0)], [(0, 4), (1, 2), (2, 3)]]:
        with pytest.raises(ValueError):
            tree_features(4, bad)


def test_source_permutation_follows_coupled_coordinates_without_changing_old_stream():
    graph, nbi, _ = graph_batch((4,))
    source = torch.randn(4, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(50))
    source -= source.mean(0)
    target = graph.ndata['x_1_true']
    kwargs = dict(radii=torch.ones(4, dtype=torch.float64), groups=torch.zeros(4, dtype=torch.long))
    old = typed_orbit_pair(source, target, generator=torch.Generator().manual_seed(51), **kwargs)
    new = typed_orbit_pair(source, target, generator=torch.Generator().manual_seed(51), return_source_permutation=True, **kwargs)
    assert torch.equal(old[0], new[0]) and torch.equal(old[1], new[1])
    assert old[2] == {k: v for k, v in new[2].items() if k != 'source_permutation'}
    permutation = new[2]['source_permutation']
    torch.testing.assert_close(torch.cdist(new[0], new[0]), torch.cdist(source, source)[permutation][:, permutation], atol=1e-12, rtol=0)
    features = tree_features(4, [(0, 1), (0, 2), (2, 3)])
    _, _, _, info = clamped_fm_path(graph, nbi, Schedule(), terminal_time=1., parameterization='displacement',
        prior_positions=source, prior_edge_context=[features], pairing='typed_rotation', pairing_radii=kwargs['radii'],
        generator=torch.Generator().manual_seed(52), pairing_generator=torch.Generator().manual_seed(51))
    torch.testing.assert_close(info['x0'], new[0], atol=1e-12, rtol=0)
    torch.testing.assert_close(info['prior_edge_context'][0], features[permutation][:, permutation])
    assert 'latent_tree_context' not in graph.edata


def test_real_backbone_zero_init_gradients_symmetry_and_checkpoint():
    model = small_model()
    graph, nbi, uem = prepare_graph()
    original = predict(model, graph, nbi, uem)
    patch_latent_tree_context(model, hidden=8)
    with pytest.raises(ValueError, match='requires latent_tree_context'):
        predict(model, graph, nbi, uem)
    context = tree_features(4, [(0, 1), (0, 2), (2, 3)])
    attach_context(graph, [context], nbi)
    assert torch.equal(original, predict(model, graph, nbi, uem))
    model.vector_field.train()
    source = graph.ndata['x_1_true'].clone() * .8
    loss = clamped_fm_loss(model, graph, nbi, uem, terminal_time=1., parameterization='displacement',
        prior_positions=source, prior_edge_context=[context], generator=torch.Generator().manual_seed(54))
    loss.backward()
    gradient = model.vector_field.latent_tree_adapter[-1].weight.grad
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    with torch.no_grad():
        model.vector_field.latent_tree_adapter[-1].weight.normal_(std=.03)
    original = predict(model, graph, nbi, uem)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(55)))
    permutation = torch.tensor([2, 0, 3, 1])
    with graph.local_scope():
        graph.ndata['x_t'] = graph.ndata['x_t'][permutation] @ rotation
        attach_context(graph, [context[permutation][:, permutation]], nbi)
        transformed = predict(model, graph, nbi, uem)
    torch.testing.assert_close(transformed, original[permutation] @ rotation, atol=3e-5, rtol=3e-5)
    restored = small_model()
    prepare_research_backbone(restored, dict(latent_tree_context=dict(hidden=8), source_prior_kind='fixed'))
    restored.vector_field.load_state_dict(model.vector_field.state_dict(), strict=True)
    assert torch.equal(original, predict(restored, graph, nbi, uem))
    with pytest.raises(ValueError, match='prior_edge_context'):
        clamped_fm_loss(model, graph, nbi, uem, prior_positions=source)
    with pytest.raises(ValueError, match='Tree-conditioned density'):
        log_density_clamped_flow(model, graph, nbi, uem, prior_log_prob=lambda *args: torch.zeros(1))


def test_sham_draw_does_not_use_source_tree_and_batch_attachment():
    prior = TreeMixturePrior('fixed').double()
    numbers = [6, 6, 1, 1]
    left = context_tree(prior, numbers, 0, 1, [(0, 1), (1, 2), (2, 3)], 71, 'sham')
    right = context_tree(prior, numbers, 0, 1, [(0, 1), (0, 2), (0, 3)], 71, 'sham')
    assert left == right
    graph, nbi, _ = graph_batch((3, 4))
    matrices = [tree_features(3, [(0, 1), (1, 2)]), tree_features(4, left)]
    attach_context(graph, matrices, nbi)
    source, destination = graph.edges()
    for edge, (i, j) in enumerate(zip(source.tolist(), destination.tolist())):
        batch = int(nbi[i]);offset = 0 if batch == 0 else 3
        torch.testing.assert_close(graph.edata['latent_tree_context'][edge], matrices[batch][i-offset, j-offset])
