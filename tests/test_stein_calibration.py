import torch

from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.proposal_score import invariant_stein_rows
from cfm_mol.stein_calibration import (radial_score_features, size_score_features,
    score_moments, fit_stein_correction, angular_potentials, angular_score_features)


def test_radial_features_match_existing_moments_and_symmetries():
    torch.manual_seed(9117)
    x = torch.randn(5, 4, 3, dtype=torch.float64); x -= x.mean(1, keepdim=True)
    score = torch.randn_like(x); score -= score.mean(1, keepdim=True)
    vectors, div = radial_score_features(x)
    moments = score_moments(score, vectors, div)
    previous = invariant_stein_rows(x, score)
    torch.testing.assert_close(moments[:, 0]/9, previous['dilation'])
    for i, name in enumerate(['radial_1', 'radial_2', 'radial_3'], 1):
        torch.testing.assert_close(moments[:, i], previous[name])
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    order = torch.tensor([2, 0, 3, 1])
    transformed, changed_div = radial_score_features(x[:, order]@rotation+2.)
    torch.testing.assert_close(transformed, vectors[:, :, order]@rotation, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(changed_div, div, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(vectors.sum(2), torch.zeros_like(vectors.sum(2)), atol=1e-14, rtol=0)


def test_size_feature_divergence_matches_independent_autograd():
    basis = centered_orthonormal_basis(3)
    z = torch.tensor([.2, -.3, .5, .4, .6, -.2], dtype=torch.float64, requires_grad=True)
    x = (basis@z.reshape(2, 3))[None]
    vectors, div = size_score_features(x)
    for i, scale in enumerate([3., 5.]):
        function = lambda v: torch.exp(-v.square().sum()/(2*scale**2))
        gradient, = torch.autograd.grad(function(z), z, retain_graph=True)
        expected = basis@gradient.reshape(2, 3)
        torch.testing.assert_close(vectors[0, i], expected, atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(div[0, i], torch.autograd.functional.hessian(function, z).trace(), atol=1e-13, rtol=1e-13)


def test_known_gaussian_projection_recovers_score_and_risk_difference():
    # Exact second moments in COM dimension3; no Monte Carlo approximation.
    basis = centered_orthonormal_basis(2)
    z = torch.cat([torch.eye(3), -torch.eye(3)]).double()*3**.5
    x = torch.einsum('nk,bkd->bnd', basis, z.reshape(6, 1, 3))
    features = x[:, None]; div = x.new_full((6, 1), 3.)
    base = -.4*x; truth = -x
    correction = fit_stein_correction(base, features, div, base_tail_precision=.4, ridge=0.)
    actual = correction.apply(base, features)
    torch.testing.assert_close(actual, truth, atol=1e-14, rtol=0)
    actual_risk_change = (actual-truth).square().sum((1, 2))-(base-truth).square().sum((1, 2))
    torch.testing.assert_close(correction.risk_difference_rows(base, features, div).mean(), actual_risk_change.mean(), atol=1e-13, rtol=0)


def test_tail_constraint_preserves_normalizability_and_improves_population_risk():
    basis = centered_orthonormal_basis(2)
    z = torch.cat([torch.eye(3), -torch.eye(3)]).double()*30**.5
    x = torch.einsum('nk,bkd->bnd', basis, z.reshape(6, 1, 3))
    base = -x; truth = -.1*x; features = x[:, None]; div = x.new_full((6, 1), 3.)
    correction = fit_stein_correction(base, features, div, base_tail_precision=1., ridge=0.)
    assert correction.tail_constraint_active
    assert correction.corrected_tail_precision == .5
    assert float((correction.apply(base, features)-truth).square().sum()) < float((base-truth).square().sum())


def test_angular_probe_gradient_trace_and_permutation_invariance():
    torch.manual_seed(9118)
    basis = centered_orthonormal_basis(3)
    z = torch.randn(6, dtype=torch.float64)
    x = (basis@z.reshape(2, 3))[None]
    vectors, div = angular_score_features(x)
    for index in range(2):
        function = lambda v: angular_potentials((basis@v.reshape(2, 3))[None])[0, index]
        gradient = torch.autograd.functional.jacobian(function, z)
        hessian = torch.autograd.functional.hessian(function, z)
        torch.testing.assert_close(vectors[0, index], basis@gradient.reshape(2, 3), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(div[0, index], hessian.trace(), atol=1e-12, rtol=1e-12)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    order = torch.tensor([2, 0, 1])
    torch.testing.assert_close(angular_potentials(x[:, order]@rotation+3.), angular_potentials(x), atol=1e-12, rtol=1e-12)
