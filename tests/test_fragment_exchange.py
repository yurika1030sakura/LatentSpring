import math
import pytest
import torch
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.fragment_exchange import (fragment_exchange_actions, fragment_bond_graph,
    inverse_fragment_action, fragment_exchange_map, fragment_force_responses, alignment_rotation)


def fixture():
    bonds = torch.zeros(9, 9, dtype=torch.float64)
    for i, j in [(0, 1), (0, 7), (1, 8), (0, 2), (2, 3), (1, 4), (4, 5), (5, 6)]:
        bonds[i, j] = bonds[j, i] = 1
    action = dict(roots=(2, 4, 0, 1), fragments=((2, 3), (4, 5, 6)))
    rng = torch.Generator().manual_seed(2301)
    x = torch.randn(9, 3, dtype=torch.float64, generator=rng)
    x -= x.mean(0)
    return x, bonds, action, rng


def test_eligibility_graph_inverse_and_permutation():
    _, bonds, action, _ = fixture()
    actions = fragment_exchange_actions(bonds, max_fragment_atoms=3)
    assert action in actions
    assert action not in fragment_exchange_actions(bonds, max_fragment_atoms=2)
    assert len(actions) == len(set(a['roots'] for a in actions))
    for a in actions:
        changed = fragment_bond_graph(bonds, a)
        inverse = inverse_fragment_action(a)
        assert inverse in fragment_exchange_actions(changed, max_fragment_atoms=3)
        torch.testing.assert_close(fragment_bond_graph(changed, inverse), bonds)
        torch.testing.assert_close(changed.sum(1), bonds.sum(1))
    perm = torch.tensor([6, 4, 2, 0, 5, 3, 1, 8, 7])
    lookup = torch.argsort(perm)
    def mapped(a):
        i, j, k, l = [int(lookup[v]) for v in a['roots']]
        first, second = [tuple(sorted(int(lookup[v]) for v in atoms)) for atoms in a['fragments']]
        if i > j:
            i, j, k, l, first, second = j, i, l, k, second, first
        return dict(roots=(i, j, k, l), fragments=(first, second))
    assert sorted([mapped(a) for a in actions], key=lambda a: a['roots']) == fragment_exchange_actions(bonds[perm][:, perm], max_fragment_atoms=3)


def test_complete_augmented_jacobian_inverse_and_singleton_limit():
    x, _, action, rng = fixture()
    basis = centered_orthonormal_basis(len(x))
    dimension = 3*(len(x)-1)
    for a in [action, dict(roots=(3, 4, 2, 1), fragments=((3,), (4, 5, 6))),
              dict(roots=(3, 6, 2, 5), fragments=((3,), (6,)))]:
        initial = torch.cat([(basis.T@x).flatten(), torch.randn(6, dtype=x.dtype, generator=rng),
                             torch.tensor([.7, -1.2], dtype=x.dtype)])
        def transform(z, move):
            xyz = basis@z[:dimension].reshape(len(x)-1, 3)
            y, old, angle, _ = fragment_exchange_map(xyz, move, z[dimension:dimension+6].reshape(2, 3), z[-2:])
            return torch.cat([(basis.T@y).flatten(), old.flatten(), angle])
        transformed = transform(initial, a)
        recovered = transform(transformed, inverse_fragment_action(a))
        torch.testing.assert_close(recovered, initial, atol=1e-9, rtol=1e-9)
        jacobian = torch.autograd.functional.jacobian(lambda z: transform(z, a), initial)
        assert abs(float(torch.linalg.slogdet(jacobian)[1])) < 1e-8
        reverse = torch.autograd.functional.jacobian(lambda z: transform(z, inverse_fragment_action(a)), transformed)
        torch.testing.assert_close(reverse@jacobian, torch.eye(len(initial), dtype=x.dtype), atol=1e-8, rtol=1e-8)


def test_internal_geometry_passive_coordinates_and_orthogonal_covariance():
    x, _, action, rng = fixture()
    vectors = torch.randn(2, 3, dtype=x.dtype, generator=rng)
    torsions = torch.tensor([1.4, -.4], dtype=x.dtype)
    y, old, reverse, rotations = fragment_exchange_map(x, action, vectors, torsions)
    for atoms in action['fragments']:
        ids = list(atoms)
        torch.testing.assert_close(torch.cdist(x[ids], x[ids]), torch.cdist(y[ids], y[ids]), atol=1e-9, rtol=1e-9)
    passive = [0, 1, 7, 8]
    torch.testing.assert_close(y[passive]-y[0], x[passive]-x[0])
    torch.testing.assert_close(torch.linalg.det(rotations), torch.ones(2, dtype=x.dtype))
    perm = torch.tensor([6, 4, 2, 0, 5, 3, 1, 8, 7])
    lookup = torch.argsort(perm)
    i, j, k, l = [int(lookup[a]) for a in action['roots']]
    first, second = [tuple(sorted(int(lookup[a]) for a in f)) for f in action['fragments']]
    a = dict(roots=(j, i, l, k), fragments=(second, first))
    yp, _, _, _ = fragment_exchange_map(x[perm], a, vectors[[1, 0]], torsions[[1, 0]])
    torch.testing.assert_close(yp, y[perm], atol=1e-9, rtol=1e-9)
    for sign in [1., -1.]:
        q = torch.linalg.qr(torch.randn(3, 3, dtype=x.dtype, generator=rng))[0]
        q[:, 0] *= sign*torch.linalg.det(q)
        yt, ot, rt, _ = fragment_exchange_map(x@q, action, vectors@q, sign*torsions)
        torch.testing.assert_close(yt, y@q, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(ot, old@q)
        torch.testing.assert_close(rt, sign*reverse)
    unit = vectors/vectors.norm(dim=1, keepdim=True)
    with pytest.raises(ValueError, match='antipodal'):
        alignment_rotation(unit, -unit)
    with pytest.raises(ValueError, match='antipodal'):
        alignment_rotation(-unit, unit)


def test_collective_force_labels_match_full_coordinate_differentiation():
    x, _, action, _ = fixture()
    stiffness = torch.linspace(.3, 1.7, len(x), dtype=x.dtype)
    def energy(z):
        return .5*(stiffness[:, None]*z.square()).sum()+.2*(z[3]-z[6]).square().sum()
    xx = x.clone().requires_grad_()
    forces = -torch.autograd.grad(energy(xx), xx)[0]
    i, j, k, l = action['roots']
    w = x[i]-x[k]
    radius, unit = w.norm(), w/w.norm()
    temperature = .4
    # The response is for a fragment at its current anchor, so use a direct
    # independently assembled single-fragment map with the same alignment rule.
    from cfm_mol.fragment_exchange import axis_rotation
    def source_update(z):
        u = unit+z[:3]
        u = u/u.norm()
        rotation = axis_rotation(u, z[3])@alignment_rotation(unit, u)
        y = x.clone()
        atoms = list(action['fragments'][0])
        y[atoms] = x[k]+radius*z[4].exp()*u+(x[atoms]-x[i])@rotation.T
        y -= y.mean(0)
        return -energy(y)/temperature
    grad = torch.autograd.functional.jacobian(source_update, torch.zeros(5, dtype=x.dtype))
    response = fragment_force_responses(x, forces, i, k, action['fragments'][0], temperature)
    torch.testing.assert_close(grad[:3], response['direction'], atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(grad[3], response['torsion'], atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(grad[4], response['log_radius'], atol=1e-9, rtol=1e-9)


def test_known_full_com_gaussian_with_action_label_and_negative_control():
    x0, _, action, _ = fixture()
    basis = centered_orthonormal_basis(len(x0))
    rng = torch.Generator().manual_seed(2302)
    count = 16384
    initial = torch.einsum('nk,bkd->bnd', basis, torch.randn(count, len(x0)-1, 3, dtype=x0.dtype, generator=rng))
    labels0 = torch.randint(2, (count,), generator=rng).bool()
    sigma = 1.1
    def run(omit_ratio):
        x, label = initial.clone(), labels0.clone()
        for _ in range(12):
            proposed = sigma*torch.randn(count, 2, 3, dtype=x.dtype, generator=rng)
            angles = 2*math.pi*torch.rand(count, 2, dtype=x.dtype, generator=rng)-math.pi
            y, reverse_vectors = torch.empty_like(x), torch.empty_like(proposed)
            for flag, move in [(False, action), (True, inverse_fragment_action(action))]:
                mask = label == flag
                yy, old, _, _ = fragment_exchange_map(x[mask], move, proposed[mask], angles[mask])
                y[mask], reverse_vectors[mask] = yy, old
            ratio = -.5*(y.square()-x.square()).sum((1, 2))
            if not omit_ratio:
                ratio += (proposed.square()-reverse_vectors.square()).sum((1, 2))/(2*sigma*sigma)
            accept = torch.rand(count, dtype=x.dtype, generator=rng).log() < ratio.clamp_max(0)
            x = torch.where(accept[:, None, None], y, x)
            label ^= accept
        return x, label
    good, labels = run(False)
    dimension = 3*(len(x0)-1)
    assert abs(float(good.square().sum((1, 2)).mean())-dimension) < .25
    assert abs(float((good[:, 3]-good[:, 6]).square().sum(1).mean())-6) < .15
    assert float(good.mean(0).abs().max()) < .035
    assert abs(float(labels.double().mean())-.5) < .015
    for label in [False, True]:
        assert abs(float(good[labels == label].square().sum((1, 2)).mean())-dimension) < .35
    bad, _ = run(True)
    assert abs(float(bad.square().sum((1, 2)).mean())-dimension) > .5
