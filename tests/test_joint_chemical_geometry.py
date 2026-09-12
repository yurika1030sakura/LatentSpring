import math
import torch
from cfm_mol.chemical_moves import covalent_radii
from cfm_mol.chemical_path_guide import exchanged_bond_graph
from cfm_mol.masked_angular_guide import MaskedAngularGuide
from cfm_mol.joint_chemical_geometry import (joint_geometry_proposal,
    radial_cartesian_log_density, normalized_direction_log_prob, direction_parameters)


def fixture():
    torch.manual_seed(2101)
    numbers = torch.tensor([6, 16, 1, 9, 1, 9])
    x = torch.tensor([[-1., 0, 0], [1., 0, 0], [-1.3, .8, .4],
                      [1.4, -.9, .8], [-1.3, -.8, -.4], [1.3, .9, -.8]], dtype=torch.float64)
    x -= x.mean(0)
    bonds = torch.zeros(6, 6, dtype=x.dtype)
    for i, j in [(0, 1), (2, 0), (3, 1), (4, 0), (5, 1)]:
        bonds[i, j] = bonds[j, i] = 1
    model = MaskedAngularGuide().double()
    with torch.no_grad():
        for head in [model.vector_weight, model.tensor_weight]:
            head.weight.normal_(0, 2)
            head.bias.normal_(0, 2)
    return x, bonds, numbers, torch.tensor([0., 1., .02585], dtype=x.dtype), model


def test_com_conditional_chart_has_constant_volume_for_both_anchor_assignments():
    # Induced six-dimensional volume when both root vectors vary and all passive
    # relative positions are held fixed; independent of roots' anchor assignment.
    x, _, _, _, _ = fixture()
    for anchors in [(0, 1), (1, 0)]:
        def chart(v):
            y = x.clone()
            y[2] = x[anchors[0]]+v[:3]
            y[3] = x[anchors[1]]+v[3:]
            return (y-y.mean(0)).flatten()
        jac = torch.autograd.functional.jacobian(chart, torch.randn(6, dtype=x.dtype))
        volume = torch.linalg.det(jac.T@jac).sqrt()
        torch.testing.assert_close(volume, volume.new_tensor(((len(x)-2)/len(x))**1.5))
    # Spherical coordinates (log r, cos(theta), phi) have Jacobian r^3.
    def spherical(z):
        ell, t, phi = z
        return ell.exp()*torch.stack([(1-t*t).sqrt()*phi.cos(), (1-t*t).sqrt()*phi.sin(), t])
    z = torch.tensor([.2, .3, 1.1], dtype=x.dtype)
    jac = torch.autograd.functional.jacobian(spherical, z)
    torch.testing.assert_close(torch.linalg.det(jac).abs(), (3*z[0]).exp())


def test_sequential_density_reversal_and_permutation_reflection_covariance():
    x, bonds, numbers, electronic, model = fixture()
    radii = covalent_radii(numbers)
    action = (2, 3, 0, 1)
    g = torch.Generator().manual_seed(2102)
    for kind in ['uniform', 'site', 'tensor']:
        for order in [0, 1]:
            kw = dict(kind=kind, order=order, model=model, radial_width=.18)
            y, q, trace = joint_geometry_proposal(x, bonds, numbers, electronic, radii, action, generator=g, **kw)
            rebuilt, evaluated, forced = joint_geometry_proposal(x, bonds, numbers, electronic, radii, action, observed=y, **kw)
            torch.testing.assert_close(rebuilt, y)
            torch.testing.assert_close(evaluated, q)
            for a, b in zip(trace['steps'], forced['steps']):
                torch.testing.assert_close(a['context'], b['context'])
                torch.testing.assert_close(a['eta'], b['eta'])
            new_bonds = exchanged_bond_graph(bonds, action)
            inverse = (2, 3, 1, 0)
            recovered, qr, _ = joint_geometry_proposal(y, new_bonds, numbers, electronic, radii, inverse, observed=x, **kw)
            torch.testing.assert_close(recovered, x)
            torch.testing.assert_close(exchanged_bond_graph(new_bonds, inverse), bonds)
            # Relabel so i/j switch canonical order; the fair augmented order
            # switches with them. Test both orders, including an O(3) reflection.
            perm = torch.tensor([5, 3, 0, 4, 2, 1])
            inv = torch.argsort(perm)
            transformed = tuple(int(inv[a]) for a in action)
            i, j, k, l = transformed
            swapped = i > j
            if swapped:
                transformed = (j, i, l, k)
            rotation = torch.linalg.qr(torch.randn(3, 3, dtype=x.dtype, generator=g))[0]
            rotation[:, 0] *= -torch.linalg.det(rotation)
            pkw = dict(kw, order=1-order if swapped else order)
            _, qt, _ = joint_geometry_proposal(x[perm]@rotation, bonds[perm][:, perm], numbers[perm], electronic,
                radii[perm], transformed, observed=y[perm]@rotation, **pkw)
            torch.testing.assert_close(qt, q, atol=1e-8, rtol=1e-8)
            assert torch.isfinite(qr)
            if kind == 'tensor':
                # A completed-endpoint first conditional is a different model.
                first = trace['steps'][0]
                eta, matrix = direction_parameters(y, new_bonds, numbers, electronic, first['root'],
                    kind=kind, model=model, site_concentration=10.)
                wrong = normalized_direction_log_prob(first['direction'][None], eta, matrix)[0]
                assert abs(float(wrong-first['angular_log_density'])) > 1e-5


def test_full_coordinate_known_target_stationarity_requires_radial_volume_factor():
    # IID exact starts from a correlated log-normal radial target and uniform
    # angles, embedded as two Cartesian roots in a COM chart. Both radii vary;
    # this checks more than sphere-only invariance. The proposal differs from
    # the target. Omitting r^3 must visibly bias the radial distribution.
    generator = torch.Generator().manual_seed(2103)
    n = 32768
    dtype = torch.float64
    mean = torch.tensor([.15, -.1], dtype=dtype)
    covariance = torch.tensor([[.09, .048], [.048, .16]], dtype=dtype)
    chol = torch.linalg.cholesky(covariance)
    precision = torch.linalg.inv(covariance)
    initial = mean+torch.randn(n, 2, dtype=dtype, generator=generator)@chol.T
    def target(log_r):
        delta = log_r-mean
        return -.5*torch.einsum('bi,ij,bj->b', delta, precision, delta)-3*log_r.sum(1)
    def run(omit_volume):
        ell = initial.clone()
        for _ in range(12):
            proposed = .5*torch.randn(n, 2, dtype=dtype, generator=generator)
            qold = radial_cartesian_log_density(ell, 0., .5).sum(1)
            qnew = radial_cartesian_log_density(proposed, 0., .5).sum(1)
            correction = qold-qnew
            if omit_volume:
                correction += 3*(ell-proposed).sum(1)
            ratio = target(proposed)-target(ell)+correction
            accept = torch.rand(n, dtype=dtype, generator=generator).log() < ratio.clamp_max(0)
            ell = torch.where(accept[:, None], proposed, ell)
        return ell
    good = run(False)
    torch.testing.assert_close(good.mean(0), mean, atol=.012, rtol=0)
    torch.testing.assert_close(torch.cov(good.T), covariance, atol=.009, rtol=0)
    # Cartesian moments also include the independently normalized solid angle.
    angles = torch.randn(n, 2, 3, dtype=dtype, generator=generator)
    angles /= angles.norm(dim=2, keepdim=True)
    vectors = good.exp()[:, :, None]*angles
    expected = (2*mean+2*covariance.diagonal()).exp()
    torch.testing.assert_close(vectors.square().sum(2).mean(0), expected, atol=.04, rtol=0)
    bad = run(True)
    assert float((bad.mean(0)-mean).abs().max()) > .12
