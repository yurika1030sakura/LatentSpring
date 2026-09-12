import math
import torch
from cfm_mol.normalized_site_guide import (NormalizedSiteGuide, mixture_log_prob,
    mixture_surface_score, mixture_draw)
from cfm_mol.joint_chemical_geometry import direction_parameters


def fixture():
    g = torch.Generator().manual_seed(2201)
    x = torch.randn(3, 7, 3, dtype=torch.float64, generator=g)
    x -= x.mean(1, keepdim=True)
    numbers = torch.tensor([6, 16, 9, 1, 8, 15, 17])
    bonds = torch.zeros(3, 7, 7, dtype=x.dtype)
    for i, j in [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (1, 6)]:
        bonds[:, i, j] = bonds[:, j, i] = 1
    electronic = torch.tensor([0., 1., .02585], dtype=x.dtype)
    roots = torch.tensor([[2, 0], [3, 0], [4, 1]])
    return x, bonds, numbers, electronic, roots, g


def test_normalized_mixture_surface_derivative_and_zero_parameter_gradients():
    g = torch.Generator().manual_seed(2202)
    eta = torch.randn(4, 5, 3, dtype=torch.float64, generator=g)*2
    weights = torch.randn(4, 5, dtype=eta.dtype, generator=g).log_softmax(1)
    u = torch.randn(4, 3, dtype=eta.dtype, generator=g)
    u = (u/u.norm(dim=1, keepdim=True)).requires_grad_()
    grad = torch.autograd.grad(mixture_log_prob(u, eta, weights).sum(), u)[0]
    projected = grad-(grad*u).sum(1, keepdim=True)*u
    torch.testing.assert_close(projected, mixture_surface_score(u, eta, weights))
    # Uniform-limit force fitting must support derivatives wrt component params.
    zero = torch.zeros_like(eta, requires_grad=True)
    loss = (mixture_surface_score(u.detach(), zero, weights)-torch.ones_like(u)).square().mean()
    dz = torch.autograd.grad(loss, zero)[0]
    assert torch.isfinite(dz).all() and float(dz.norm()) > 0
    # Independent sphere quadrature checks the actual normalized density.
    grid = torch.quasirandom.SobolEngine(2, scramble=True, seed=2203).draw(65536).double()
    z = 2*grid[:, 0]-1
    phi = 2*math.pi*grid[:, 1]
    sphere = torch.stack([(1-z*z).sqrt()*phi.cos(), (1-z*z).sqrt()*phi.sin(), z], 1)
    integral = 4*math.pi*mixture_log_prob(sphere, eta[0:1].expand(len(grid), -1, -1), weights[0:1].expand(len(grid), -1)).exp().mean()
    assert abs(float(integral)-1) < 2e-4


def test_masked_density_invariance_permutation_reflection_and_continuous_descriptors():
    x, bonds, numbers, electronic, roots, generator = fixture()
    model = NormalizedSiteGuide().double()
    with torch.no_grad():
        for head in [model.center_weight, model.component_weight, model.offset_weight]:
            head.weight.normal_(0, .3)
    parameters, log_weights = model(x, bonds, numbers, electronic, roots)
    u = x[torch.arange(3), roots[:, 0]]-x[torch.arange(3), roots[:, 1]]
    radius = u.norm(dim=1, keepdim=True)
    u /= radius
    q = mixture_log_prob(u, parameters, log_weights)
    moved = x.clone()
    for index, (leaf, anchor) in enumerate(roots):
        new_u = torch.randn(3, dtype=x.dtype, generator=generator)
        moved[index, leaf] = x[index, anchor]+radius[index]*new_u/new_u.norm()
    moved -= moved.mean(1, keepdim=True)
    p2, w2 = model(moved, bonds, numbers, electronic, roots)
    torch.testing.assert_close(p2, parameters, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(w2, log_weights)
    perm = torch.tensor([6, 4, 2, 0, 5, 3, 1])
    lookup = torch.argsort(perm)
    rotation = torch.linalg.qr(torch.randn(3, 3, dtype=x.dtype, generator=generator))[0]
    rotation[:, 0] *= -torch.linalg.det(rotation)
    pt, wt = model(x[:, perm]@rotation, bonds[:, perm][:, :, perm], numbers[perm], electronic, lookup[roots])
    torch.testing.assert_close(mixture_log_prob(u@rotation, pt, wt), q, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(mixture_surface_score(u@rotation, pt, wt), mixture_surface_score(u, parameters, log_weights)@rotation)
    assert not any(isinstance(module, torch.nn.Embedding) and module.num_embeddings == 119 for module in model.modules())
    torch.testing.assert_close(model.atomic_features[17, 0]-model.atomic_features[16, 0], torch.tensor(1/118, dtype=x.dtype))
    for index in range(3):
        pp, ww = model(x[index:index+1], bonds[index:index+1], numbers, electronic, roots[index:index+1])
        torch.testing.assert_close(pp[0], parameters[index])
        torch.testing.assert_close(ww[0], log_weights[index])


def test_initial_vector_matches_physical_prior_and_isolated_pair_is_uniform():
    x, bonds, numbers, electronic, roots, _ = fixture()
    model = NormalizedSiteGuide(mixture=False).double()
    params, weights = model(x, bonds, numbers, electronic, roots)
    for index, root in enumerate(roots.tolist()):
        eta, _ = direction_parameters(x[index], bonds[index], numbers, electronic, root,
                                      kind='site', model=None, site_concentration=10.)
        torch.testing.assert_close(params[index, 0], eta[0])
    pair = torch.tensor([[[-.5, 0, 0], [.5, 0, 0]]], dtype=x.dtype)
    pair_bonds = torch.tensor([[[0., 1.], [1., 0.]]], dtype=x.dtype)
    pp, ww = NormalizedSiteGuide().double()(pair, pair_bonds, torch.tensor([1, 1]), electronic, torch.tensor([[0, 1]]))
    torch.testing.assert_close(pp, torch.zeros_like(pp))
    torch.testing.assert_close(mixture_log_prob(torch.tensor([[1., 0, 0]], dtype=x.dtype), pp, ww), torch.tensor([-math.log(4*math.pi)], dtype=x.dtype))


def test_sampler_moments_and_mixture_mh_preserve_known_directional_target():
    from cfm_mol.spherical_proposal import vmf_sample, vmf_log_prob
    generator = torch.Generator().manual_seed(2204)
    n = 16384
    eta = torch.tensor([[[0., 0., 3.], [0., 0., -1.]]], dtype=torch.float64).expand(n, -1, -1)
    weights = torch.tensor([[.3, .7]], dtype=eta.dtype).log().expand(n, -1)
    drawn, _ = mixture_draw(eta, weights, generator=generator)
    expected = .3*(1/math.tanh(3)-1/3)-.7*(1/math.tanh(1)-1)
    assert abs(float(drawn[:, 2].mean())-expected) < .015
    target = torch.tensor([[0., 0., 2.]], dtype=eta.dtype).expand(n, -1)
    x, _ = vmf_sample(target, generator=generator)
    for _ in range(8):
        y, _ = mixture_draw(eta, weights, generator=generator)
        ratio = vmf_log_prob(y, target)-vmf_log_prob(x, target)+mixture_log_prob(x, eta, weights)-mixture_log_prob(y, eta, weights)
        take = torch.rand(n, dtype=x.dtype, generator=generator).log() < ratio.clamp_max(0)
        x = torch.where(take[:, None], y, x)
    assert abs(float(x[:, 2].mean())-(1/math.tanh(2)-.5)) < .018


def test_defensive_joint_uses_marginal_densities_and_reverses_both_orders():
    from cfm_mol.chemical_moves import covalent_radii
    from cfm_mol.chemical_path_guide import exchanged_bond_graph
    from cfm_mol.joint_chemical_geometry import defensive_joint_proposal, joint_geometry_proposal
    x, bonds, numbers, electronic, _, generator = fixture()
    x, bonds = x[0], bonds[0]
    action = (2, 6, 0, 1)
    radii = covalent_radii(numbers)
    model = NormalizedSiteGuide().double()
    with torch.no_grad():
        model.center_weight.weight.normal_(0, .8)
    for order in [0, 1]:
        kwargs = dict(order=order, model=model, radial_width=.15)
        y, q, trace = defensive_joint_proposal(x, bonds, numbers, electronic, radii, action, generator=generator, **kwargs)
        _, evaluated, _ = defensive_joint_proposal(x, bonds, numbers, electronic, radii, action, observed=y, **kwargs)
        torch.testing.assert_close(evaluated, q)
        qs = []
        for kind in ['site', 'normalized_site']:
            _, value, _ = joint_geometry_proposal(x, bonds, numbers, electronic, radii, action, observed=y, kind=kind, **kwargs)
            qs.append(value)
        torch.testing.assert_close(torch.logsumexp(torch.stack(qs), 0)-math.log(2), q)
        assert float(q-qs[0]) >= -math.log(2)-1e-10
        changed = exchanged_bond_graph(bonds, action)
        recovered, qr, _ = defensive_joint_proposal(y, changed, numbers, electronic, radii, (2, 6, 1, 0), observed=x, **kwargs)
        torch.testing.assert_close(recovered, x)
        assert torch.isfinite(qr)
        # Both directional laws and both augmented root orders transform with
        # an atom permutation. This includes swapping the canonical leaf order.
        perm = torch.tensor([6, 4, 2, 0, 5, 3, 1])
        lookup = torch.argsort(perm)
        i, j, k, l = [int(lookup[a]) for a in action]
        perm_action = (j, i, l, k) if i > j else (i, j, k, l)
        rotation = -torch.eye(3, dtype=x.dtype)
        _, qt, _ = defensive_joint_proposal(x[perm]@rotation, bonds[perm][:, perm], numbers[perm], electronic,
            radii[perm], perm_action, observed=y[perm]@rotation,
            **dict(kwargs, order=1-order if i > j else order))
        torch.testing.assert_close(qt, q, atol=1e-8, rtol=1e-9)


def test_defensive_mh_probability_flow_dominates_physical_component():
    torch.manual_seed(2205)
    pi = torch.tensor([.05, .1, .15, .7], dtype=torch.float64)
    physical = torch.rand(4, 4, dtype=pi.dtype)
    physical /= physical.sum(1, keepdim=True)
    poor = torch.full((4, 4), .001, dtype=pi.dtype)
    poor[:, 0] = .997
    def kernel(q):
        flow = torch.minimum(pi[:, None]*q, pi[None, :]*q.T)
        off_diagonal = flow/pi[:, None]
        off_diagonal.fill_diagonal_(0.)
        return off_diagonal+torch.diag(1-off_diagonal.sum(1))
    mixed = kernel(.5*physical+.5*poor)
    baseline = kernel(physical)
    torch.testing.assert_close(pi@mixed, pi)
    torch.testing.assert_close(mixed.sum(1), torch.ones_like(pi))
    off = ~torch.eye(4, dtype=torch.bool)
    assert (mixed[off] >= .5*baseline[off]-1e-12).all()
