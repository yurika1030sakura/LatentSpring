import torch
import pytest
from cfm_mol.chemical_work import PairedChemicalWork, LinearBondWork, harmonic_change, informed_log_prob


def example():
    torch.manual_seed(70)
    x = torch.randn(2, 6, 3, dtype=torch.float64)
    y = x.clone(); y[:, :2] += torch.randn(2, 2, 3, dtype=torch.float64)
    y -= y.mean(1, keepdim=True)
    b = torch.zeros(2, 6, 6, dtype=torch.float64)
    b[:, 0, 2] = b[:, 2, 0] = b[:, 1, 3] = b[:, 3, 1] = b[:, 2, 3] = b[:, 3, 2] = 1.
    c = b.clone(); c[:, 0, 2] = c[:, 2, 0] = c[:, 1, 3] = c[:, 3, 1] = 0.
    c[:, 0, 3] = c[:, 3, 0] = c[:, 1, 2] = c[:, 2, 1] = 1.
    return x, y, b, c, torch.tensor([1, 8, 6, 7, 1, 17]), torch.tensor([[0., 1., .0258]]*2).double(), torch.tensor([[0, 1]]*2)


def model(geometry=True):
    result = PairedChemicalWork(hidden=8, radial=6, geometry=geometry).double()
    for head in (result.node_energy[-1], result.pair_energy[-1]):
        torch.nn.init.normal_(head.weight, std=.2)
    return result


@pytest.mark.parametrize('geometry', [False, True])
def test_reversal_symmetry_and_passive_mask(geometry):
    x, y, b, c, z, state, active = example(); m = model(geometry)
    actual = m(x, y, b, c, z, state, active)
    torch.testing.assert_close(actual, -m(y, x, c, b, z, state, active), atol=1e-10, rtol=1e-10)
    before = m.encode(x, b, z, state, active)
    perturbed = x.clone(); perturbed[:, :2] += 50
    changed = b.clone(); changed[:, :2, :] = 2.; changed[:, :, :2] = 2.
    after = m.encode(perturbed, changed, z, state, active)
    torch.testing.assert_close(before['nodes'], after['nodes'], atol=0, rtol=0)
    torch.testing.assert_close(before['positions'], after['positions'], atol=0, rtol=0)


def test_rigid_motion_permutation_and_gradient():
    x, y, b, c, z, state, active = example(); m = model()
    actual = m(x, y, b, c, z, state, active)
    q = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))[0]; q[:, 0] *= -1
    torch.testing.assert_close(actual, m(x@q+3, y@q-2, b, c, z, state, active), atol=1e-10, rtol=1e-10)
    permutation = torch.tensor([3, 5, 1, 4, 0, 2]); inverse = permutation.argsort()
    torch.testing.assert_close(actual, m(x[:, permutation], y[:, permutation], b[:, permutation][:, :, permutation],
        c[:, permutation][:, :, permutation], z[permutation], state, inverse[active.flip(1)]), atol=1e-10, rtol=1e-10)
    actual.sum().backward()
    assert sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None) > 0


def test_graph_control_has_no_geometry_and_linear_reverses():
    x, y, b, c, z, state, active = example(); m = model(False)
    value = m(x, y, b, c, z, state, active)-harmonic_change(x, y)
    xx, yy = x*1.5, y*.3
    torch.testing.assert_close(value, m(xx, yy, b, c, z, state, active)-harmonic_change(xx, yy), atol=1e-10, rtol=1e-10)
    linear = LinearBondWork(sorted(set(z.tolist()))).double()
    torch.nn.init.normal_(linear.coefficients)
    torch.testing.assert_close(linear(x, y, b, c, z, state, active), -linear(y, x, c, b, z, state, active))


def test_policy_normalization_and_positive_support():
    work = torch.tensor([-100., 0., 100.], dtype=torch.float64)
    p = informed_log_prob(work, torch.zeros_like(work), .0258, .1).exp()
    torch.testing.assert_close(p.sum(), torch.tensor(1., dtype=torch.float64))
    assert (p >= .1/3-1e-15).all() and p[0] > p[1] > p[2]
    with pytest.raises(ValueError): informed_log_prob(work*float('nan'), work, .0258)


@pytest.mark.parametrize('control', ['uniform', 'force'])
def test_real_map_catalogue_normalization_and_mh(control):
    import math
    from test_bounded_action_geometry import pair_record
    from cfm_mol.chemical_sampler import ChemicalTarget
    from cfm_mol.chemical_work_policy import catalogue, policy, select, finish
    row = pair_record()
    class Oracle:
        evaluated = 0
        def evaluate_chunked(self, positions, max_request):
            self.evaluated += len(positions)
            return .01*positions.square().sum((1, 2)), -.02*positions
    oracle = Oracle()
    target = ChemicalTarget(oracle, dict(numbers=row['numbers'].tolist(), charge=0, spin_multiplicity=1), .026, .1)
    old = target.evaluate([target.coordinate_state(row['x'])], phase='source')[0]
    options = catalogue(target, old); probability = policy(target, old, options, control)
    assert oracle.evaluated == 2 and options['valid']
    torch.testing.assert_close(probability['log_probability'].exp().sum(), torch.tensor(1., dtype=torch.float64))
    new, record = select(options, probability, .37)
    target.evaluate([new], phase='selected')
    done = finish(target, old, new, record, control, math.log(.42))
    assert oracle.evaluated == 4
    expected = (-.06*float(new['positions'].square().sum()-old['positions'].square().sum())/.026
                + record['log_volume']+record['reverse_log_probability']-record['forward_log_probability'])
    assert abs(expected-done['log_acceptance_ratio']) < 1e-10
    assert done['accepted'] == (math.log(.42) < min(0., expected))
    if control == 'uniform':
        assert abs(done['action_log_ratio']-math.log(done['forward_valid_count']/done['reverse_valid_count'])) < 1e-12
