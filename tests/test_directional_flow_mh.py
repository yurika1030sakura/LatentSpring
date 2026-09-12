import math
import torch
from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.directional_flow_mh import conditional_auxiliary_log_prob


def linear_transport(x, a, directions, basis, s=1.15, q=1.3, t=5.):
    z = torch.einsum('nk,bnd->bkd', basis, x)
    e = a-x
    projected = torch.einsum('nk,bnd->bkd', basis, e)
    forward_z = s*z+t*projected
    reverse_e = e/q
    reverse_z = (z-t*torch.einsum('nk,bnd->bkd', basis, reverse_e))/s
    inverse = directions.bool()[:, None, None]
    yy = torch.where(inverse, reverse_z, forward_z)
    ee = torch.where(inverse, reverse_e, q*e)
    y = torch.einsum('nk,bkd->bnd', basis, yy)
    dimension = z.shape[1]*3
    volume = (dimension*math.log(s)+3*x.shape[1]*math.log(q))*torch.where(directions.bool(), -1., 1.).double()
    return y, y+ee, volume


def test_auxiliary_has_full_dimension_and_keeps_its_centroid_probability():
    x = torch.zeros(2, 5, 3, dtype=torch.float64)
    a = x.clone()
    a[1, :, 0] = .2
    logp = conditional_auxiliary_log_prob(a, x, .1)
    assert abs(float(logp[0])+15*math.log(.1*math.sqrt(2*math.pi))) < 1e-12
    assert abs(float(logp[1]-logp[0])+5*.2**2/(2*.1**2)) < 1e-12


def test_full_joint_inverse_jacobian_and_acceptance_ratio_reversal():
    rng = torch.Generator().manual_seed(2401)
    n = 5
    basis = centered_orthonormal_basis(n)
    d = 3*(n-1)
    z = torch.randn(d+3*n, dtype=torch.float64, generator=rng)
    def transform(value, direction):
        x = (basis@value[:d].reshape(n-1, 3))[None]
        a = value[d:].reshape(1, n, 3)
        y, b, v = linear_transport(x, a, torch.tensor([direction]), basis)
        return torch.cat([(basis.T@y[0]).flatten(), b.flatten()]), v[0]
    out, volume = transform(z, 0)
    recovered, inverse_volume = transform(out, 1)
    torch.testing.assert_close(recovered, z, atol=1e-10, rtol=1e-10)
    assert abs(float(volume+inverse_volume)) < 1e-12
    jac = torch.autograd.functional.jacobian(lambda z: transform(z, 0)[0], z)
    torch.testing.assert_close(torch.linalg.slogdet(jac)[1], volume)
    x = (basis@z[:d].reshape(n-1, 3))[None]
    a = z[d:].reshape(1, n, 3)
    y = (basis@out[:d].reshape(n-1, 3))[None]
    b = out[d:].reshape(1, n, 3)
    ratio = -.5*(y.square()-x.square()).sum()+conditional_auxiliary_log_prob(b, y, .1)[0]-conditional_auxiliary_log_prob(a, x, .1)[0]+volume
    backward = -.5*(x.square()-y.square()).sum()+conditional_auxiliary_log_prob(a, x, .1)[0]-conditional_auxiliary_log_prob(b, y, .1)[0]+inverse_volume
    assert abs(float(ratio+backward)) < 1e-10


def test_known_gaussian_stationarity_and_missing_auxiliary_negative_control():
    rng = torch.Generator().manual_seed(2402)
    n, count = 5, 32768
    basis = centered_orthonormal_basis(n)
    initial = torch.einsum('nk,bkd->bnd', basis, torch.randn(count, n-1, 3, dtype=torch.float64, generator=rng))
    def run(omit):
        x = initial.clone()
        for _ in range(32):
            a = x+.1*torch.randn(x.shape, dtype=x.dtype, generator=rng)
            directions = torch.randint(2, (count,), generator=rng)
            y, b, volume = linear_transport(x, a, directions, basis)
            ratio = -.5*(y.square()-x.square()).sum((1, 2))+volume
            if not omit:
                ratio += conditional_auxiliary_log_prob(b, y, .1)-conditional_auxiliary_log_prob(a, x, .1)
            accepted = torch.rand(count, dtype=x.dtype, generator=rng).log() < ratio.clamp_max(0)
            x = torch.where(accepted[:, None, None], y, x)
        return x
    good = run(False)
    assert abs(float(good.square().sum((1, 2)).mean())-3*(n-1)) < .15
    assert float(good.mean(0).abs().max()) < .03
    bad = run(True)
    assert abs(float(bad.square().sum((1, 2)).mean())-3*(n-1)) > .5


def test_real_chemical_interface_query_counts_and_recorded_ratio():
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from cfm_mol.chemical_sampler import ChemicalTarget
    from cfm_mol.directional_flow_mh import directional_flow_transition
    molecule = Chem.AddHs(Chem.MolFromSmiles('CCO'))
    assert AllChem.EmbedMolecule(molecule, randomSeed=2403) == 0
    x = torch.tensor(molecule.GetConformer().GetPositions(), dtype=torch.float64)
    x -= x.mean(0)
    class Oracle:
        evaluated = 0
        def evaluate_chunked(self, x, max_request):
            self.evaluated += len(x)
            return .5*x.square().sum((1, 2)), -x
    class Transport:
        aux_scale = .1
        def transform(self, x, a, d):
            y, b, volume = linear_transport(x, a, d, centered_orthonormal_basis(x.shape[1]), s=1.005, q=1.01, t=.1)
            return dict(positions=y, auxiliary=b, log_volume=volume)
    oracle = Oracle()
    target = ChemicalTarget(oracle, dict(numbers=[a.GetAtomicNum() for a in molecule.GetAtoms()], charge=0, spin_multiplicity=1), 1., .1)
    states = target.evaluate([target.coordinate_state(x), target.coordinate_state(-x)], phase='initial')
    rng = torch.Generator().manual_seed(2404)
    valid = 0
    for step in range(4):
        old = list(states)
        states, rows = directional_flow_transition(target, states, Transport(), generator=rng, phase=str(step))
        for before, row in zip(old, rows):
            if not row['valid']:
                assert not row['accepted']
                continue
            valid += 1
            after = target.states[row['new_state_id']]
            ratio = -.55*float(after['positions'].square().sum()-before['positions'].square().sum())
            ratio += float(row['reverse_auxiliary_log_prob']-row['forward_auxiliary_log_prob']+row['log_volume'])
            assert abs(ratio-row['log_acceptance_ratio']) < 1e-8
            assert row['accepted'] == (row['log_uniform'] < min(0., ratio))
    assert valid > 0 and oracle.evaluated == 4+2*valid
