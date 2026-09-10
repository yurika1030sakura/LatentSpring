import torch

from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.proposal_score import ProposalEnergyCritic, invariant_stein_rows


def test_energy_critic_score_is_equivariant_and_conservative():
    torch.manual_seed(9101)
    model = ProposalEnergyCritic(hidden=8, radial=8).double()
    torch.nn.init.normal_(model.readout[-1].weight, std=.1)
    basis = centered_orthonormal_basis(4)
    z = torch.randn(2, 9, dtype=torch.float64)
    numbers = torch.tensor([1, 6, 8, 1])
    electronic = torch.tensor([.2, 0., -3.65], dtype=torch.float64)
    x = torch.einsum('nk,bkd->bnd', basis, z.reshape(2, 3, 3))
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    permutation = torch.tensor([3, 1, 2, 0])
    changed = (x[:, permutation]@rotation)+torch.tensor([1., 2., 3.])
    torch.testing.assert_close(model(x, numbers, electronic), model(changed, numbers[permutation], electronic), atol=1e-11, rtol=1e-11)
    score = model.score(z, basis, numbers, electronic)
    changed_z = torch.einsum('nk,bnd->bkd', basis, changed).flatten(1)
    changed_score = model.score(changed_z, basis, numbers[permutation], electronic)
    original_cartesian = torch.einsum('nk,bkd->bnd', basis, score.reshape(2, 3, 3))
    expected = torch.einsum('nk,bnd->bkd', basis, original_cartesian[:, permutation]@rotation).flatten(1)
    torch.testing.assert_close(changed_score, expected, atol=1e-10, rtol=1e-10)
    direction = torch.randn_like(z)
    h = 1e-5
    energy = lambda u: model(torch.einsum('nk,bkd->bnd', basis, u.reshape(2, 3, 3)), numbers, electronic)
    fd = (energy(z+h*direction)-energy(z-h*direction))/(2*h)
    torch.testing.assert_close(fd, -(score*direction).sum(-1), atol=1e-8, rtol=1e-7)


def test_radial_stein_divergence_matches_independent_hessian_trace():
    basis = centered_orthonormal_basis(3)
    z = torch.tensor([.3, -.2, .8, 1., .5, -.4], dtype=torch.float64, requires_grad=True)
    def probe(z, center):
        x = basis@z.reshape(2, 3)
        pairs = torch.triu_indices(3, 3, 1)
        radius = ((x[pairs[0]]-x[pairs[1]]).square().sum(-1)+1e-8).sqrt()
        return torch.exp(-.5*((radius-center)/.5)**2).mean()
    x = (basis@z.reshape(2, 3))[None]
    rows = invariant_stein_rows(x, torch.zeros_like(x))
    for center in [1., 2., 3.]:
        hessian = torch.autograd.functional.hessian(lambda u: probe(u, center), z)
        torch.testing.assert_close(rows[f'radial_{center:g}'][0], hessian.trace(), atol=1e-10, rtol=1e-10)
