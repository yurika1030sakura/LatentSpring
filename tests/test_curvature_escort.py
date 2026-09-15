import torch

from cfm_mol.curvature_escort import (centered_basis, affine_parameters, affine_candidates,
    complete_log_weights, fit_curvature)


def test_anisotropic_quadratic_escort_has_constant_complete_work():
    torch.manual_seed(1209)
    basis = centered_basis(4)
    d = basis.shape[1]
    a = torch.randn(4, 3, dtype=torch.float64)
    a -= a.mean(0)
    f = torch.randn(4, 3, dtype=torch.float64)
    f -= f.mean(0)
    raw = torch.randn(d, d, dtype=torch.float64)
    h = raw@raw.T+torch.eye(d, dtype=torch.float64)
    sigma, kT = .2, .3
    matrix, delta, logdet = affine_parameters(f, sigma, kT, h, basis)
    noise = torch.randn(2048, 4, 3, dtype=torch.float64)
    noise -= noise.mean(1, keepdim=True)
    source, proposal = affine_candidates(a, noise, sigma, matrix, delta, basis)
    y = (proposal-a).reshape(-1, 12)@basis
    force = basis.T@f.flatten()
    energy = -(y*force).sum(-1)+.5*(y@h*y).sum(-1)
    logw = complete_log_weights(a, source, proposal, 0., energy, sigma, kT, logdet,
        torch.ones(len(y), dtype=torch.bool))
    precision = torch.eye(d, dtype=torch.float64)/sigma**2+h/kT
    exact_logz = -.5*torch.linalg.slogdet(torch.eye(d, dtype=torch.float64)+sigma**2*h/kT)[1]
    exact_logz += .5*(force/kT)@torch.linalg.solve(precision, force/kT)
    torch.testing.assert_close(logw, exact_logz.expand_as(logw), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(matrix@matrix.T*sigma**2, torch.linalg.inv(precision), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(delta, torch.linalg.solve(precision, force/kT), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(proposal.mean(1), torch.zeros(len(y),3,dtype=y.dtype), atol=1e-12,rtol=0)


def test_secant_fit_and_affine_map_are_rotation_equivariant():
    torch.manual_seed(1211)
    basis = centered_basis(5)
    n, d = 5, basis.shape[1]
    raw = torch.randn(d,d,dtype=torch.float64)
    h = raw@raw.T
    displacement = torch.randn(8,n,3,dtype=torch.float64)*.02
    displacement -= displacement.mean(1,keepdim=True)
    forces = ((displacement.reshape(8,-1)@basis)@h@basis.T).reshape_as(displacement)
    fitted,kappa = fit_curvature(displacement,forces,basis)
    rotation = torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))[0]
    fitted_rot,kappa_rot = fit_curvature(displacement@rotation,forces@rotation,basis)
    force = torch.randn(n,3,dtype=torch.float64);force-=force.mean(0)
    anchor = torch.zeros(n,3,dtype=torch.float64)
    m,shift,det = affine_parameters(force,.02,.025,fitted,basis)
    mr,shiftr,detr = affine_parameters(force@rotation,.02,.025,fitted_rot,basis)
    _, y = affine_candidates(anchor,displacement,.02,m,shift,basis)
    _, yr = affine_candidates(anchor,displacement@rotation,.02,mr,shiftr,basis)
    torch.testing.assert_close(yr,y@rotation,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(det,detr,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(kappa,kappa_rot,atol=1e-10,rtol=1e-10)
