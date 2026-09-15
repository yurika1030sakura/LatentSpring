"""Affine local escorts with exact intrinsic Gaussian density correction.

The local restrained target is unchanged. Curvature is estimated from an
independent pilot; the resulting map is held fixed for production sampling.
"""
import torch


def centered_basis(n, *, dtype=torch.float64, device='cpu'):
    # Helmert contrasts form an orthonormal basis orthogonal to translation.
    basis = torch.zeros(n, n-1, dtype=dtype, device=device)
    for j in range(n-1):
        scale = ((j+1)*(j+2))**-.5
        basis[:j+1, j] = scale
        basis[j+1, j] = -(j+1)*scale
    return torch.kron(basis, torch.eye(3, dtype=dtype, device=device))


def fit_curvature(displacements, force_differences, basis, *, ridge_fraction=.01):
    """Symmetric ridge secant fit, then project curvature onto the PSD cone.

Solve min_H ||HD-G||_F^2 + lambda ||H-kappa I||_F^2 over symmetric H.
D and G are independent-pilot displacement and negative force-difference
columns in the centered coordinate basis. No production energies enter fitting.
"""
    dimension = basis.shape[1]
    identity = torch.eye(dimension, dtype=basis.dtype, device=basis.device)
    if len(displacements) == 0:
        return identity*0, basis.new_tensor(0.)
    d = basis.T @ displacements.reshape(len(displacements), -1).T
    g = basis.T @ force_differences.reshape(len(displacements), -1).T
    squared = d.square().sum().clamp_min(1e-20)
    kappa = (d*g).sum().div(squared).clamp_min(0.)
    ridge = ridge_fraction*squared/len(displacements)
    eigenvalues, eigenvectors = torch.linalg.eigh(d@d.T)
    eigenvalues = eigenvalues.clamp_min(0.)
    rhs = g@d.T+d@g.T+2*ridge*kappa*identity
    rotated = eigenvectors.T@rhs@eigenvectors
    h = eigenvectors@(rotated/(eigenvalues[:,None]+eigenvalues[None,:]+2*ridge))@eigenvectors.T
    values, vectors = torch.linalg.eigh((h+h.T)/2)
    return (vectors*values.clamp_min(0.)[None])@vectors.T, kappa


def affine_parameters(force, sigma, kT, hessian, basis):
    values, vectors = torch.linalg.eigh((hessian+hessian.T)/2)
    precision = 1+sigma**2*values/kT
    if (precision <= 0).any():
        raise ValueError('Restrained quadratic precision must be positive')
    scale = precision.rsqrt()
    matrix = (vectors*scale[None])@vectors.T
    intrinsic_force = basis.T@force.reshape(-1)
    displacement = vectors@((vectors.T@intrinsic_force)*(sigma**2/kT)/precision)
    return matrix, displacement, scale.log().sum()


def affine_candidates(anchor, noise, sigma, matrix, displacement, basis):
    source = anchor[None]+sigma*noise
    intrinsic = (source-anchor).reshape(len(source), -1)@basis
    mapped = intrinsic@matrix.T+displacement
    proposal = anchor[None]+(mapped@basis.T).reshape_as(source)
    return source, proposal


def complete_log_weights(anchor, source, proposal, anchor_energy, energy, sigma, kT, logdet, valid):
    source2 = (source-anchor).square().sum((-1,-2))
    proposal2 = (proposal-anchor).square().sum((-1,-2))
    logw = -(energy-anchor_energy)/kT+(source2-proposal2)/(2*sigma**2)+logdet
    return torch.where(valid, logw, torch.full_like(logw, -torch.inf))
