"""Residual score calibration in a finite conservative feature space.

Stein score estimation, score matching and quadratic projection are established
methods. This module is a candidate correction to a frozen learned score, not
a claim of new identities or globally certified score accuracy.
"""
from dataclasses import dataclass
import math

import torch


def radial_score_features(x, centers=(1., 2., 3.), width=.5, *, include_scale=True):
    """Return gradients and COM-free divergences of invariant scalar features.

    Shape: vectors[B,P,N,3], divergence[B,P]. The first potential, if requested,
    is ||Px||^2/2. Others average radial Gaussian features over unordered pairs.
    Pair radius uses sqrt(r^2+1e-8); the derivative formula matches this radius.
    """
    if x.ndim != 3 or x.shape[-1] != 3 or not 2 <= x.shape[1] <= 200 or not torch.isfinite(x).all():
        raise ValueError('Finite molecular coordinates required')
    if not math.isfinite(width) or width <= 0:
        raise ValueError('Positive finite feature width required')
    centers = x.new_tensor(centers)
    if centers.ndim != 1 or not torch.isfinite(centers).all() or len(centers) == 0:
        raise ValueError('Finite radial centers required')
    batch, atoms, _ = x.shape
    pairs = torch.triu_indices(atoms, atoms, 1, device=x.device)
    delta = x[:, pairs[0]]-x[:, pairs[1]]
    squared = delta.square().sum(-1)
    radius = (squared+1e-8).sqrt()
    offset = radius[..., None]-centers
    rbf = torch.exp(-.5*(offset/width)**2)
    first = -offset/width**2*rbf
    second = (offset.square()/width**4-1/width**2)*rbf
    pair_grad = (first[..., None]*delta[:, :, None]/radius[:, :, None, None]).permute(0, 2, 1, 3)/len(pairs[0])
    vectors = x.new_zeros(batch, len(centers), atoms, 3)
    vectors.index_add_(2, pairs[0], pair_grad)
    vectors.index_add_(2, pairs[1], -pair_grad)
    divergences = 2*(second*squared[..., None]/radius[..., None]**2+
        first*(3/radius-squared/radius**3)[..., None]).mean(1)
    if include_scale:
        vectors = torch.cat([(x-x.mean(1, keepdim=True))[:, None], vectors], 1)
        divergences = torch.cat([x.new_full((batch, 1), 3*(atoms-1)), divergences], 1)
    return vectors, divergences


def size_score_features(x, scales=(3., 5.)):
    """Unfitted assessment directions from exp(-||Px||^2/(2 a^2))."""
    centered = x-x.mean(1, keepdim=True)
    squared = centered.square().sum((1, 2))
    a = x.new_tensor(scales)
    if a.ndim != 1 or (a <= 0).any() or not torch.isfinite(a).all():
        raise ValueError('Positive finite size scales required')
    value = torch.exp(-squared[:, None]/(2*a.square()))
    vectors = -centered[:, None]*value[:, :, None, None]/a[None, :, None, None].square()
    divergences = value*(squared[:, None]/a**4-3*(x.shape[1]-1)/a.square())
    return vectors, divergences


def angular_potentials(x, scales=(2., 3.)):
    """Bounded symmetric triangle-angle probes, used only for assessment."""
    if x.ndim != 3 or x.shape[-1] != 3 or x.shape[1] < 3:
        raise ValueError('At least three atoms required for angular assessment')
    triangles = torch.combinations(torch.arange(x.shape[1], device=x.device), r=3)
    i, j, k = triangles.unbind(-1)
    angle = x.new_zeros(len(x), len(triangles))
    for vertex, first, second in [(i, j, k), (j, k, i), (k, i, j)]:
        u = x[:, first]-x[:, vertex]; v = x[:, second]-x[:, vertex]
        angle = angle+(u*v).sum(-1).square()/((u.square().sum(-1)+.01)*(v.square().sum(-1)+.01))/3
    perimeter_squared = ((x[:, i]-x[:, j]).square()+(x[:, j]-x[:, k]).square()+(x[:, k]-x[:, i]).square()).sum(-1)
    a = x.new_tensor(scales)
    if a.ndim != 1 or (a <= 0).any() or not torch.isfinite(a).all():
        raise ValueError('Positive angular envelope scales required')
    return (angle[..., None]*torch.exp(-perimeter_squared[..., None]/(2*a.square()))).mean(1)


def angular_score_features(x, scales=(2., 3.)):
    """Exact ambient Hessian trace equals COM-free trace by translation symmetry.

    This diagnostic is expensive and intended for the small condition screen,
    not a claim of scalable large-molecule Hessian evaluation.
    """
    with torch.enable_grad():
        coordinates = x.detach().requires_grad_(True)
        potentials = angular_potentials(coordinates, scales)
        vectors, divergences = [], []
        for j in range(potentials.shape[1]):
            gradient, = torch.autograd.grad(potentials[:, j].sum(), coordinates, create_graph=True, retain_graph=True)
            trace = x.new_zeros(len(x))
            for atom in range(x.shape[1]):
                for axis in range(3):
                    derivative, = torch.autograd.grad(gradient[:, atom, axis].sum(), coordinates, retain_graph=True)
                    trace = trace+derivative[:, atom, axis]
            vectors.append(gradient.detach()); divergences.append(trace.detach())
    return torch.stack(vectors, 1), torch.stack(divergences, 1)


def score_moments(score, vectors, divergences):
    if score.ndim != 3 or vectors.ndim != 4 or vectors.shape[0] != len(score) or vectors.shape[2:] != score.shape[1:]:
        raise ValueError('Molecular score and feature shapes disagree')
    if divergences.shape != vectors.shape[:2]:
        raise ValueError('One divergence per sample and feature required')
    if not all(torch.isfinite(v).all() for v in [score, vectors, divergences]):
        raise ValueError('Finite score features required')
    return (score[:, None]*vectors).sum((-1, -2))+divergences


@dataclass
class SteinCorrection:
    coefficients: torch.Tensor
    feature_scales: torch.Tensor
    ridge: float
    tail_constraint_active: bool
    corrected_tail_precision: float

    def apply(self, score, vectors):
        if vectors.shape[1] != len(self.coefficients) or vectors.shape[0] != len(score) or vectors.shape[2:] != score.shape[1:]:
            raise ValueError('Correction features differ from fitted basis')
        return score-(vectors*self.coefficients[None, :, None, None]).sum(1)

    def risk_difference_rows(self, score, vectors, divergences):
        """Unbiased expected Fisher-risk difference on independent q samples.

        For a fixed fitted correction and valid Stein integration by parts,
        E[rows] = E[||s_corrected-s_q||^2-||s_base-s_q||^2]. Finite estimates
        need uncertainty; fitting and evaluating on the same rows is optimistic.
        """
        residual = score_moments(score, vectors, divergences)
        displacement = (vectors*self.coefficients[None, :, None, None]).sum(1)
        return displacement.square().sum((-1, -2))-2*(residual*self.coefficients).sum(-1)


def fit_stein_correction(score, vectors, divergences, *, base_tail_precision, ridge=.01):
    """Ridge projection with a one-coordinate active-set tail constraint.

    Feature0 must be grad(||Px||^2/2). Other potentials must be bounded.
    Enforce c0 >= -base_tail_precision/2, retaining a positive quadratic tail.
    Zero correction is feasible. At population moments the convex minimizer
    cannot increase Fisher risk; this guarantee is not automatic for estimates.
    """
    if not math.isfinite(base_tail_precision) or base_tail_precision <= 0 or not math.isfinite(ridge) or ridge < 0:
        raise ValueError('Positive base tail precision and nonnegative ridge required')
    residual = score_moments(score, vectors, divergences).double()
    flat = vectors.detach().double().flatten(2)
    gram = torch.einsum('bpd,bqd->pq', flat, flat)/len(flat)
    return fit_stein_moments(gram, residual.mean(0), base_tail_precision=base_tail_precision, ridge=ridge)


def fit_stein_moments(gram, residual, *, base_tail_precision, ridge=.01):
    """The same constrained solve from streamed or antithetic moment estimates."""
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or residual.shape != (len(gram),):
        raise ValueError('Invalid Stein moment shapes')
    if not all(torch.isfinite(v).all() for v in [gram, residual]) or not torch.allclose(gram, gram.T, atol=1e-9, rtol=1e-9):
        raise ValueError('Finite symmetric Gram matrix required')
    if not math.isfinite(base_tail_precision) or base_tail_precision <= 0 or not math.isfinite(ridge) or ridge < 0:
        raise ValueError('Invalid tail precision or ridge')
    scales = gram.diagonal().clamp_min(1e-20).sqrt()
    normalized = gram/scales[:, None]/scales[None, :]
    system = normalized+ridge*torch.eye(len(scales), dtype=gram.dtype, device=gram.device)
    rhs = residual/scales
    unit = torch.linalg.solve(system, rhs)
    lower = -.5*base_tail_precision*scales[0]
    active = bool(unit[0] < lower)
    if active:
        unit[0] = lower
        if len(unit) > 1:
            unit[1:] = torch.linalg.solve(system[1:, 1:], rhs[1:]-system[1:, 0]*lower)
    coefficients = (unit/scales).detach()
    if not torch.isfinite(coefficients).all():
        raise FloatingPointError('Non-finite score correction')
    return SteinCorrection(coefficients, scales, ridge, active, base_tail_precision+float(coefficients[0]))
