"""Contractive pointwise gradient maps with exact zero-centroid volume.

This is a prospective coupling primitive, not a trained molecular method.
The determinant identity is a Schur-complement identity; contractive residual
and convex-potential flows are established prior art.
"""
import math

import torch


def convex_point_map(x, raw_weights, directions, offsets, raw_scale, raw_radial, *, length=1., contraction=.25):
    """Return F_i(x_i) and its SPD3x3 derivative, with frozen context inputs.

    All context parameters are shared across points in a group. They must not
    depend on that group's active coordinates in a coupling application.
    ||DF-lambda I|| <= contraction*lambda globally. Initial zero raw weights,
    radial amplitude and log scale yield the exact identity for any directions.
    """
    if x.ndim != 3 or x.shape[-1] != 3 or not 1 <= x.shape[1] <= 200:
        raise ValueError('Point matrix must have shape[B,N,3], N<=200')
    batch = len(x)
    if raw_weights.ndim != 2 or directions.shape != raw_weights.shape+(3,) or offsets.shape != raw_weights.shape or raw_weights.shape[0] != batch or raw_weights.shape[1] < 1:
        raise ValueError('Invalid shared context shapes')
    if raw_scale.shape != (batch,) or raw_radial.shape != (batch,):raise ValueError('One scale and radial amplitude per group required')
    if not math.isfinite(length) or length <= 0 or not math.isfinite(contraction) or not 0 < contraction < 1:
        raise ValueError('Positive length and contraction in(0,1) required')
    if not all(torch.isfinite(v).all() for v in [x, raw_weights, directions, offsets, raw_scale, raw_radial]):raise ValueError('Finite inputs required')
    scale = torch.exp(.25*torch.tanh(raw_scale))
    b = directions/(length*(directions.square().sum(-1, keepdim=True)+.01).sqrt())
    # Half the contraction budget for signed softplus features, half for radial.
    weights = 2*contraction*scale[:, None]*length**2*torch.tanh(raw_weights)/raw_weights.shape[1]
    logits = torch.einsum('bnd,bmd->bnm', x, b)+offsets[:, None]
    activation = torch.sigmoid(logits)
    shift = torch.einsum('bm,bnm,bmd->bnd', weights, activation, b)
    derivative = torch.einsum('bm,bnm,bmi,bmj->bnij', weights, activation*(1-activation), b, b)
    radius = (length**2+x.square().sum(-1)).sqrt()
    radial_weight = .5*contraction*scale*length*torch.tanh(raw_radial)
    shift = shift+radial_weight[:, None, None]*x/radius[..., None]
    identity = torch.eye(3, dtype=x.dtype, device=x.device)
    derivative = derivative+radial_weight[:, None, None, None]*(identity/radius[..., None, None]-
        x[..., :, None]*x[..., None, :]/radius[..., None, None]**3)
    derivative = derivative+scale[:, None, None, None]*identity
    return scale[:, None, None]*x+shift, derivative, scale


def centered_convex_forward(x, raw_weights, directions, offsets, raw_scale, raw_radial, **options):
    """Bijective map on H: y_i=F_i(x_i)-mean_j F_j(x_j).

    On H, det(DT)=prod_i det(A_i)*det(mean_i A_i^-1), A_i=DF_i.
    The correction accounts for centering; ambient determinants alone are wrong.
    """
    if x.shape[1] < 2 or float(x.mean(1).abs().max()) > 1e-8:raise ValueError('Require centered group with at least two points')
    mapped, blocks, _ = convex_point_map(x, raw_weights, directions, offsets, raw_scale, raw_radial, **options)
    chol = torch.linalg.cholesky(blocks)
    inverse_blocks = torch.cholesky_inverse(chol)
    complement = torch.linalg.cholesky(inverse_blocks.mean(1))
    logdet = (2*chol.diagonal(dim1=-2, dim2=-1).log().sum((-1, -2))+
        2*complement.diagonal(dim1=-2, dim2=-1).log().sum(-1))
    return mapped-mapped.mean(1, keepdim=True), logdet


@torch.no_grad()
def centered_convex_inverse(y, raw_weights, directions, offsets, raw_scale, raw_radial, *, tolerance=1e-10, max_iterations=64, **options):
    """No-grad reconstruction solve on H; not a differentiable likelihood API."""
    if y.shape[1] < 2 or float(y.mean(1).abs().max()) > 1e-8:raise ValueError('Require centered inverse inputs')
    if tolerance <= 0 or max_iterations < 1:raise ValueError('Invalid inverse tolerance/limit')
    context = (raw_weights, directions, offsets, raw_scale, raw_radial)
    scale = torch.exp(.25*torch.tanh(raw_scale))[:, None, None]
    x = y/scale
    for iteration in range(max_iterations):
        mapped, _, _ = convex_point_map(x, *context, **options)
        shift = mapped-scale*x
        updated = (y-(shift-shift.mean(1, keepdim=True)))/scale
        # Eliminate accumulated roundoff in the constrained subspace.
        updated = updated-updated.mean(1, keepdim=True)
        if float((updated-x).abs().max()) <= tolerance:
            x = updated
            reconstructed, _ = centered_convex_forward(x, *context, **options)
            residual = float((reconstructed-y).abs().max())
            if residual <= tolerance:
                return x, {'iterations': iteration+1, 'maximum_residual': residual}
        x = updated
    raise RuntimeError('Centered inverse did not meet its declared residual tolerance')


@torch.no_grad()
def convex_point_inverse(y, raw_weights, directions, offsets, raw_scale, raw_radial, *, tolerance=1e-10, max_iterations=64, **options):
    """No-grad contraction inverse without a centroid constraint."""
    if not math.isfinite(tolerance) or tolerance <= 0 or max_iterations < 1:
        raise ValueError('Invalid inverse tolerance/limit')
    context = (raw_weights, directions, offsets, raw_scale, raw_radial)
    scale = torch.exp(.25*torch.tanh(raw_scale))[:, None, None]
    x = y/scale
    for iteration in range(max_iterations):
        mapped, _, _ = convex_point_map(x, *context, **options)
        updated = (y-(mapped-scale*x))/scale
        if float((updated-x).abs().max()) <= tolerance:
            reconstructed, _, _ = convex_point_map(updated, *context, **options)
            residual = float((reconstructed-y).abs().max())
            if residual <= tolerance:
                return updated, {'iterations': iteration+1, 'maximum_residual': residual}
        x = updated
    raise RuntimeError('Point inverse did not meet its declared residual tolerance')
