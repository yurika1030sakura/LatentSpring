"""Learned conservative features and finite-noise Stein moment estimation.

This is a frozen-feature convex score-matching diagnostic. Deep-feature score
matching and its linear solves have direct prior art; neither is claimed new.
"""
import math

import torch
from torch.nn import functional as F


def neural_head_features(model, z, basis, numbers, electronic):
    """Gradients of the existing critic's 32 invariant final-head features.

    A temporary hook preserves the original forward computation and checkpoint
    keys. Feature0 is ||x||^2/2. Parameter tensors and mode are unchanged.
    The returned features and score are observations, not actor gradients.
    """
    with torch.enable_grad():
        coordinates = z.detach().double().requires_grad_(True)
        x = torch.einsum('nk,bkd->bnd', basis, coordinates.reshape(len(z), -1, 3))
        saved = []
        hook = model.readout[-1].register_forward_pre_hook(lambda _, inputs: saved.append(inputs[0]))
        try:
            energy = model(x, numbers, electronic)
        finally:
            hook.remove()
        if len(saved) != 1:
            raise RuntimeError('Expected one final linear-head call')
        potentials = saved[0].sum(1)
        jacobians = [torch.autograd.grad(potentials[:, j].sum(), coordinates, retain_graph=True)[0]
                     for j in range(potentials.shape[1])]
        intrinsic = torch.stack([coordinates]+jacobians, 1)
        direct_score, = torch.autograd.grad(energy.sum(), coordinates)
        tail = F.softplus(model.log_confinement).detach()+1e-4
        weights = torch.cat([tail.reshape(1), model.readout[-1].weight.detach().flatten()])
        reconstructed = -(intrinsic*weights[None, :, None]).sum(1)
        error = float((reconstructed+direct_score).abs().max())
        if error > 1e-7:
            raise RuntimeError('Feature gradient does not reconstruct the saved critic score')
        vectors = torch.einsum('nk,bpkd->bpnd', basis, intrinsic.reshape(len(z), len(weights), -1, 3))
        score = torch.einsum('nk,bkd->bnd', basis, reconstructed.reshape(len(z), -1, 3))
    return vectors.detach(), score.detach(), float(tail), error


def antithetic_divergence(plus, minus, epsilon_cartesian, sigma, *, exact_scale=True):
    """Pair-level unbiased Stein divergence contribution at finite sigma.

    plus/minus are f(mean +/- sigma*epsilon); every sample has the same marginal
    q. E[epsilon.(f_plus-f_minus)/(2sigma)] = E_q div(f). This uses Gaussian
    integration by parts, not a small-sigma approximation. Pair rows, not their
    two members, are independent units for uncertainty estimates.
    """
    if plus.shape != minus.shape or plus.ndim != 4 or epsilon_cartesian.shape != plus.shape[:1]+plus.shape[2:]:
        raise ValueError('Invalid antithetic feature shapes')
    if not math.isfinite(sigma) or sigma <= 0 or not all(torch.isfinite(v).all() for v in [plus, minus, epsilon_cartesian]):
        raise ValueError('Finite features and positive sigma required')
    result = ((plus-minus)*epsilon_cartesian[:, None]).sum((-1, -2))/(2*sigma)
    if exact_scale:
        result[:, 0] = 3*(plus.shape[2]-1)
    return result


def typed_radial_features(x, numbers, centers=(.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25), width=.5):
    """Strong fixed-feature control: element-pair-specific radial potentials.

    Each potential averages over its own pair type. The number of features may
    exceed that of the learned network; this is not a parameter-matched control.
    """
    if x.ndim != 3 or numbers.shape != (x.shape[1],) or width <= 0:
        raise ValueError('Invalid typed radial inputs')
    batch, atoms, _ = x.shape
    pairs = torch.triu_indices(atoms, atoms, 1, device=x.device)
    pair_numbers = torch.stack([numbers[pairs[0]], numbers[pairs[1]]], -1).sort(-1).values
    types = sorted(set(map(tuple, pair_numbers.cpu().tolist())))
    vectors = [(x-x.mean(1, keepdim=True))[:, None]]
    divergences = [x.new_full((batch, 1), 3*(atoms-1))]
    names = ['scale']
    center_labels = [float(value) for value in centers]
    centers = x.new_tensor(center_labels)
    for first, second in types:
        mask = (pair_numbers[:, 0] == first) & (pair_numbers[:, 1] == second)
        i, j = pairs[:, mask]
        delta = x[:, i]-x[:, j]
        squared = delta.square().sum(-1)
        radius = (squared+1e-8).sqrt()
        offset = radius[..., None]-centers
        rbf = torch.exp(-.5*(offset/width)**2)
        first_derivative = -offset/width**2*rbf
        second_derivative = (offset.square()/width**4-1/width**2)*rbf
        pair_gradient = (first_derivative[..., None]*delta[:, :, None]/radius[:, :, None, None]).permute(0, 2, 1, 3)/len(i)
        field = x.new_zeros(batch, len(centers), atoms, 3)
        field.index_add_(2, i, pair_gradient); field.index_add_(2, j, -pair_gradient)
        divergence = 2*(second_derivative*squared[..., None]/radius[..., None]**2+
            first_derivative*(3/radius-squared/radius**3)[..., None]).mean(1)
        vectors.append(field); divergences.append(divergence)
        names += [f'{first}_{second}_{float(center):g}' for center in center_labels]
    return torch.cat(vectors, 1), torch.cat(divergences, 1), names
