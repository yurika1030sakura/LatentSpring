"""Coordinate-only recovery supervision for a pretrained molecular flow.

The original CFM objective is retained on alternating updates. Recovery is an
auxiliary endpoint regression objective on perturbed training interpolants;
it is not asserted to be the velocity of the original probability path.
Local weights are smooth reference-distance neighborhoods, not bond labels.
"""
import torch
from . import matched_egnn as base, connectivity_feedback as feedback
from .chemical_moves import covalent_radii


def recovery_noise(clean, generator):
    """Masked atom errors and a displaced spatial region, in Angstrom."""
    batch, atoms, _ = clean.shape
    scales = clean.new_tensor([.05, .15, .30])
    scale = scales[torch.randint(3, (batch, 1, 1), generator=generator,
                               device=clean.device)]
    mask = torch.rand((batch, atoms, 1), generator=generator,
                      device=clean.device) < .25
    local = scale * mask * torch.randn(clean.shape, generator=generator,
                                      device=clean.device, dtype=clean.dtype)
    direction = torch.randn((batch, 1, 3), generator=generator,
                            device=clean.device, dtype=clean.dtype)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    projection = (clean * direction).sum(-1)
    region = projection > projection.median(-1, keepdim=True).values
    shift = .35 * torch.randn((batch, 1, 3), generator=generator,
                              device=clean.device, dtype=clean.dtype)
    use_region = torch.rand((batch, 1, 1), generator=generator,
                            device=clean.device) < .5
    return base.center(local + use_region * region[..., None] * shift)


def local_geometry_errors(predicted, reference, numbers):
    """Reference-relative pair distances and neighbor angles; no flat-ring rule."""
    if predicted.shape != reference.shape or numbers.shape != reference.shape[:2]:
        raise ValueError('Mismatched batched coordinates and element identities')
    radii = covalent_radii(numbers.detach().cpu().reshape(-1).tolist()).to(reference)
    radii = radii.reshape_as(numbers)
    lengths = radii[:, :, None] + radii[:, None, :]
    ref_vector = reference[:, :, None] - reference[:, None, :]
    pred_vector = predicted[:, :, None] - predicted[:, None, :]
    ref_distance = (ref_vector.square().sum(-1) + 1e-12).sqrt()
    pred_distance = (pred_vector.square().sum(-1) + 1e-12).sqrt()
    offdiag = 1 - torch.eye(reference.shape[1], device=reference.device,
                             dtype=reference.dtype)[None]
    weights = (torch.sigmoid((1.30 - ref_distance / lengths) / .10) * offdiag).detach()
    pair = ((pred_distance - ref_distance) / lengths).square()
    pair = ((pair * weights).sum((1, 2)) / weights.sum((1, 2)).clamp_min(1e-8)).mean()
    ref_unit = ref_vector / ref_distance[..., None]
    pred_unit = pred_vector / pred_distance[..., None]
    ref_gram = torch.einsum('bijd,bikd->bijk', ref_unit, ref_unit)
    pred_gram = torch.einsum('bijd,bikd->bijk', pred_unit, pred_unit)
    triple = weights[:, :, :, None] * weights[:, :, None, :] * offdiag[:, None]
    angle = ((pred_gram - ref_gram).square() * triple).sum((1, 2, 3))
    angle = (angle / triple.sum((1, 2, 3)).clamp_min(1e-8)).mean()
    return pair, angle


def recovery_loss(model, clean, numbers, spec, source, context, seed, *,
                  corrupt=False, local_geometry=False):
    if spec['kind'] != 'harmonic_fm' or model.norm_values[0] != 1.:
        raise ValueError('Recovery pilot requires the harmonic FM in Angstrom')
    rng = torch.Generator(device=clean.device).manual_seed(seed)
    x0, target = base.fm_endpoints(base.center(clean), numbers, 'harmonic_fm', source, seed)
    progress = .60 + .35 * torch.rand((len(clean), 1), generator=rng,
                                    device=clean.device, dtype=clean.dtype)
    state = (1 - progress[..., None]) * x0 + progress[..., None] * target
    noise = recovery_noise(target, rng)
    if corrupt:
        state = base.center(state + noise)
    velocities = feedback.prediction(model, state, progress, numbers, spec, context,
                                     two_pass=True, return_first=True)
    endpoint_losses = []
    pair_losses = []
    angle_losses = []
    for velocity in velocities:
        endpoint = base.center(state + (1 - progress[..., None]) * velocity)
        endpoint_losses.append((endpoint - target).square().mean())
        if local_geometry:
            pair, angle = local_geometry_errors(endpoint, target, numbers)
            pair_losses.append(pair)
            angle_losses.append(angle)
    coordinate = torch.stack(endpoint_losses).mean()
    pair = torch.stack(pair_losses).mean() if pair_losses else coordinate.new_zeros(())
    angle = torch.stack(angle_losses).mean() if angle_losses else coordinate.new_zeros(())
    objective = 4 * coordinate + .5 * pair + .1 * angle
    return objective, dict(coordinate=float(coordinate.detach()), pair=float(pair.detach()),
                           angle=float(angle.detach()), corruption_rms=float(noise.square().mean().sqrt()))
