"""MH correction of an invertible learned map using a random inverse direction.

The physical source likelihood is unnecessary. Auxiliary coordinates have a
known conditional law and are refreshed before each direction-augmented move.
"""
import math
import torch


def conditional_auxiliary_log_prob(auxiliary, physical, scale):
    if (auxiliary.shape != physical.shape or physical.ndim != 3 or physical.shape[-1] != 3
            or not math.isfinite(scale) or scale <= 0):
        raise ValueError('Matched physical/auxiliary batches and positive scale required')
    # These auxiliary coordinates are NOT projected onto the COM subspace.
    dimension = physical.shape[1]*3
    return -.5*((auxiliary-physical)/scale).square().sum((1, 2))-dimension*math.log(scale*math.sqrt(2*math.pi))


@torch.no_grad()
def directional_flow_transition(target, states, transport, *, generator, phase):
    x = torch.stack([s['positions'] for s in states])
    noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
    auxiliary = x+transport.aux_scale*noise
    directions = torch.randint(2, (len(x),), generator=generator)  #0 forward,1 inverse
    mapped = transport.transform(x, auxiliary, directions)
    y, b, volume = mapped['positions'], mapped['auxiliary'], mapped['log_volume']
    if y.shape != x.shape or b.shape != x.shape or volume.shape != (len(x),):
        raise ValueError('Invalid learned-map output shapes')
    if not all(torch.isfinite(v).all() for v in [y, b, volume]):
        raise FloatingPointError('Nonfinite learned-map output; no silent rejection')
    forward = conditional_auxiliary_log_prob(auxiliary, x, transport.aux_scale)
    reverse = conditional_auxiliary_log_prob(b, y, transport.aux_scale)
    candidates, rows = [], []
    for index, old in enumerate(states):
        row = dict(kind='directional_flow', phase=phase, old_state_id=old['state_id'], new_state_id=-1,
            direction=int(directions[index]), reverse_direction=1-int(directions[index]),
            auxiliary_noise=noise[index], source_auxiliary=auxiliary[index], proposal_auxiliary=b[index],
            proposal_positions=y[index], log_volume=volume[index],
            forward_auxiliary_log_prob=forward[index], reverse_auxiliary_log_prob=reverse[index],
            auxiliary_scale=transport.aux_scale, valid=False, accepted=False)
        candidate = None
        try:
            candidate = target.coordinate_state(y[index])
            row['valid'] = True
        except ValueError as exc:
            row['rejection_reason'] = str(exc)
        candidates.append(candidate)
        rows.append(row)
    target.evaluate([s for s in candidates if s is not None], phase=phase)
    log_u = torch.rand(len(states), dtype=x.dtype, generator=generator).log()
    updated = list(states)
    for index, (old, new, row) in enumerate(zip(states, candidates, rows)):
        row['log_uniform'] = float(log_u[index])
        if new is None:
            continue
        ratio = -float(new['potential_eV']-old['potential_eV'])/target.kT+float(reverse[index]-forward[index]+volume[index])
        take = float(log_u[index]) < min(0., ratio)
        row.update(new_state_id=new['state_id'], log_acceptance_ratio=ratio, accepted=take)
        if take:
            updated[index] = new
    return updated, rows
