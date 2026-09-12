"""Bounded conditional exchange selection and exact observed placement densities.

The move-family schedule is external and fixed. The selector returns probabilities
only over supplied eligible exchanges; it does not change local-move frequency.
"""
import math
import torch
from torch import nn

from cfm_mol.chemical_policy import ChemicalMovePolicy
from cfm_mol.chemical_moves import covalent_radii
from cfm_mol.joint_chemical_geometry import distinct_anchor_actions
from cfm_mol.joint_arc_geometry import observed_joint_arc_density
from cfm_mol.bounded_arc_guide import BoundedArcGuide
from cfm_mol.accepted_utility import accepted_importance_utility


class BoundedConditionalActionPolicy(ChemicalMovePolicy):
    """Symmetric conditional action logits, with no move-family head."""
    def __init__(self, hidden=16, radial=16, logit_bound=1.):
        if hidden < 1 or radial < 2 or not 0 < logit_bound < float('inf'):
            raise ValueError('Positive dimensions and finite action-logit bound required')
        super().__init__(hidden=hidden, radial=radial)
        del self.family_head
        z = torch.arange(119, dtype=torch.float64)
        self.register_buffer('atomic_features', torch.stack([
            z/118, torch.log1p(z)/math.log(119), covalent_radii(range(119))/2], 1))
        self.elements = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.logit_bound = logit_bound
        self.configuration = dict(hidden=hidden, radial=radial, logit_bound=logit_bound)

    def element_nodes(self, numbers):
        if ((numbers < 1) | (numbers > 118)).any():
            raise ValueError('Atomic numbers must lie in 1..118')
        return self.elements(self.atomic_features[numbers].to(self.elements[0].weight.dtype))

    def forward(self, x, bonds, numbers, electronic, actions, mask=None):
        _, raw, mask = self._action_logits(x, bonds, numbers, electronic, actions, mask)
        if raw.shape[1] == 0:
            return raw
        logits = self.logit_bound*torch.tanh(raw/self.logit_bound)
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite conditional action logits')
        available = mask.any(1)
        logits = logits.masked_fill(~mask, -torch.inf)
        logits = torch.where(available[:, None], logits, torch.zeros_like(logits))
        # Empty rows mean no exchange; the caller performs a self transition.
        return logits.log_softmax(1).masked_fill(~mask, -torch.inf)

    def legal_log_probabilities(self, x, bonds, numbers, electronic):
        actions = distinct_anchor_actions(numbers, bonds)
        tensor = torch.tensor(actions, dtype=torch.long, device=x.device).reshape(1, -1, 4)
        return actions, self(x[None], bonds[None], numbers, electronic, tensor)[0]


class ActionGeometryGuide(nn.Module):
    """Matched exp(2) proposal-ratio ablations from exact physical initialization."""
    def __init__(self, variant, geometry=None, action=None):
        super().__init__()
        if variant not in ('action', 'geometry', 'joint'):
            raise ValueError('Expected action, geometry or joint variant')
        self.variant = variant
        geometry = dict(geometry or {})
        action = dict(action or {})
        geometry.setdefault('log_score_bound', .25 if variant == 'joint' else .5)
        action.setdefault('logit_bound', .5 if variant == 'joint' else 1.)
        # Construct geometry first so geometry-only repeats the prior initialization.
        self.geometry = BoundedArcGuide(**geometry)
        self.action = BoundedConditionalActionPolicy(**action)
        self.geometry.requires_grad_(variant != 'action')
        self.action.requires_grad_(variant != 'geometry')
        self.configuration = dict(variant=variant, geometry=self.geometry.configuration,
                                  action=self.action.configuration)

    @property
    def log_ratio_bound(self):
        return ((0 if self.variant == 'geometry' else 2*self.action.logit_bound)
                + (0 if self.variant == 'action' else 4*self.geometry.log_score_bound))

    def selected_action_log_prob(self, x, bonds, numbers, electronic, selected):
        if self.variant == 'geometry':
            actions = distinct_anchor_actions(numbers, bonds)
            probabilities = x.new_full((len(actions),), -math.log(max(1, len(actions))))
        else:
            actions, probabilities = self.action.legal_log_probabilities(x, bonds, numbers, electronic)
        if tuple(selected) not in actions:
            raise ValueError('Recorded forward/inverse action is not eligible')
        return probabilities[actions.index(tuple(selected))], len(actions)


def full_observed_densities(model, row, options):
    """Complete augmented action/coordinate densities at a fixed recorded pair."""
    if not row['valid']:
        raise ValueError('A supported scored edge is required')
    x, y, z, e = row['x'], row['y'], row['numbers'], row['electronic']
    af, nf = model.selected_action_log_prob(x, row['bonds'], z, e, row['action'])
    ar, nr = model.selected_action_log_prob(y, row['new_bonds'], z, e, row['inverse_action'])
    if abs(math.log(nf/nr)-float(row['action_log_ratio'])) > 1e-10:
        raise ValueError('Recorded behavioral action count ratio does not match legal sets')
    if model.variant == 'action':
        qf, qr = row['log_behavior_forward'], row['log_behavior_reverse']
    else:
        qf, _ = observed_joint_arc_density(x, y, row['bonds'], z, e, row['radii'], row['action'],
            kind='arc_bounded', model=model.geometry, **options)
        qr, _ = observed_joint_arc_density(y, x, row['new_bonds'], z, e, row['radii'], row['inverse_action'],
            kind='arc_bounded', model=model.geometry, **options)
    if not torch.isfinite(qf) or not torch.isfinite(qr):
        raise ValueError('Recorded supported coordinate edge changed support')
    return qf+af, qr+ar, row['log_behavior_forward']-math.log(nf), row['log_behavior_reverse']-math.log(nr)


def edge_values(model, row, options):
    zero = sum(p.sum()*0 for p in model.parameters() if p.requires_grad)
    if not row['valid']:
        return zero, zero, zero, zero, zero, zero
    qf, qr, bf, br = full_observed_densities(model, row, options)
    lf, lr = qf-bf, qr-br
    if max(abs(float(lf)), abs(float(lr))) > model.log_ratio_bound+1e-7:
        raise ValueError('Complete action/geometry likelihood-ratio bound violated')
    utility, flow = accepted_importance_utility(qf, qr, bf, row['target_log_ratio'],
                                               torch.zeros_like(qf), row['reward_eV'])
    importance = lf.exp()
    return (utility, 2*importance, flow*int(row['connectivity_changed']), importance,
            (lf.square()+lr.square())/2, lf)
