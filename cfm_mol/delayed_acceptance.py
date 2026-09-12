"""Antisymmetric cheap screens with ordinary two-stage Metropolis correction.

This is established delayed acceptance, not a new balance theorem. Fixed-step
invariance does not certify the distribution at a query-budget stopping time.
"""
import math
import torch
from torch import nn

from cfm_mol.bounded_action_geometry import BoundedConditionalActionPolicy
from cfm_mol.chemical_moves import covalent_radii


def delayed_log_acceptance(log_ratio, log_factor):
    """Return first, corrective second and total log acceptance probabilities."""
    if log_ratio.shape != log_factor.shape or not torch.isfinite(log_factor).all():
        raise ValueError('Matched shapes and finite cheap factors required')
    if torch.isnan(log_ratio).any() or torch.isposinf(log_ratio).any():
        raise ValueError('MH log ratio must be finite or minus infinity')
    zero = torch.zeros_like(log_factor)
    first = torch.minimum(zero, log_factor)
    second = torch.minimum(zero, log_ratio-log_factor)
    return first, second, first+second


def cheap_state_features(x, bonds, numbers):
    """Bond strain, nonbonded repulsion and COM radius; no oracle inputs.

    The first two quantities are heuristic surrogate features, not physical
    energies. Fixed coefficients used below define a declared cheap control.
    """
    if (x.ndim != 2 or x.shape[-1] != 3 or not 2 <= len(x) <= 200 or bonds.shape != (len(x),len(x))
            or numbers.shape != (len(x),) or numbers.dtype != torch.long
            or ((numbers < 1) | (numbers > 118)).any()
            or not all(torch.isfinite(v).all() for v in (x,bonds))):
        raise ValueError('Finite single-molecule coordinates and graph required')
    radii = covalent_radii(numbers).to(x)
    i,j = torch.triu_indices(len(x),len(x),1,device=x.device)
    distance = (x[i]-x[j]).norm(dim=-1)
    if (distance <= 0).any():
        raise ValueError('Distinct atom coordinates required')
    scaled = distance/(radii[i]+radii[j])
    bonded = bonds[i,j] > 0
    strain = ((scaled[bonded]-1).square()*bonds[i,j][bonded]).sum()
    repulsion = scaled[~bonded].pow(-12).sum()
    radius = (x-x.mean(0)).square().sum()
    return torch.stack([strain,repulsion,radius])


def paired_screen_features(x, y, bonds, new_bonds, numbers, electronic,
                           proposal_log_ratio, restraint=.1):
    """Odd forward features: three cheap energy decreases and exact log q_r/q_f."""
    if electronic.shape != (3,) or not torch.isfinite(electronic).all() or electronic[2] <= 0:
        raise ValueError('Original charge/spin and positive temperature required')
    if restraint < 0 or not math.isfinite(restraint):
        raise ValueError('Nonnegative finite restraint required')
    delta = cheap_state_features(x,bonds,numbers)-cheap_state_features(y,new_bonds,numbers)
    energy_scales = x.new_tensor([10.,1.,restraint/2])
    ratio = torch.as_tensor(proposal_log_ratio,dtype=x.dtype,device=x.device)
    if ratio.ndim != 0 or not torch.isfinite(ratio):
        raise ValueError('Finite complete proposal log ratio required')
    return torch.cat([delta*energy_scales/electronic[2],ratio[None]])


class DelayedAcceptanceScreen(nn.Module):
    """Zero, fixed physical, linear, or neural antisymmetric bounded log factor.

    Learned screens start at zero, recovering ordinary MH. Both orientations use
    the same network; no action softmax or family probability is learned here.
    """
    def __init__(self, variant, log_factor_bound=math.log(16), restraint=.1,
                 hidden=16, radial=16):
        super().__init__()
        if variant not in ('zero','physical','linear','neural'):
            raise ValueError('Unknown delayed-acceptance screen')
        if not 0 < log_factor_bound < float('inf') or not 0 <= restraint < float('inf'):
            raise ValueError('Finite bound and restraint required')
        self.variant,self.log_factor_bound,self.restraint=variant,log_factor_bound,restraint
        self.coefficients=nn.Parameter(torch.ones(4) if variant=='physical' else torch.zeros(4),
                                       requires_grad=variant in ('linear','neural'))
        self.encoder=BoundedConditionalActionPolicy(hidden=hidden,radial=radial) if variant=='neural' else None
        self.configuration=dict(variant=variant,log_factor_bound=log_factor_bound,restraint=restraint,hidden=hidden,radial=radial)

    def orientation_score(self,x,bonds,numbers,electronic,action):
        actions=torch.tensor([[action]],dtype=torch.long,device=x.device)
        # Use the symmetric UNNORMALIZED readout. A conditional action softmax
        # would introduce an unrelated legal-set normalization into this screen.
        return self.encoder._action_logits(x[None],bonds[None],numbers,electronic,actions)[1][0,0]

    def log_factor(self,x,y,bonds,new_bonds,numbers,electronic,action,proposal_log_ratio):
        features=paired_screen_features(x,y,bonds,new_bonds,numbers,electronic,
                                        proposal_log_ratio,self.restraint)
        raw=(self.coefficients*features).sum()
        if self.encoder is not None:
            i,j,k,l=action;inverse=(i,j,l,k)
            raw=raw+self.orientation_score(x,bonds,numbers,electronic,action)-self.orientation_score(y,new_bonds,numbers,electronic,inverse)
        value=self.log_factor_bound*torch.tanh(raw/self.log_factor_bound)
        if not torch.isfinite(value):
            raise ValueError('Nonfinite cheap screen')
        return value


def screen_edge_values(model,row):
    """Empirical signed accepted work and raw query cost on fixed physical pairs."""
    zero=sum(p.sum()*0 for p in model.parameters())
    if not row['valid']:
        return zero,zero,zero,zero,zero
    proposal_ratio=row['log_behavior_reverse']-row['log_behavior_forward']+row['action_log_ratio']
    factor=model.log_factor(row['x'],row['y'],row['bonds'],row['new_bonds'],row['numbers'],
                            row['electronic'],row['action'],proposal_ratio)
    ratio=row['target_log_ratio']+proposal_ratio
    first,second,total=delayed_log_acceptance(ratio,factor)
    acceptance=total.exp()
    return (row['reward_eV']*acceptance,2*first.exp(),
            acceptance*int(row['connectivity_changed']),factor,acceptance)
