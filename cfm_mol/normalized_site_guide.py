"""Masked, normalized equivariant directional mixtures about a physical site prior.

Components are indexed by passive atoms, with paired vector offsets; no arbitrary
eigenvector or learned atom-index ordering is selected. Atomic inputs are fixed
continuous descriptors, rather than untrained element-specific embedding rows.
"""
import math
import torch
from torch import nn
from cfm_mol.masked_angular_guide import masked_angular_context
from cfm_mol.chemical_moves import covalent_radii
from cfm_mol.spherical_proposal import vmf_log_normalizer, vmf_sample


def mixture_log_prob(direction, parameters, log_weights):
    b, k, _ = parameters.shape
    if direction.shape != (b, 3) or log_weights.shape != (b, k):
        raise ValueError('Incompatible normalized mixture inputs')
    normalizer = vmf_log_normalizer(parameters.reshape(-1, 3)).reshape(b, k)
    return torch.logsumexp(log_weights+normalizer+torch.einsum('bki,bi->bk', parameters, direction), 1)


def mixture_surface_score(direction, parameters, log_weights):
    b, k, _ = parameters.shape
    normalizer = vmf_log_normalizer(parameters.reshape(-1, 3)).reshape(b, k)
    posterior = (log_weights+normalizer+torch.einsum('bki,bi->bk', parameters, direction)).softmax(1)
    raw = (posterior[:, :, None]*parameters).sum(1)
    return raw-(raw*direction).sum(1, keepdim=True)*direction


@torch.no_grad()
def mixture_draw(parameters, log_weights, *, generator):
    component = torch.multinomial(log_weights.exp(), 1, generator=generator)[:, 0]
    selected = parameters[torch.arange(len(parameters), device=parameters.device), component]
    direction, random = vmf_sample(selected, generator=generator)
    return direction, dict(component=component, random=random)


def physical_site_parameter(masked, roles, bonds, roots, concentration):
    batch = torch.arange(len(masked), device=masked.device)
    neighbors = (bonds[batch, roots[:, 1]] > 0) & (roles != 1)
    unit = masked/masked.norm(dim=2, keepdim=True).clamp_min(1e-12)
    away = -(neighbors[:, :, None]*unit).sum(1)
    return concentration*away/away.norm(dim=1, keepdim=True).clamp_min(1e-12)


class NormalizedSiteGuide(nn.Module):
    def __init__(self, hidden=16, radial=16, bound=64., mixture=True, site_concentration=10.):
        super().__init__()
        if hidden < 1 or radial < 2 or bound <= 0 or site_concentration < 0:
            raise ValueError('Invalid normalized guide configuration')
        self.configuration = dict(hidden=hidden, radial=radial, bound=bound, mixture=mixture,
                                  site_concentration=site_concentration)
        self.bound, self.use_mixture, self.site_concentration = bound, mixture, site_concentration
        z = torch.arange(119, dtype=torch.float64)
        descriptor = torch.stack([z/118, torch.log1p(z)/math.log(119), covalent_radii(range(119))/2], 1)
        self.register_buffer('atomic_features', descriptor)
        self.elements = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.roles = nn.Embedding(3, hidden)
        self.state = nn.Sequential(nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.register_buffer('radial_centers', torch.linspace(0, 8, radial))
        self.messages = nn.ModuleList([nn.Sequential(nn.Linear(2*hidden+radial+1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU()) for _ in range(2)])
        self.updates = nn.ModuleList([nn.Sequential(nn.Linear(2*hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden)) for _ in range(2)])
        self.center_weight = nn.Linear(hidden, 1)
        self.component_weight = nn.Linear(hidden, 1)
        self.offset_weight = nn.Linear(hidden, 1)
        for head in [self.center_weight, self.component_weight, self.offset_weight]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x, bonds, numbers, electronic, roots):
        if (not 2 <= x.shape[1] <= 200 or bonds.shape != x.shape[:2]+(x.shape[1],)
                or numbers.shape != (x.shape[1],) or numbers.dtype != torch.long
                or ((numbers < 1) | (numbers > 118)).any()):
            raise ValueError('Invalid molecular guide batch')
        masked, radius, roles = masked_angular_context(x, roots)
        if electronic.shape == (3,):
            electronic = electronic.expand(len(x), -1)
        if electronic.shape != (len(x), 3) or not torch.isfinite(electronic).all() or not torch.isfinite(bonds).all():
            raise ValueError('Finite graph/electronic condition required')
        n = x.shape[1]
        distance = ((masked[:, :, None]-masked[:, None, :]).square().sum(-1)+1e-8).sqrt()
        radial = torch.exp(-.5*((distance[..., None]-self.radial_centers)/.4)**2)
        nodes = self.elements(self.atomic_features[numbers].to(x.dtype))[None]+self.roles(roles)
        nodes = nodes+self.state(torch.cat([electronic, radius[:, None]], 1))[:, None]
        pair_mask = (~torch.eye(n, dtype=torch.bool, device=x.device))[None, :, :, None]
        for message, update in zip(self.messages, self.updates):
            a = nodes[:, :, None].expand(-1, -1, n, -1)
            b = nodes[:, None, :].expand(-1, n, -1, -1)
            pair = message(torch.cat([a+b, (a-b).square(), radial, bonds[..., None]/3], -1))
            nodes = nodes+update(torch.cat([nodes, (pair*pair_mask).sum(2)/n], -1))
        vectors = masked/(masked.square().sum(-1, keepdim=True)+.01).sqrt()
        residual = (self.center_weight(nodes)*vectors).sum(1)/math.sqrt(n)
        residual = residual*self.bound/(self.bound+residual.norm(dim=1, keepdim=True))
        site = physical_site_parameter(masked, roles, bonds, roots, self.site_concentration)
        center = site+residual
        if not self.use_mixture:
            return center[:, None], x.new_zeros(len(x), 1)
        # Nonzero smooth offsets at initialization avoid stationary zero-width
        # symmetric components. The untrained density is recorded as a control.
        width = self.bound*torch.tanh(torch.nn.functional.softplus(self.offset_weight(nodes))/self.bound)
        offset = width*vectors
        parameters = torch.stack([center[:, None]+offset, center[:, None]-offset], 2).flatten(1, 2)
        logits = self.component_weight(nodes)[:, :, 0].masked_fill(roles == 1, -torch.inf)
        weights = logits.log_softmax(1)[:, :, None].expand(-1, -1, 2).flatten(1, 2)-math.log(2)
        return parameters, weights
