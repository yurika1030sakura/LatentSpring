"""Small invariant energy critic for a frozen conditional proposal score.

This is a standard invariant message-passing energy architecture, not a new
molecular generator. It is fitted without target energy labels. Its score must
be independently checked before use in an actor update.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ProposalEnergyCritic(nn.Module):
    def __init__(self, hidden=32, radial=24):
        super().__init__()
        self.elements = nn.Embedding(119, hidden)
        self.state = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.register_buffer('centers', torch.linspace(0, 8, radial))
        self.messages = nn.ModuleList([nn.Sequential(nn.Linear(2*hidden+radial, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU()) for _ in range(2)])
        self.updates = nn.ModuleList([nn.Sequential(nn.Linear(2*hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden)) for _ in range(2)])
        self.readout = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        self.log_confinement = nn.Parameter(torch.tensor(-2.3))

    def forward(self, x, numbers, electronic):
        batch, atoms, _ = x.shape
        if not 2 <= atoms <= 200 or numbers.shape != (atoms,) or electronic.shape != (3,):
            raise ValueError('Invalid fixed molecular condition')
        x = x-x.mean(1, keepdim=True)
        distance = ((x[:, :, None]-x[:, None, :]).square().sum(-1)+1e-8).sqrt()
        radial = torch.exp(-.5*((distance[..., None]-self.centers)/.4)**2)
        nodes = (self.elements(numbers)+self.state(electronic)[None]).expand(batch, atoms, -1)
        mask = (~torch.eye(atoms, dtype=torch.bool, device=x.device))[None, :, :, None]
        for message, update in zip(self.messages, self.updates):
            left = nodes[:, :, None].expand(-1, -1, atoms, -1)
            right = nodes[:, None, :].expand(-1, atoms, -1, -1)
            pair = message(torch.cat([left+right, (left-right).square(), radial], -1))
            pooled = (pair*mask).sum(2)/atoms
            nodes = nodes+update(torch.cat([nodes, pooled], -1))
        # Bounded radial features and a positive quadratic tail make exp(-E)
        # integrable on the COM-free subspace for each finite parameter state.
        return self.readout(nodes).sum((1, 2))+.5*(F.softplus(self.log_confinement)+1e-4)*x.square().sum((1, 2))

    def score(self, z, basis, numbers, electronic, *, create_graph=False):
        with torch.enable_grad():
            z = z.detach().requires_grad_(True)
            x = torch.einsum('nk,bkd->bnd', basis, z.reshape(len(z), -1, 3))
            energy = self(x, numbers, electronic)
            return -torch.autograd.grad(energy.sum(), z, create_graph=create_graph)[0]


def invariant_stein_rows(x, score_cartesian):
    """Per-sample necessary score checks: dilation and three radial gradient probes.

    These finite moment checks cannot certify a complete score or global density.
    The radial Hessian trace is on COM-free H; pair gradients have zero COM.
    """
    atoms = x.shape[1]
    pairs = torch.triu_indices(atoms, atoms, 1, device=x.device)
    delta = x[:, pairs[0]]-x[:, pairs[1]]
    squared = delta.square().sum(-1)
    radius = (squared+1e-8).sqrt()
    score_delta = score_cartesian[:, pairs[0]]-score_cartesian[:, pairs[1]]
    result = {'dilation': ((x*score_cartesian).sum((1, 2))+3*(atoms-1))/(3*(atoms-1))}
    for center in [1., 2., 3.]:
        width = .5
        rbf = torch.exp(-.5*((radius-center)/width)**2)
        first = -(radius-center)/width**2*rbf
        second = (((radius-center)/width**2)**2-1/width**2)*rbf
        contraction = (score_delta*delta).sum(-1)*first/radius
        laplacian = 2*(second*squared/radius**2+first*(3/radius-squared/radius**3))
        result[f'radial_{center:g}'] = (contraction+laplacian).sum(-1)/len(pairs[0])
    return result
