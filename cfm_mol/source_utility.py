"""Exact source-space adaptation with unchanged tree edge and decoder kernels.

The importance identity and KL data processing are standard results. These
quantities are not final-output density ratios or Boltzmann certificates.
"""
import math
import torch
from cfm_mol.tree_mixture_prior import radial_log_density, sample_tree_coordinates


def batched_log_partition(log_weights):
    """Positive star-mesh elimination, with arbitrary leading batch axes."""
    n = log_weights.shape[-1]
    if log_weights.shape[-2] != n:
        raise ValueError('Square edge matrices required')
    w = log_weights.masked_fill(torch.eye(n, device=log_weights.device, dtype=torch.bool), -torch.inf)
    total = w.new_zeros(w.shape[:-2])
    for remaining in range(n, 1, -1):
        edge = w[..., 0, 1:]
        degree = torch.logsumexp(edge, -1)
        total = total + degree
        w = torch.logaddexp(w[..., 1:, 1:], edge[..., :, None] + edge[..., None, :] - degree[..., None, None])
        w = w.masked_fill(torch.eye(remaining-1, device=w.device, dtype=torch.bool), -torch.inf)
    return total


def batched_log_prob(x, loga, lengths, width):
    """Exact coordinate marginal, for a batch of centered source coordinates."""
    n = len(loga)
    if x.ndim != 3 or x.shape[1:] != (n, 3) or not torch.isfinite(x).all():
        raise ValueError('Finite (batch, atoms, 3) coordinates required')
    if float(x.mean(1).abs().max()) > 1e-5:
        raise ValueError('Source coordinates must be centered')
    if n == 1:
        return x.new_zeros(len(x))
    i, j = torch.triu_indices(n, n, 1, device=x.device)
    distances = (x[:, i]-x[:, j]).norm(dim=-1)
    h = x.new_full((len(x), n, n), -torch.inf)
    values = radial_log_density(distances, lengths[i, j], width)
    h[:, i, j] = values
    h[:, j, i] = values
    return 1.5*math.log(n) + batched_log_partition(loga+h) - batched_log_partition(loga)


def tree_kl(loga, loga0):
    """KL(tree(a)||tree(a0)); effective-resistance edge marginals.

    The grounded inverse is well conditioned for this pilot's bounded affinity
    heads. Do not use this routine for arbitrary nearly disconnected graphs.
    Autograd includes the derivative of the edge marginals.
    """
    n = len(loga)
    if loga.shape != loga0.shape or loga.shape != (n, n):
        raise ValueError('Matching square log affinity matrices required')
    if not torch.allclose(loga, loga.T) or not torch.allclose(loga0, loga0.T):
        raise ValueError('Symmetric affinities required')
    if n < 2:
        return loga.sum()*0
    off = ~torch.eye(n, device=loga.device, dtype=torch.bool)
    weights = torch.exp(loga-loga[off].max().detach()).masked_fill(~off, 0.)
    laplacian = torch.diag(weights.sum(-1))-weights
    inverse = torch.linalg.inv(laplacian[:-1, :-1])
    grounded = torch.nn.functional.pad(inverse, (0, 1, 0, 1))
    diagonal = grounded.diagonal()
    marginals = weights*(diagonal[:, None]+diagonal[None, :]-2*grounded)
    value = .5*(marginals*(loga-loga0)).sum()-batched_log_partition(loga)+batched_log_partition(loga0)
    if not torch.isfinite(value) or float(value.detach()) < -1e-8:
        raise FloatingPointError('Invalid tree KL')
    return value.clamp_min(0.)


class TrustMixturePrior:
    """Condition-wise mixture with epsilon*KL_tree <= delta.

    Both components have identical normalized edge kernels; the caller must
    keep the generation kernel fixed to invoke the output KL bound.
    """
    def __init__(self, candidate, base, delta=.25):
        if candidate.width != base.width or base.mode != 'fixed' or not delta > 0:
            raise ValueError('Fixed base, identical edge widths and positive budget required')
        if not torch.equal(candidate.radii, base.radii) or not torch.equal(candidate.base_log_propensity, base.base_log_propensity):
            raise ValueError('Source kernels and base propensities must agree')
        self.candidate, self.base, self.delta = candidate, base, delta

    def components(self, numbers, charge, spin):
        a, length = self.candidate.parameters_for(numbers, charge, spin)
        a0, length0 = self.base.parameters_for(numbers, charge, spin)
        if not torch.equal(length, length0):
            raise ValueError('Tree edge kernels changed')
        kl = tree_kl(a, a0)
        epsilon = (self.delta/kl.clamp_min(1e-12)).clamp_max(1.)
        return a, a0, length, kl, epsilon

    def log_prob_batch(self, x, numbers, charge, spin):
        a, a0, length, kl, epsilon = self.components(numbers, charge, spin)
        log0 = batched_log_prob(x.to(a0), a0, length, self.base.width)
        log1 = batched_log_prob(x.to(a), a, length, self.base.width)
        ratio = (1-epsilon)+epsilon*torch.exp(log1-log0)
        return log0+ratio.log(), dict(tree_kl=kl, epsilon=epsilon, output_kl_bound=epsilon*kl)

    def log_prob(self, x, numbers, charge, spin):
        return self.log_prob_batch(x[None], numbers, charge, spin)[0][0]

    @torch.no_grad()
    def sample(self, numbers, charge, spin, *, rng, generator):
        a, a0, length, _, epsilon = self.components(numbers, charge, spin)
        chosen = a if rng.random() < float(epsilon) else a0
        return sample_tree_coordinates(chosen, length, rng=rng, generator=generator, width=self.base.width)


def importance_utility(logq, logq0, rewards):
    """Unnormalized IS with leave-one-out reward centering.

    For a fixed model and iid bank, this estimates J(eta)-J(base) without the
    same-sample baseline bias. It is not unbiased after optimizing on the bank.
    """
    if logq.shape != logq0.shape or rewards.shape != logq.shape or len(rewards) < 2:
        raise ValueError('At least two paired scalar densities and rewards required')
    weights = (logq-logq0).exp()
    gain = (weights*(rewards-rewards.mean())).mean()*len(rewards)/(len(rewards)-1)
    diagnostics = dict(weight_mean=weights.mean(), source_bank_ess=weights.sum().square()/weights.square().sum(),
                       max_weight=weights.max(), uncentered_utility=(weights*rewards).mean())
    return gain, diagnostics
