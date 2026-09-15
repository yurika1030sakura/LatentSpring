"""Shared EGNN for an unpretrained, position-conditional generator comparison.

EDM uses its official variance-preserving schedule and epsilon prediction. The
GAGA variant truncates training and sampling at a declared timestep, with the
Gaussian variance propagated from training coordinates. Both FM arms use the
same velocity architecture, coupling, optimizer and numerical solver.
"""
import hashlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from cfm_mol.orbit_pairing import typed_orbit_pair
from cfm_mol.chemical_moves import covalent_radii
from cfm_mol.tree_prior_controls import cayley_tree, edge_embedding
from cfm_mol.tree_mixture_prior import TreeMixturePrior


def center(x):
    return x-x.mean(-2, keepdim=True)


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initialize(spec, device):
    upstream = Path(spec['upstream'])
    for name, digest in spec['upstream_sha256'].items():
        assert hashlib.sha256((upstream/name).read_bytes()).hexdigest() == digest
    sys.path.insert(0, str(upstream))
    from qm9.models import get_model
    from configs.datasets_config import get_dataset_info
    args = SimpleNamespace(**spec['upstream_args'])
    assert not args.conditioning and not args.include_charges
    info = get_dataset_info(args.dataset, args.remove_h)
    assert info['atomic_nb'] == spec['atomic_numbers']
    torch.manual_seed(spec['initialization_seed'])
    model, _, _ = get_model(args, device, info, None)
    model.to(device)

    def finite_before_fallback(module, inputs, output):
        if any(not torch.isfinite(value).all() for value in output):
            raise FloatingPointError('Nonfinite EGNN output before upstream fallback')

    model.dynamics.egnn.register_forward_hook(finite_before_fallback)
    return model


def vector(model, x, t, numbers, spec):
    """All graphs in a minibatch have the same atom count, without padding."""
    batch, n, _ = x.shape
    assert numbers.shape == (batch, n)
    lookup = torch.full((84,), -1, dtype=torch.long, device=x.device)
    lookup[torch.tensor(spec['atomic_numbers'], device=x.device)] = torch.arange(len(spec['atomic_numbers']), device=x.device)
    indices = lookup[numbers]
    if (indices < 0).any():
        raise ValueError('Unsupported species')
    h = torch.nn.functional.one_hot(indices, len(spec['atomic_numbers'])).to(x)/model.norm_values[1]
    node_mask = x.new_ones(batch, n, 1)
    edge_mask = (~torch.eye(n, dtype=torch.bool, device=x.device))[None].expand(batch, -1, -1).reshape(-1, 1).to(x)
    result = model.phi(torch.cat([x, h], -1), t, node_mask, edge_mask, None)[..., :3]
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite model prediction')
    return center(result)


class HarmonicSource:
    """Exact factorized tree law via Pruefer sampling, with cached atomic scales."""
    def __init__(self, width=.2):
        self.base = TreeMixturePrior('fixed', width=width).double()
        self.width = width
        self.cache = {}

    def sample(self, numbers, rng):
        key = tuple(numbers)
        if key not in self.cache:
            _, length = self.base.parameters_for(key, 0, 1)
            u = self.base.base_log_propensity[list(key)].exp().detach().numpy()
            self.cache[key] = (u, length.detach().numpy()*math.exp(.5*self.width**2)/math.sqrt(3))
        u, scales = self.cache[key]
        edges = cayley_tree(u, rng)
        std = np.asarray([scales[i, j] for i, j in edges])
        x = edge_embedding(len(key), edges) @ (std[:, None]*rng.normal(size=(len(key)-1, 3)))
        return torch.from_numpy(x)


def fm_endpoints(clean, numbers, kind, source, seed):
    """Proper rotation, within-species assignment, and shared Haar augmentation."""
    rng = np.random.default_rng(seed)
    starts, targets = [], []
    for i, (x, z) in enumerate(zip(clean.detach().cpu().double(), numbers.cpu())):
        zz = z.tolist()
        x0 = source.sample(zz, rng) if kind == 'harmonic_fm' else center(torch.from_numpy(rng.normal(size=x.shape)))
        x0, x1, _ = typed_orbit_pair(x0, center(x), covalent_radii(zz, dtype=torch.float64), z,
            generator=torch.Generator().manual_seed(seed+i+500000003))
        starts.append(x0)
        targets.append(x1)
    return torch.stack(starts).to(clean), torch.stack(targets).to(clean)


def loss(model, clean, numbers, kind, spec, source, seed):
    rng = torch.Generator(device=clean.device).manual_seed(seed)
    batch = len(clean)
    clean = center(clean)/model.norm_values[0]
    if kind in ['gaussian_fm', 'harmonic_fm']:
        x0, target = fm_endpoints(clean, numbers, kind, source, seed)
        t = torch.rand((batch, 1), device=clean.device, generator=rng)
        xt = (1-t[..., None])*x0+t[..., None]*target
        prediction = vector(model, xt, t, numbers, spec)
        return (prediction-(target-x0)).square().mean()
    if kind not in ['edm', 'gaga']:
        raise ValueError(kind)
    maximum = spec['gaga_max_t'] if kind == 'gaga' else model.T
    t = torch.randint(0, maximum+1, (batch, 1), device=clean.device, generator=rng).to(clean)/model.T
    epsilon = center(torch.randn(clean.shape, device=clean.device, generator=rng))
    gamma = model.gamma(t)
    xt = model.alpha(gamma, clean)*clean+model.sigma(gamma, clean)*epsilon
    return (vector(model, xt, t, numbers, spec)-epsilon).square().mean()


@torch.no_grad()
def sample(model, numbers, kind, spec, source, seed, batch, calls=128):
    device = next(model.parameters()).device
    z = torch.tensor(numbers, dtype=torch.long, device=device)[None].expand(batch, -1)
    shape = (batch, len(numbers), 3)
    rng = torch.Generator(device=device).manual_seed(seed)
    if kind == 'harmonic_fm':
        nrng = np.random.default_rng(seed)
        x = torch.stack([source.sample(numbers, nrng) for _ in range(batch)]).to(device).float()
    else:
        x = center(torch.randn(shape, generator=rng, device=device))
    if kind in ['gaussian_fm', 'harmonic_fm']:
        assert calls % 2 == 0
        initial = x.clone()
        steps = calls//2
        for i in range(steps):
            t = x.new_full((batch, 1), i/steps)
            first = vector(model, x, t, z, spec)
            middle = center(x+first/(2*steps))
            x = center(x+vector(model, middle, t+.5/steps, z, spec)/steps)
        return x*model.norm_values[0], initial
    if kind not in ['edm', 'gaga']:
        raise ValueError(kind)
    maximum = spec['gaga_max_t'] if kind == 'gaga' else model.T
    if kind == 'gaga':
        t = x.new_full((batch, 1), maximum/model.T)
        gamma = model.gamma(t)
        variance = model.alpha(gamma, x).square()*spec['data_variance_per_dof']+model.sigma(gamma, x).square()
        x *= variance.sqrt()
    initial = x.clone()
    ticks = np.rint(np.linspace(0, maximum, calls)).astype(int)
    assert len(np.unique(ticks)) == calls
    for low, high in reversed(list(zip(ticks[:-1], ticks[1:]))):
        s, t = [x.new_full((batch, 1), int(k)/model.T) for k in [low, high]]
        gs, gt = model.gamma(s), model.gamma(t)
        variance, std, ratio = model.sigma_and_alpha_t_given_s(gt, gs, x)
        ss, st = model.sigma(gs, x), model.sigma(gt, x)
        mean = x/ratio-variance/(ratio*st)*vector(model, x, t, z, spec)
        x = center(mean+std*ss/st*center(torch.randn(shape, generator=rng, device=device)))
    t = x.new_zeros(batch, 1)
    gamma = model.gamma(t)
    alpha, sigma = model.alpha(gamma, x), model.sigma(gamma, x)
    x = center((x-sigma*vector(model, x, t, z, spec))/alpha+
               sigma/alpha*center(torch.randn(shape, generator=rng, device=device)))
    return x*model.norm_values[0], initial
