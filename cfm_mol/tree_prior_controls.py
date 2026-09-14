"""Second-moment controls for the bond-free spatial tree prior.

Harmonic trees match the lognormal-edge tree's covariance exactly, conditional
on every tree. A single Gaussian uses a fixed Monte Carlo average of analytic
tree covariances, symmetrized over identical species. The latter covariance is
an approximation to the infinite tree mixture, not an exact moment identity.
"""
import hashlib
import heapq
import math

import numpy as np
import torch
from torch import nn

from cfm_mol.nonequilibrium import centered_orthonormal_basis
from cfm_mol.tree_mixture_prior import TreeMixturePrior, log_tree_partition, weighted_tree


def cayley_tree(propensities, rng):
    """Weighted Cayley law via iid Pruefer entries; a_ij = u_i u_j only."""
    n = len(propensities)
    if n == 1:
        return []
    code = rng.choice(n, size=n-2, p=propensities / np.sum(propensities))
    degree = np.bincount(code, minlength=n) + 1
    leaves = [i for i in range(n) if degree[i] == 1]
    heapq.heapify(leaves)
    edges = []
    for parent in code:
        child = heapq.heappop(leaves)
        edges.append((child, int(parent)))
        degree[child] -= 1
        degree[parent] -= 1
        if degree[parent] == 1:
            heapq.heappush(leaves, int(parent))
    edges.append(tuple(leaves))
    return edges


def edge_embedding(n, edges):
    """Map independent edge displacements to centered node coordinates."""
    adjacency = [[] for _ in range(n)]
    for e, (i, j) in enumerate(edges):
        adjacency[j].append((i, e, 1))
        adjacency[i].append((j, e, -1))
    matrix = np.zeros((n, max(0, n-1)), dtype=np.float64)
    seen = {0}
    stack = [0]
    while stack:
        parent = stack.pop()
        for child, edge, sign in adjacency[parent]:
            if child in seen:
                continue
            matrix[child] = matrix[parent]
            matrix[child, edge] = sign
            seen.add(child)
            stack.append(child)
    if len(edges) != n-1 or len(seen) != n:
        raise ValueError('A connected tree is required')
    return matrix - matrix.mean(0)


def symmetrize_species_covariance(covariance, numbers):
    """Exact group average over within-species permutations, without enumeration."""
    groups = [np.flatnonzero(np.asarray(numbers) == z) for z in sorted(set(numbers))]
    result = np.empty_like(covariance)
    for left in groups:
        for right in groups:
            block = covariance[np.ix_(left, right)]
            if np.array_equal(left, right):
                diagonal = np.trace(block) / len(left)
                off = (block.sum() - np.trace(block)) / (len(left)*(len(left)-1)) if len(left) > 1 else 0.
                result[np.ix_(left, left)] = off
                result[left, left] = diagonal
            else:
                result[np.ix_(left, right)] = block.mean()
    result = .5*(result + result.T)
    return result - result.mean(0)[None, :] - result.mean(1)[:, None] + result.mean()


class TreePriorControl(nn.Module):
    def __init__(self, control, width=.2, covariance_trees=128, covariance_seed=30191):
        super().__init__()
        if control not in ['harmonic_tree', 'covariance_gaussian']:
            raise ValueError('Unknown tree-prior control')
        if not 0 < width < 2 or covariance_trees < 1:
            raise ValueError('Positive covariance budget and supported edge width required')
        self.control = control
        self.width = width
        self.covariance_trees = int(covariance_trees)
        self.covariance_seed = int(covariance_seed)
        self.configuration = dict(control=control, width=width,
            covariance_trees=self.covariance_trees, covariance_seed=self.covariance_seed)
        self.base = TreeMixturePrior('fixed', width=width).double()
        self._cache = {}
        self.requires_grad_(False)

    def covariance_for(self, numbers, charge, spin):
        # The fixed affinity law does not use charge/spin, but validation does.
        self.base.parameters_for(numbers, charge, spin)
        numbers = np.asarray(numbers, dtype=np.int64)
        permutation = np.argsort(numbers, kind='stable')
        inverse = np.argsort(permutation)
        key = tuple(numbers[permutation].tolist())
        if key not in self._cache:
            n = len(key)
            if n < 2:
                self._cache[key] = torch.zeros((n, n), dtype=torch.float64)
            else:
                _, length = self.base.parameters_for(key, charge, spin)
                variance = length.detach().cpu().numpy()**2 * math.exp(self.width**2) / 3
                u = self.base.base_log_propensity[list(key)].exp().detach().cpu().numpy()
                seed = int.from_bytes(hashlib.sha256(repr((key, self.covariance_seed)).encode()).digest()[:8], 'little')
                rng = np.random.default_rng(seed)
                covariance = np.zeros((n, n))
                for _ in range(self.covariance_trees):
                    edges = cayley_tree(u, rng)
                    embedding = edge_embedding(n, edges)
                    edge_variance = np.asarray([variance[i, j] for i, j in edges])
                    covariance += (embedding * edge_variance) @ embedding.T
                covariance = symmetrize_species_covariance(covariance / self.covariance_trees, key)
                self._cache[key] = torch.from_numpy(covariance)
        return self._cache[key][inverse][:, inverse].clone()

    def _factor(self, numbers, charge, spin):
        covariance = self.covariance_for(numbers, charge, spin)
        basis = centered_orthonormal_basis(len(numbers))
        cholesky = torch.linalg.cholesky(basis.T @ covariance @ basis)
        return basis, cholesky

    @torch.no_grad()
    def sample(self, numbers, charge, spin, *, rng, generator):
        a, length = self.base.parameters_for(numbers, charge, spin)
        n = len(numbers)
        if n == 1:
            return length.new_zeros((1, 3)), []
        if self.control == 'covariance_gaussian':
            basis, cholesky = self._factor(numbers, charge, spin)
            z = torch.randn((n-1, 3), dtype=torch.float64, generator=generator)
            return basis @ cholesky @ z, []
        edges = weighted_tree(a.detach().cpu().numpy(), rng)
        std = torch.stack([length[i, j] for i, j in edges]) * math.exp(.5*self.width**2) / math.sqrt(3)
        z = torch.randn((n-1, 3), dtype=length.dtype, generator=generator)
        return torch.from_numpy(edge_embedding(n, edges)).to(z) @ (std[:, None] * z), edges

    def log_prob(self, x, numbers, charge, spin):
        a, length = self.base.parameters_for(numbers, charge, spin)
        n = len(numbers)
        if x.shape != (n, 3) or not torch.isfinite(x).all() or x.mean(0).abs().max() > 1e-5:
            raise ValueError('Finite COM-centered coordinates required')
        if n == 1:
            return x.new_zeros(())
        if self.control == 'covariance_gaussian':
            basis, cholesky = self._factor(numbers, charge, spin)
            basis, cholesky = basis.to(x), cholesky.to(x)
            z = torch.linalg.solve_triangular(cholesky, basis.T @ x, upper=False)
            return -.5*z.square().sum() - 3*cholesky.diag().log().sum() - 1.5*(n-1)*math.log(2*math.pi)
        variance = length.to(x).square() * math.exp(self.width**2) / 3
        distance2 = (x[:, None] - x[None, :]).square().sum(-1)
        log_h = -.5*distance2/variance - 1.5*torch.log(2*math.pi*variance)
        return 1.5*math.log(n) + log_tree_partition(a.to(x) + log_h) - log_tree_partition(a.to(x))
