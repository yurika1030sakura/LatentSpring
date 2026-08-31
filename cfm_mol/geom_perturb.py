"""Controlled geometric perturbations for the P3 "geometry-distance confound" check.

Motivation
----------
The headline BGFM Boltzmann-correlation eval perturbs each held-out molecule with
isotropic Gaussian noise, ``x* + sigma * eps``.  Every perturbation therefore has a
DIFFERENT distance from the reference geometry (||eps|| concentrates but still has
O(1/sqrt(3N)) relative spread, and for small molecules that spread is large).  Both
``log p_theta`` and ``E`` are monotone in that distance -- the model only has to
learn "further from the data manifold => lower density" to score a positive
per-group correlation.  That is *not* Boltzmann structure.

This module provides perturbation families that break the confound:

``fixed_rmsd``   random direction, then rescaled so the Kabsch-aligned RMSD to the
                 reference is EXACTLY the requested value.  Within a group the
                 geometry-distance is constant by construction, so any surviving
                 correlation between log p and -E is direction-selective, i.e. the
                 model is resolving the local energy landscape, not the radius.
``torsion``      rotate a random subset of rotatable (acyclic, non-terminal) bonds.
                 Bond lengths and bond angles are exactly preserved.
``bond_angle``   perturb bond angles only; bond lengths exactly preserved.
``normal_mode``  displace along low-frequency normal modes of an anisotropic
                 network model (or a real xTB Hessian when available).

All of them optionally accept ``match_rmsd`` so that a torsional / angular /
normal-mode perturbation ALSO lands on a fixed RMSD shell -- the strongest form
of the control (same distance, same internal-coordinate family, different
direction).

Everything here is pure numpy: no torch, no DGL, no RDKit.  That is deliberate --
it must be testable on a login node in either conda env, and it must not fall over
on transition-metal / radical / hypervalent systems where ``xyz2mol`` refuses to
assign bond orders.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
from typing import Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Element data (index = Z, entry 0 is a dummy)
# ---------------------------------------------------------------------------

# Cordero (2008) covalent radii in Angstrom.
_COV_R = [
    0.00,
    0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58,
    1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06,
    2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22,
    1.22, 1.20, 1.19, 1.20, 1.20, 1.16,
    2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44,
    1.42, 1.39, 1.39, 1.38, 1.39, 1.40,
    2.44, 2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92,
    1.92, 1.89, 1.90, 1.87, 1.87,
    1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32,
    1.45, 1.46, 1.48, 1.40, 1.50, 1.50,
    2.60, 2.21, 2.15, 2.06, 2.00, 1.96, 1.90, 1.87, 1.80, 1.69,
]

# Standard atomic weights (amu), same indexing.
_MASS = [
    0.0,
    1.008, 4.003, 6.941, 9.012, 10.811, 12.011, 14.007, 15.999, 18.998, 20.180,
    22.990, 24.305, 26.982, 28.086, 30.974, 32.065, 35.453, 39.948,
    39.098, 40.078, 44.956, 47.867, 50.942, 51.996, 54.938, 55.845, 58.933,
    58.693, 63.546, 65.380,
    69.723, 72.640, 74.922, 78.960, 79.904, 83.798,
    85.468, 87.620, 88.906, 91.224, 92.906, 95.960, 98.000, 101.070, 102.906,
    106.420, 107.868, 112.411,
    114.818, 118.710, 121.760, 127.600, 126.904, 131.293,
    132.905, 137.327, 138.905, 140.116, 140.908, 144.242, 145.000, 150.360,
    151.964, 157.250, 158.925, 162.500, 164.930, 167.259, 168.934, 173.054,
    174.967,
    178.490, 180.948, 183.840, 186.207, 190.230, 192.217, 195.084, 196.967,
    200.590,
    204.383, 207.200, 208.980, 209.000, 210.000, 222.000,
    223.000, 226.000, 227.000, 232.038, 231.036, 238.029, 237.000, 244.000,
    243.000, 247.000,
]


def covalent_radii(z: np.ndarray) -> np.ndarray:
    tab = np.asarray(_COV_R, dtype=float)
    zz = np.clip(np.asarray(z, dtype=int), 0, tab.size - 1)
    r = tab[zz]
    r[r <= 0.0] = 1.0
    return r


def atomic_masses(z: np.ndarray) -> np.ndarray:
    tab = np.asarray(_MASS, dtype=float)
    zz = np.clip(np.asarray(z, dtype=int), 0, tab.size - 1)
    m = tab[zz]
    m[m <= 0.0] = 12.0
    return m


# ---------------------------------------------------------------------------
# RMSD helpers
# ---------------------------------------------------------------------------

def kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    """Optimally (translation + rotation) aligned RMSD between two N x 3 sets.

    RMSD = sqrt( (1/N) sum_i ||p_i - R q_i - t||^2 ), minimised over R in SO(3).
    Reflections are excluded (the ``d`` sign fix), so a mirror image is NOT
    reported as identical.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    assert p.shape == q.shape and p.ndim == 2 and p.shape[1] == 3
    n = p.shape[0]
    if n == 0:
        return 0.0
    pc = p - p.mean(axis=0, keepdims=True)
    qc = q - q.mean(axis=0, keepdims=True)
    h = qc.T @ pc
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    dmat = np.diag([1.0, 1.0, d])
    rot = vt.T @ dmat @ u.T
    diff = pc - qc @ rot.T
    return float(np.sqrt((diff ** 2).sum() / n))


def rigid_body_basis(pos: np.ndarray, masses: np.ndarray | None = None) -> np.ndarray:
    """Orthonormal basis (6 x 3N, fewer if degenerate) of rigid-body motions.

    Columns of the returned matrix span translations + infinitesimal rotations
    about the (mass-weighted, if given) centroid.  Used to strip the components
    of a displacement that a Kabsch alignment would remove anyway -- without this
    a "random direction" wastes part of its magnitude on motion that produces
    zero RMSD, which would make the fixed-RMSD rescaling ill-conditioned.
    """
    pos = np.asarray(pos, dtype=float)
    n = pos.shape[0]
    w = np.ones(n) if masses is None else np.asarray(masses, dtype=float)
    com = (pos * w[:, None]).sum(axis=0) / w.sum()
    r = pos - com
    cols = []
    for ax in range(3):
        t = np.zeros((n, 3)); t[:, ax] = 1.0
        cols.append(t.reshape(-1))
    for ax in range(3):
        e = np.zeros(3); e[ax] = 1.0
        cols.append(np.cross(e[None, :], r).reshape(-1))
    m = np.stack(cols, axis=1)                       # (3N, 6)
    q, r_ = np.linalg.qr(m)
    keep = np.abs(np.diag(r_)) > 1e-8
    return q[:, keep]


def project_out_rigid(pos: np.ndarray, disp: np.ndarray,
                      masses: np.ndarray | None = None) -> np.ndarray:
    """Remove translation + infinitesimal-rotation components from ``disp``."""
    basis = rigid_body_basis(pos, masses)
    v = np.asarray(disp, dtype=float).reshape(-1)
    v = v - basis @ (basis.T @ v)
    return v.reshape(disp.shape)


def scale_to_rmsd(base: np.ndarray, make_pos, target_rmsd: float,
                  s0: float = 1.0, tol: float = 1e-6, max_iter: int = 60,
                  s_lo: float = 0.0, s_hi: float | None = None) -> tuple[np.ndarray, float, float]:
    """Find scale ``s`` with ``kabsch_rmsd(make_pos(s), base) == target_rmsd``.

    ``make_pos(s)`` returns an N x 3 geometry; it must be continuous in ``s`` with
    ``make_pos(0) == base``.  For a pure Cartesian displacement RMSD is exactly
    linear in ``s`` and this converges in one step; for torsion / bond-angle
    perturbations it is monotone-but-curved, so we bracket then bisect.

    Returns ``(positions, s, achieved_rmsd)``.
    """
    if target_rmsd <= 0:
        return np.asarray(base, dtype=float).copy(), 0.0, 0.0

    def f(s):
        return kabsch_rmsd(make_pos(s), base)

    # 1. Newton-ish secant from the (exact for linear) initial guess.
    s = float(s0)
    r = f(s)
    for _ in range(6):
        if r < 1e-12:
            s *= 2.0
            r = f(s)
            continue
        if abs(r - target_rmsd) <= tol * max(target_rmsd, 1e-9):
            return make_pos(s), s, r
        s = s * (target_rmsd / r)
        r = f(s)
    if abs(r - target_rmsd) <= 1e-4 * max(target_rmsd, 1e-9):
        return make_pos(s), s, r

    # 2. Robust fallback: bracket then bisect (handles the periodic / curved
    #    torsion case where the secant step can overshoot past a turning point).
    hi = s if s > 0 else 1.0
    if s_hi is not None:
        hi = min(hi, s_hi)
    n_expand = 0
    while f(hi) < target_rmsd and n_expand < 40:
        hi *= 1.6
        n_expand += 1
        if s_hi is not None and hi >= s_hi:
            hi = s_hi
            break
    lo = s_lo
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        rm = f(mid)
        if abs(rm - target_rmsd) <= tol * max(target_rmsd, 1e-9):
            return make_pos(mid), mid, rm
        if rm < target_rmsd:
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    return make_pos(mid), mid, f(mid)


# ---------------------------------------------------------------------------
# Connectivity (covalent-radius graph; no bond orders, no RDKit)
# ---------------------------------------------------------------------------

def build_adjacency(z: np.ndarray, pos: np.ndarray, tol: float = 1.25,
                    max_bonds_per_atom: int = 8) -> list[list[int]]:
    """Distance-based connectivity: bond iff d_ij < tol * (r_i + r_j).

    Deliberately bond-order free: OMol25 covers transition-metal, radical and
    hypervalent species for which valence-based perception (xyz2mol) fails.  We
    only need the *graph*, to decide which dihedrals exist.
    """
    pos = np.asarray(pos, dtype=float)
    n = pos.shape[0]
    r = covalent_radii(z)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    cut = tol * (r[:, None] + r[None, :])
    np.fill_diagonal(cut, -1.0)
    adj: list[list[int]] = [[] for _ in range(n)]
    ii, jj = np.nonzero(d < cut)
    for i, j in zip(ii.tolist(), jj.tolist()):
        if i < j:
            adj[i].append(j)
            adj[j].append(i)
    # Guard against pathological over-connection on collapsed geometries.
    for i in range(n):
        if len(adj[i]) > max_bonds_per_atom:
            nb = sorted(adj[i], key=lambda j: d[i, j])[:max_bonds_per_atom]
            adj[i] = nb
    # Re-symmetrise after trimming.
    sym = [set() for _ in range(n)]
    for i in range(n):
        for j in adj[i]:
            sym[i].add(j); sym[j].add(i)
    return [sorted(s) for s in sym]


def _component_excluding_edge(adj: list[list[int]], start: int,
                              blocked: tuple[int, int]) -> set[int]:
    """Atoms reachable from ``start`` without traversing the edge ``blocked``."""
    a, b = blocked
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if (u == a and v == b) or (u == b and v == a):
                continue
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def rotatable_bonds(adj: list[list[int]]) -> list[tuple[int, int, list[int]]]:
    """Acyclic, non-terminal bonds (i, j) plus the atom list on j's side.

    A bond is rotatable here iff it is a *bridge* (removing it disconnects the
    molecule -- i.e. it is not in a ring) and both endpoints have degree >= 2
    (otherwise the rotation is a no-op).  Bond order is not consulted: rotating a
    formal double bond costs energy, which is exactly the kind of signal we want
    the eval to probe, and OMol25 has no bond orders anyway.
    """
    n = len(adj)
    out = []
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            if len(adj[i]) < 2 or len(adj[j]) < 2:
                continue
            side_j = _component_excluding_edge(adj, j, (i, j))
            if i in side_j:
                continue                        # ring bond -> not rotatable
            if len(side_j) == 0 or len(side_j) == n:
                continue
            out.append((i, j, sorted(side_j)))
    return out


def bond_angle_triples(adj: list[list[int]]) -> list[tuple[int, int, int, list[int]]]:
    """(i, j, k, movable) with i-j-k bonded and j-k a bridge, so k's side moves."""
    n = len(adj)
    out = []
    for j in range(n):
        nbrs = adj[j]
        if len(nbrs) < 2:
            continue
        for a in range(len(nbrs)):
            for b in range(len(nbrs)):
                if a == b:
                    continue
                i, k = nbrs[a], nbrs[b]
                side_k = _component_excluding_edge(adj, k, (j, k))
                if j in side_k:
                    continue                    # ring bond
                if not side_k or len(side_k) >= n:
                    continue
                out.append((i, j, k, sorted(side_k)))
    return out


def _rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    nrm = np.linalg.norm(axis)
    if nrm < 1e-12:
        return np.eye(3)
    u = axis / nrm
    c, s = math.cos(theta), math.sin(theta)
    ux, uy, uz = u
    return np.array([
        [c + ux * ux * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
        [uy * ux * (1 - c) + uz * s, c + uy * uy * (1 - c), uy * uz * (1 - c) - ux * s],
        [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz * uz * (1 - c)],
    ])


# ---------------------------------------------------------------------------
# Perturbation families.  Each returns (positions, info_dict).
# ---------------------------------------------------------------------------

def perturb_gaussian(rng: np.random.Generator, base: np.ndarray, sigma: float,
                     **_kw) -> tuple[np.ndarray, dict]:
    """Reference behaviour: isotropic i.i.d. Gaussian displacement (Angstrom)."""
    d = rng.standard_normal(base.shape) * float(sigma)
    pos = base + d
    return pos, {"scale_used": float(sigma)}


def perturb_fixed_rmsd(rng: np.random.Generator, base: np.ndarray, rmsd: float,
                       masses: np.ndarray | None = None,
                       **_kw) -> tuple[np.ndarray, dict]:
    """Random Cartesian direction rescaled to hit ``rmsd`` EXACTLY.

    Rigid-body components are projected out first, so the requested RMSD is
    achieved by genuine internal deformation rather than by a translation/rotation
    that Kabsch would undo.  After projection RMSD is exactly linear in the scale
    so the rescale is closed-form (``scale_to_rmsd`` still verifies it).
    """
    d = rng.standard_normal(base.shape)
    d = project_out_rigid(base, d, masses)
    nrm = np.linalg.norm(d)
    if nrm < 1e-12:
        d = rng.standard_normal(base.shape)
        d = project_out_rigid(base, d, masses)
        nrm = max(np.linalg.norm(d), 1e-12)
    n = base.shape[0]
    # ||s*d||/sqrt(N) == rmsd  (exact once rigid modes are removed)
    d = d * (float(rmsd) * math.sqrt(n) / nrm)
    pos, s, achieved = scale_to_rmsd(base, lambda s: base + s * d, float(rmsd), s0=1.0)
    return pos, {"scale_used": float(s), "direction_norm": float(nrm)}


def perturb_torsion(rng: np.random.Generator, base: np.ndarray, z: np.ndarray,
                    sigma_deg: float = 30.0, n_torsions: int = 2,
                    adj: list[list[int]] | None = None,
                    match_rmsd: float | None = None,
                    bond_tol: float = 1.25, **_kw) -> tuple[np.ndarray, dict]:
    """Rotate a random subset of rotatable dihedrals; bond lengths/angles exact."""
    if adj is None:
        adj = build_adjacency(z, base, tol=bond_tol)
    rots = rotatable_bonds(adj)
    if not rots:
        return None, {"failed": "no_rotatable_bonds"}
    k = min(int(n_torsions), len(rots))
    pick = rng.choice(len(rots), size=k, replace=False)
    angles = rng.standard_normal(k) * math.radians(float(sigma_deg))
    chosen = [rots[int(p)] for p in pick]

    def make(s: float) -> np.ndarray:
        pos = base.copy()
        for (i, j, side), ang in zip(chosen, angles):
            axis = pos[j] - pos[i]
            rot = _rotation_matrix(axis, float(ang) * float(s))
            idx = np.asarray(side, dtype=int)
            pos[idx] = (pos[idx] - pos[j]) @ rot.T + pos[j]
        return pos

    if match_rmsd is not None and match_rmsd > 0:
        pos, s, achieved = scale_to_rmsd(base, make, float(match_rmsd), s0=1.0, s_hi=200.0)
        info = {"scale_used": float(s), "n_torsions": k,
                "torsion_deg": [float(math.degrees(a * s)) for a in angles]}
    else:
        pos = make(1.0)
        info = {"scale_used": 1.0, "n_torsions": k,
                "torsion_deg": [float(math.degrees(a)) for a in angles]}
    info["bonds_rotated"] = [[int(i), int(j)] for (i, j, _s) in chosen]
    return pos, info


def perturb_bond_angle(rng: np.random.Generator, base: np.ndarray, z: np.ndarray,
                       sigma_deg: float = 5.0, n_angles: int = 2,
                       adj: list[list[int]] | None = None,
                       match_rmsd: float | None = None,
                       bond_tol: float = 1.25, **_kw) -> tuple[np.ndarray, dict]:
    """Perturb bond angles only; every bond LENGTH is preserved exactly."""
    if adj is None:
        adj = build_adjacency(z, base, tol=bond_tol)
    triples = bond_angle_triples(adj)
    if not triples:
        return None, {"failed": "no_bond_angles"}
    k = min(int(n_angles), len(triples))
    pick = rng.choice(len(triples), size=k, replace=False)
    angles = rng.standard_normal(k) * math.radians(float(sigma_deg))
    chosen = [triples[int(p)] for p in pick]
    # Pre-draw fallback axes so make(s) is deterministic in s.
    fallback = rng.standard_normal((k, 3))

    def make(s: float) -> np.ndarray:
        pos = base.copy()
        for t, ((i, j, kk, side), ang) in enumerate(zip(chosen, angles)):
            v1 = pos[i] - pos[j]
            v2 = pos[kk] - pos[j]
            axis = np.cross(v1, v2)
            if np.linalg.norm(axis) < 1e-8:      # near-linear i-j-k
                axis = np.cross(v2, fallback[t])
                if np.linalg.norm(axis) < 1e-8:
                    continue
            rot = _rotation_matrix(axis, float(ang) * float(s))
            idx = np.asarray(side, dtype=int)
            pos[idx] = (pos[idx] - pos[j]) @ rot.T + pos[j]
        return pos

    if match_rmsd is not None and match_rmsd > 0:
        pos, s, _a = scale_to_rmsd(base, make, float(match_rmsd), s0=1.0, s_hi=200.0)
        info = {"scale_used": float(s), "n_angles": k}
    else:
        pos = make(1.0)
        info = {"scale_used": 1.0, "n_angles": k}
    info["angles_perturbed"] = [[int(i), int(j), int(kk)] for (i, j, kk, _s) in chosen]
    return pos, info


# --- normal modes ----------------------------------------------------------

def anm_modes(z: np.ndarray, pos: np.ndarray, cutoff: float = 8.0,
              gamma_power: float = 2.0, mass_weight: bool = True,
              n_modes: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Low-frequency modes of an Anisotropic Network Model (Hessian-free proxy).

    ANM Hessian: for every pair within ``cutoff`` a harmonic spring with constant
    gamma_ij = (d0_ij)^(-gamma_power) acting along the pair axis.  This reproduces
    the soft collective directions of a real Hessian well enough for a *control*
    experiment (and unlike ``xtb --hess`` it costs microseconds, so it scales to
    hundreds of molecules).  Use ``xtb_modes`` when a true Hessian is wanted.

    Returns ``(eigvals, eigvecs)`` with eigvecs of shape (3N, n_modes), sorted by
    increasing eigenvalue, with the 6 rigid-body modes removed.  If ``mass_weight``
    the modes are returned in CARTESIAN space (already un-mass-weighted).
    """
    pos = np.asarray(pos, dtype=float)
    n = pos.shape[0]
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    h = np.zeros((3 * n, 3 * n))
    for i in range(n):
        for j in range(i + 1, n):
            dij = d[i, j]
            if dij > cutoff or dij < 1e-6:
                continue
            u = (pos[j] - pos[i]) / dij
            g = dij ** (-gamma_power)
            blk = g * np.outer(u, u)
            h[3 * i:3 * i + 3, 3 * j:3 * j + 3] -= blk
            h[3 * j:3 * j + 3, 3 * i:3 * i + 3] -= blk
            h[3 * i:3 * i + 3, 3 * i:3 * i + 3] += blk
            h[3 * j:3 * j + 3, 3 * j:3 * j + 3] += blk
    if mass_weight:
        m = atomic_masses(z)
        w = 1.0 / np.sqrt(np.repeat(m, 3))
        h = h * w[:, None] * w[None, :]
    evals, evecs = np.linalg.eigh(h)
    if mass_weight:
        evecs = evecs * w[:, None]               # back to Cartesian
        evecs = evecs / np.maximum(np.linalg.norm(evecs, axis=0, keepdims=True), 1e-12)
    # Drop rigid-body (near-zero) modes robustly by projecting them out.
    basis = rigid_body_basis(pos)
    resid = np.linalg.norm(evecs - basis @ (basis.T @ evecs), axis=0)
    keep = resid > 0.5
    evals, evecs = evals[keep], evecs[:, keep]
    order = np.argsort(evals)
    evals, evecs = evals[order], evecs[:, order]
    k = min(int(n_modes), evecs.shape[1])
    return evals[:k], evecs[:, :k]


_XTB_HESS_HEADER = re.compile(r"^\s*\$hessian", re.I)


def xtb_modes(z: np.ndarray, pos: np.ndarray, charge: int = 0,
              n_modes: int = 12, timeout: int = 900,
              xtb_bin: str | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """True mass-weighted normal modes from ``xtb --hess``.  None if xtb fails.

    Cost is O(minutes) per molecule for 40-60 atoms, so this is opt-in; the ANM
    proxy is the default because the control only needs *a* physically-shaped
    soft-mode subspace, not spectroscopic accuracy.
    """
    xtb = xtb_bin or shutil.which("xtb")
    if xtb is None:
        return None
    from_z = _symbols_from_z(z)
    n = len(from_z)
    with tempfile.TemporaryDirectory() as td:
        xyz = os.path.join(td, "m.xyz")
        with open(xyz, "w") as f:
            f.write(f"{n}\nctrl\n")
            for sym, (x, y, zz) in zip(from_z, np.asarray(pos, dtype=float)):
                f.write(f"{sym} {x:.8f} {y:.8f} {zz:.8f}\n")
        cmd = [xtb, xyz, "--hess", "--gfn", "2", "--chrg", str(int(charge)), "--silent"]
        try:
            subprocess.run(cmd, cwd=td, capture_output=True, text=True, timeout=timeout)
        except (subprocess.TimeoutExpired, OSError):
            return None
        hpath = os.path.join(td, "hessian")
        if not os.path.exists(hpath):
            return None
        vals: list[float] = []
        with open(hpath) as f:
            for line in f:
                if _XTB_HESS_HEADER.match(line) or line.strip().startswith("$"):
                    continue
                for tok in line.split():
                    try:
                        vals.append(float(tok))
                    except ValueError:
                        pass
        if len(vals) < (3 * n) ** 2:
            return None
        h = np.asarray(vals[: (3 * n) ** 2], dtype=float).reshape(3 * n, 3 * n)
    h = 0.5 * (h + h.T)
    m = atomic_masses(z)
    w = 1.0 / np.sqrt(np.repeat(m, 3))
    hm = h * w[:, None] * w[None, :]
    evals, evecs = np.linalg.eigh(hm)
    evecs = evecs * w[:, None]
    evecs = evecs / np.maximum(np.linalg.norm(evecs, axis=0, keepdims=True), 1e-12)
    basis = rigid_body_basis(pos)
    resid = np.linalg.norm(evecs - basis @ (basis.T @ evecs), axis=0)
    keep = resid > 0.5
    evals, evecs = evals[keep], evecs[:, keep]
    order = np.argsort(evals)
    evals, evecs = evals[order], evecs[:, order]
    k = min(int(n_modes), evecs.shape[1])
    return evals[:k], evecs[:, :k]


def perturb_normal_mode(rng: np.random.Generator, base: np.ndarray, z: np.ndarray,
                        rmsd: float | None = None, sigma: float | None = None,
                        modes: tuple[np.ndarray, np.ndarray] | None = None,
                        thermal: bool = True, **_kw) -> tuple[np.ndarray, dict]:
    """Displace along a random thermal-ish combination of low-frequency modes.

    Amplitude of mode k is drawn ~ N(0, 1/sqrt(lambda_k)) when ``thermal`` (soft
    modes dominate, as in a Boltzmann ensemble), then the whole displacement is
    rescaled to the requested RMSD (or scaled by ``sigma`` if no RMSD is given).
    """
    if modes is None:
        modes = anm_modes(z, base)
    evals, evecs = modes
    if evecs.shape[1] == 0:
        return None, {"failed": "no_modes"}
    if thermal:
        amp = 1.0 / np.sqrt(np.maximum(evals, np.max(evals) * 1e-6))
    else:
        amp = np.ones_like(evals)
    c = rng.standard_normal(evecs.shape[1]) * amp
    d = (evecs @ c).reshape(base.shape)
    d = project_out_rigid(base, d)
    nrm = np.linalg.norm(d)
    if nrm < 1e-12:
        return None, {"failed": "degenerate_mode_mix"}
    if rmsd is not None and rmsd > 0:
        d = d * (float(rmsd) * math.sqrt(base.shape[0]) / nrm)
        pos, s, _a = scale_to_rmsd(base, lambda s: base + s * d, float(rmsd), s0=1.0)
        return pos, {"scale_used": float(s), "n_modes": int(evecs.shape[1])}
    d = d / nrm * float(sigma or 0.1) * math.sqrt(base.shape[0])
    return base + d, {"scale_used": float(sigma or 0.1), "n_modes": int(evecs.shape[1])}


_SYMS = [
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
    "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm",
]


def _symbols_from_z(z: Iterable[int]) -> list[str]:
    return [_SYMS[int(i)] if 0 < int(i) < len(_SYMS) else "C" for i in z]


PERTURB_MODES = ("gaussian", "fixed_rmsd", "torsion", "bond_angle", "normal_mode")
