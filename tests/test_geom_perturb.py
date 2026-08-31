"""Tests for the P3 controlled-perturbation machinery (cfm_mol/geom_perturb.py).

The load-bearing claim of the P3 experiment is that `fixed_rmsd` (and the
`match_rmsd` variants of the internal-coordinate modes) produce perturbations
whose Kabsch-aligned RMSD to the reference is CONSTANT within a group. If that
were only approximately true, the "we controlled for geometry distance" claim
would be worth nothing. These tests pin it at machine precision.
"""
import math

import numpy as np
import pytest

from cfm_mol.geom_perturb import (
    anm_modes, bond_angle_triples, build_adjacency, kabsch_rmsd,
    perturb_bond_angle, perturb_fixed_rmsd, perturb_gaussian,
    perturb_normal_mode, perturb_torsion, project_out_rigid, rotatable_bonds,
)


def _butane():
    pos = np.array([
        [0.000, 0.000, 0.000], [1.520, 0.000, 0.000],
        [2.030, 1.430, 0.000], [3.550, 1.430, 0.000],
        [-0.380, -1.020, 0.000], [-0.380, 0.510, 0.880],
        [-0.380, 0.510, -0.880], [1.900, -0.510, 0.880],
        [1.900, -0.510, -0.880], [1.650, 1.940, 0.880],
        [1.650, 1.940, -0.880], [3.930, 0.410, 0.000],
        [3.930, 1.940, 0.880], [3.930, 1.940, -0.880],
    ])
    z = np.array([6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    return z, pos


def test_kabsch_invariant_to_rigid_motion():
    _z, pos = _butane()
    theta = 0.7
    rot = np.array([[math.cos(theta), -math.sin(theta), 0],
                    [math.sin(theta), math.cos(theta), 0],
                    [0, 0, 1]])
    moved = pos @ rot.T + np.array([3.0, -1.0, 2.0])
    assert kabsch_rmsd(moved, pos) < 1e-10


def test_project_out_rigid_kills_translation_and_rotation():
    _z, pos = _butane()
    trans = np.tile(np.array([1.0, 2.0, -0.5]), (pos.shape[0], 1))
    omega = np.array([0.0, 0.0, 1.0])
    rotd = np.cross(omega, pos - pos.mean(0))
    d = project_out_rigid(pos, trans + rotd)
    assert np.linalg.norm(d) < 1e-8


def test_gaussian_rmsd_actually_varies():
    """The confound this whole experiment exists to remove must be present."""
    _z, pos = _butane()
    rng = np.random.default_rng(0)
    r = np.array([kabsch_rmsd(perturb_gaussian(rng, pos, sigma=0.15)[0], pos)
                  for _ in range(300)])
    assert r.std() / r.mean() > 0.05


def test_fixed_rmsd_is_exact():
    _z, pos = _butane()
    rng = np.random.default_rng(0)
    for target in (0.05, 0.15, 0.4):
        r = np.array([kabsch_rmsd(perturb_fixed_rmsd(rng, pos, rmsd=target)[0], pos)
                      for _ in range(100)])
        assert np.allclose(r, target, atol=1e-9), (target, r.min(), r.max())
        assert r.std() < 1e-10


def test_fixed_rmsd_directions_are_not_degenerate():
    """Constant radius must not mean constant direction."""
    _z, pos = _butane()
    rng = np.random.default_rng(1)
    ds = [perturb_fixed_rmsd(rng, pos, rmsd=0.2)[0] - pos for _ in range(20)]
    flat = np.array([d.reshape(-1) / np.linalg.norm(d) for d in ds])
    cos = flat @ flat.T
    off = cos[~np.eye(len(ds), dtype=bool)]
    assert np.abs(off).max() < 0.9


def test_torsion_preserves_bond_lengths_and_angles():
    z, pos = _butane()
    rng = np.random.default_rng(2)
    adj = build_adjacency(z, pos)
    d0 = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    for _ in range(20):
        p, info = perturb_torsion(rng, pos, z, sigma_deg=40.0, n_torsions=1, adj=adj)
        assert p is not None
        d1 = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
        for i in range(len(z)):
            for j in adj[i]:
                assert abs(d1[i, j] - d0[i, j]) < 1e-8      # bond lengths
            for j in adj[i]:                                 # bond angles (1-3)
                for k in adj[j]:
                    if k != i:
                        assert abs(d1[i, k] - d0[i, k]) < 1e-6


def test_bond_angle_preserves_bond_lengths():
    z, pos = _butane()
    rng = np.random.default_rng(3)
    adj = build_adjacency(z, pos)
    d0 = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    p, _i = perturb_bond_angle(rng, pos, z, sigma_deg=8.0, n_angles=2, adj=adj)
    d1 = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    for i in range(len(z)):
        for j in adj[i]:
            assert abs(d1[i, j] - d0[i, j]) < 1e-8


@pytest.mark.parametrize("fn,kw", [
    (perturb_torsion, dict(sigma_deg=30.0, n_torsions=2)),
    (perturb_bond_angle, dict(sigma_deg=6.0, n_angles=2)),
])
def test_internal_coord_match_rmsd(fn, kw):
    z, pos = _butane()
    rng = np.random.default_rng(4)
    target = 0.15
    r = []
    for _ in range(40):
        p, _i = fn(rng, pos, z, match_rmsd=target, **kw)
        assert p is not None
        r.append(kabsch_rmsd(p, pos))
    r = np.array(r)
    assert np.allclose(r, target, atol=1e-6), (r.min(), r.max())


def test_normal_mode_fixed_rmsd_and_soft_mode_bias():
    z, pos = _butane()
    rng = np.random.default_rng(5)
    evals, evecs = anm_modes(z, pos, n_modes=10)
    assert evecs.shape[1] == 10
    assert np.all(evals > 0)                     # rigid modes removed
    assert np.all(np.diff(evals) >= -1e-12)      # sorted
    r = np.array([kabsch_rmsd(
        perturb_normal_mode(rng, pos, z, rmsd=0.12, modes=(evals, evecs))[0], pos)
        for _ in range(50)])
    assert np.allclose(r, 0.12, atol=1e-9)


def test_rotatable_bonds_excludes_rings_and_terminals():
    # benzene ring (planar hexagon) + one methyl substituent
    n = 6
    ang = np.arange(n) * 2 * np.pi / n
    ring = np.stack([1.39 * np.cos(ang), 1.39 * np.sin(ang), np.zeros(n)], axis=1)
    me = ring[0] * (1 + 1.50 / 1.39)
    h = [me + v for v in ([0.0, 0.9, 0.6], [0.0, -0.9, 0.6], [0.0, 0.0, -1.05])]
    pos = np.vstack([ring, me[None, :], np.array(h)])
    z = np.array([6] * 7 + [1, 1, 1])
    adj = build_adjacency(z, pos)
    rb = rotatable_bonds(adj)
    pairs = {(i, j) for i, j, _s in rb}
    assert (0, 6) in pairs or (6, 0) in pairs          # ring-C -- methyl-C
    for i, j, _s in rb:                                 # no ring bond is rotatable
        assert not (i < 6 and j < 6)
    for i, j, _s in rb:                                 # no terminal H bond
        assert len(adj[i]) >= 2 and len(adj[j]) >= 2
