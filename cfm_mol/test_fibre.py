"""Sanity tests for domain.py, fibre.py, flow.py.

Run with: python -m cfm_mol.test_fibre
"""
import torch

from cfm_mol.domain import default_d_min_table, steric_all_ok
from cfm_mol.fibre import (
    euler_step_on_fibre,
    retract,
    sample_interpolant,
    sample_prior,
    tangent_project,
)
from cfm_mol.flow import LinearVelocityNet, cfm_loss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _two_carbon_bad():
    """Two carbons 0.3 Å apart -- below default d_min ~ 1.064 Å."""
    r = torch.tensor([[[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]]])
    a = torch.tensor([[1, 1]])  # MiDi atom-type 1 = C
    return r, a


def _two_carbon_good():
    """Two carbons 1.54 Å apart -- feasible."""
    r = torch.tensor([[[0.0, 0.0, 0.0], [1.54, 0.0, 0.0]]])
    a = torch.tensor([[1, 1]])
    return r, a


def _small_batch(B=8, N=8, seed=0):
    """A batch of B generic feasible molecules of N atoms."""
    torch.manual_seed(seed)
    r = torch.randn(B, N, 3) * 2.0
    a = torch.randint(1, 5, (B, N))               # C, N, O, F
    d_min = default_d_min_table(n_atom_types=10)
    r, _ = retract(r, a, d_min)
    return r, a, d_min


# ---------------------------------------------------------------------------
# retract
# ---------------------------------------------------------------------------

def test_retract_resolves_violation():
    r, a = _two_carbon_bad()
    d_min = default_d_min_table(n_atom_types=10)
    r_new, conv = retract(r, a, d_min)
    assert conv.all().item()
    d_after = (r_new[0, 0] - r_new[0, 1]).norm().item()
    assert d_after >= d_min[1, 1].item()
    print(f"ok: retract_resolves_violation (d: 0.3 -> {d_after:.3f})")


def test_retract_leaves_good_alone():
    r, a = _two_carbon_good()
    d_min = default_d_min_table(n_atom_types=10)
    r_new, conv = retract(r, a, d_min)
    assert conv.all().item()
    assert torch.allclose(r_new, r, atol=1e-6), "retract should be identity on feasible input"
    print("ok: retract_leaves_good_alone")


# ---------------------------------------------------------------------------
# tangent projection
# ---------------------------------------------------------------------------

def test_tangent_project_clips_approach():
    """Two atoms on the steric boundary, velocity pushing them together --
    tangent projection should zero out the approach component."""
    d_min = default_d_min_table(n_atom_types=10)
    d_CC = d_min[1, 1].item()
    r = torch.tensor([[[0.0, 0.0, 0.0], [d_CC, 0.0, 0.0]]])
    a = torch.tensor([[1, 1]])
    v = torch.tensor([[[+1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]])
    v_proj = tangent_project(v, r, a, d_min, margin=0.05)

    axis = (r[0, 0] - r[0, 1]) / (r[0, 0] - r[0, 1]).norm()
    rel_vel_along = ((v_proj[0, 0] - v_proj[0, 1]) * axis).sum().item()
    assert rel_vel_along >= -1e-6, f"rel_vel_along should be >=0 after clip, got {rel_vel_along}"
    print(f"ok: tangent_project_clips_approach (rel_vel_along: -2.0 -> {rel_vel_along:.3f})")


def test_tangent_project_preserves_free_velocity():
    """Two atoms far apart -> velocity unchanged."""
    r, a = _two_carbon_good()
    d_min = default_d_min_table(n_atom_types=10)
    v = torch.randn_like(r)
    v_proj = tangent_project(v, r, a, d_min, margin=0.05)
    assert torch.allclose(v_proj, v), "pair far from boundary -- projection should be identity"
    print("ok: tangent_project_preserves_free_velocity")


# ---------------------------------------------------------------------------
# Interpolant + Euler step
# ---------------------------------------------------------------------------

def test_interpolant_stays_on_fibre():
    r1, a, d_min = _small_batch(B=8, N=8)
    r0 = sample_prior(n_atoms=8, batch_size=8, a=a, d_min_table=d_min, scale=3.0)
    t = torch.rand(8)
    r_t, u = sample_interpolant(r0, r1, t, a, d_min)
    assert steric_all_ok(r_t, a, d_min).all().item(), "interpolant fell off the fibre"
    assert u.shape == r1.shape
    print("ok: interpolant_stays_on_fibre (8 random t in [0,1])")


def test_euler_step_stays_on_fibre():
    r, a, d_min = _small_batch(B=4, N=6)
    v = torch.randn_like(r) * 0.5
    r_next = euler_step_on_fibre(r, v, dt=0.1, a=a, d_min_table=d_min)
    assert steric_all_ok(r_next, a, d_min).all().item()
    print("ok: euler_step_stays_on_fibre")


# ---------------------------------------------------------------------------
# End-to-end CFM loss
# ---------------------------------------------------------------------------

def test_cfm_loss_runs_and_backprops():
    """End-to-end: tiny MLP velocity net + cfm_loss -> scalar loss -> gradient.

    Integration test for the whole training pipeline (minus the real network).
    """
    r1, a, d_min = _small_batch(B=4, N=6)
    b = torch.zeros(4, 6, 6, dtype=torch.long)     # placeholder bond orders
    net = LinearVelocityNet(hidden=16)
    loss = cfm_loss(net, r1, a, b, d_min)
    assert torch.isfinite(loss).item(), f"loss is not finite: {loss}"
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in net.parameters() if p.grad is not None)
    assert grad_norm > 0, "no gradient flowed into the velocity net"
    print(f"ok: cfm_loss_runs_and_backprops (loss={loss.item():.4f}, grad_norm={grad_norm:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Projection tests
# ---------------------------------------------------------------------------

def test_project_valence_fixes_methane():
    """Methane (CH4) with one H missing: valence of C is 3, should project to 4."""
    from cfm_mol.domain import DEFAULT_VALENCE_SETS
    from cfm_mol.projection import project_valence
    # Atom types: [C, H, H, H, H]; MiDi order has 1=C, 0=H.
    a = torch.tensor([[1, 0, 0, 0, 0]])
    # Bond matrix: C-H, C-H, C-H, but missing one C-H. Order 5x5 with indices:
    # b[0,1]=1 (C-H), b[0,2]=1, b[0,3]=1, b[0,4]=0 (missing)
    b = torch.zeros(1, 5, 5, dtype=torch.long)
    for i in range(1, 4):
        b[0, 0, i] = b[0, i, 0] = 1
    # MiDi atom-type 0=H, 1=C, 2=N...
    valence_table = {0: (1,), 1: (4,)}
    b_proj = project_valence(a, b, valence_table)
    carbon_valence = b_proj[0, 0].float().sum().item()
    assert carbon_valence == 4.0, f"carbon valence should be 4, got {carbon_valence}"
    print(f"ok: project_valence_fixes_methane (C valence 3 -> {int(carbon_valence)})")


def test_project_connectivity_merges_components():
    """Two disconnected C-C pairs -- projection adds a bridging bond."""
    from cfm_mol.projection import project_connectivity
    # 4 atoms, two C-C pairs: (0-1) and (2-3). No edges between.
    b = torch.zeros(1, 4, 4, dtype=torch.long)
    b[0, 0, 1] = b[0, 1, 0] = 1
    b[0, 2, 3] = b[0, 3, 2] = 1
    # Coordinates: atom 1 near atom 2, so they should be the bridge.
    r = torch.tensor([[[0.0, 0.0, 0.0],
                       [1.5, 0.0, 0.0],
                       [3.0, 0.0, 0.0],
                       [4.5, 0.0, 0.0]]])
    b_proj = project_connectivity(b, r=r)
    # After projection, graph should be connected; verify via BFS.
    from cfm_mol.domain import connectivity_ok
    conn = connectivity_ok(b_proj, torch.tensor([4]))
    assert conn.item(), "graph should be connected after projection"
    # The bridge should be atoms 1 and 2 (closest between components).
    assert b_proj[0, 1, 2].item() == 1, f"bridge bond 1-2 not added: {b_proj[0]}"
    print("ok: project_connectivity_merges_components")


def test_sample_coords_only_stays_feasible():
    """Full inference Euler loop -- all samples must be on the fibre."""
    from cfm_mol.sampling import sample_coords_only
    _, a, d_min = _small_batch(B=8, N=6)
    b = torch.zeros(8, 6, 6, dtype=torch.long)
    net = LinearVelocityNet(hidden=32)
    r_samples, validity = sample_coords_only(net, a, b, d_min, n_steps=20)
    assert validity == 1.0, f"100% of Euler samples should be on fibre, got {100*validity}%"
    print(f"ok: sample_coords_only_stays_feasible (validity={100*validity}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_retract_resolves_violation()
    test_retract_leaves_good_alone()
    test_tangent_project_clips_approach()
    test_tangent_project_preserves_free_velocity()
    test_interpolant_stays_on_fibre()
    test_euler_step_stays_on_fibre()
    test_cfm_loss_runs_and_backprops()
    test_project_valence_fixes_methane()
    test_project_connectivity_merges_components()
    test_sample_coords_only_stays_feasible()
    print("\nall tests passed.")
